"""Experimental tracking + Smart Tagging + Viterbi possession estimator.

No ground truth or baseline prediction files are read. Defaults are engineering
hypotheses, not fitted parameters. Superseded by solution/possession_v2.py (see README.md).
"""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
import unicodedata

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from games import GAME, GAMES, half_video_blocks, require
from starter.baseline import team_names, team_of_each_track, write_submission
from starter.load_smart_tagging import load as load_tags

LABELS = np.array(['H', 'A', 'D'])


@dataclass(frozen=True)
class Config:
    control_radius_m: float = 2.0
    control_weight: float = 2.0
    tag_weight: float = 0.45
    outside_margin_m: float = 0.5
    outside_weight: float = 3.0
    stationary_speed_mps: float = 0.5
    stationary_seconds: float = 1.0
    max_ball_speed_mps: float = 45.0
    transition_cost: float = 1.5  # score-seconds; emissions divided by fps


def normalise(value):
    value = unicodedata.normalize('NFKD', value or '')
    return ''.join(c for c in value.casefold() if c.isalnum())


def read_tracking(path, teams, n_frames, pitch, fps, config):
    """Stream the official raw format; enforce 1-based consecutive video IDs.

    Keep team distances, ball motion and median player speed, not player identity.
    Missing measurements stay NaN/inf rather than becoming dead-ball evidence.
    """
    ball = np.full((n_frames, 2), np.nan)
    distance = np.full((n_frames, 2), np.inf)
    player_speed = np.full(n_frames, np.nan)
    count = 0
    with open(path) as handle:
        for i, line in enumerate(handle):
            parts = line.strip().split(':')
            if len(parts) != 3 or i >= n_frames or int(parts[0]) != i + 1:
                raise ValueError('ATD must contain consecutive frame IDs 1..n, one row per frame')
            groups = parts[1].split(';')
            if len(groups) != len(teams):
                raise ValueError('ATD slot count does not match metadata at frame {}'.format(i))
            players = np.array([[float(x) for x in g.split(',')[:3]] for g in groups])
            if players.shape != (len(teams), 3):
                raise ValueError('Expected x,y,speed for each track')
            b = np.array([float(x) for x in parts[2].split(',')[:2]])
            if b.shape != (2,):
                raise ValueError('Expected ball x,y')
            ball[i] = b * pitch
            xy = players[:, :2] * pitch
            valid = np.isfinite(xy).all(axis=1)
            speeds = players[valid, 2]
            speeds = speeds[np.isfinite(speeds) & (speeds >= 0)]
            if len(speeds):
                player_speed[i] = np.median(speeds)
            if np.isfinite(ball[i]).all():
                for side in (0, 1):
                    selected = valid & (teams == side)
                    if selected.any():
                        distance[i, side] = np.linalg.norm(xy[selected] - ball[i], axis=1).min()
            count += 1
    if count != n_frames:
        raise ValueError('Expected {} ATD frames, found {}'.format(n_frames, count))
    speed = np.full(n_frames, np.nan)
    speed[1:] = np.linalg.norm(np.diff(ball, axis=0), axis=1) * fps
    # Reject evidence on both endpoints of an implausible jump. Do not interpolate it.
    jumps = np.flatnonzero(speed > config.max_ball_speed_mps)
    bad = np.unique(np.concatenate((jumps, jumps - 1)))
    ball[bad] = np.nan
    distance[bad] = np.inf
    speed[:] = np.nan
    speed[1:] = np.linalg.norm(np.diff(ball, axis=0), axis=1) * fps
    return ball, distance, speed, player_speed


def tag_evidence(events, names, n_frames, config):
    """Bound correlated tags; defence supports the opposing team's possession.

    Set pieces are context only: neither their team nor their full interval is
    assigned possession/dead ball directly. Unknown relevant team names fail fast.
    """
    evidence = np.zeros((n_frames, 3))
    set_piece = np.zeros(n_frames, dtype=bool)
    mapping = {normalise(name): i for i, name in enumerate(names)}
    if len(mapping) != 2 or '' in mapping:
        raise ValueError('Need two distinct Smart Tagging team names')
    offensive = {'build up', 'progression', 'creating chances', 'attacking transition',
                 'long ball', 'ball in final third', 'ball in the box'}
    defensive = {'defensive transition', 'defending in attacking third',
                 'defending in middle third', 'defending in defensive third'}
    ignored = set()
    for event in events:
        raw_start, raw_end = event['start_frame'], event['end_frame']
        if raw_end < raw_start:
            raise ValueError('Smart Tagging interval ends before it starts')
        start, end = max(0, raw_start), min(n_frames, raw_end)
        if start >= end:
            continue
        code = ' '.join((event['code'] or '').casefold().split())
        if code == 'set pieces':
            set_piece[start:end] = True
            continue
        if code not in offensive | defensive:
            ignored.add(code)
            continue
        name = normalise(event['tags'].get('Team'))
        if name in {'', 'na'}:
            continue  # Missing team attribution provides no possession evidence.
        if name not in mapping:
            raise ValueError('Unmapped Team {!r}; set --home-tag/--away-tag explicitly'
                             .format(event['tags'].get('Team')))
        side = mapping[name]
        if code in defensive:
            side = 1 - side
        evidence[start:end, side] = config.tag_weight
    return evidence, set_piece, sorted(ignored)


def sustained(mask, min_frames):
    """Keep complete true runs only if long enough; no bridging missing data."""
    result = np.zeros(len(mask), dtype=bool)
    boundaries = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    for start, end in boundaries.reshape(-1, 2):
        if end - start >= min_frames:
            result[start:end] = True
    return result


def emissions(ball, distance, speed, player_speed, events, names, pitch, fps, config):
    scores, set_piece, ignored = tag_evidence(events, names, len(ball), config)
    visible = np.isfinite(ball).all(axis=1)
    # Close control is evidence, not a hard nearest-player assignment.
    control = np.exp(-0.5 * (distance / config.control_radius_m) ** 2)
    scores[:, :2] += config.control_weight * control
    outside = visible & ((ball[:, 0] < -config.outside_margin_m)
                         | (ball[:, 0] > pitch[0] + config.outside_margin_m)
                         | (ball[:, 1] < -config.outside_margin_m)
                         | (ball[:, 1] > pitch[1] + config.outside_margin_m))
    scores[outside, :2] = 0
    scores[outside, 2] += config.outside_weight
    # Stationary ball alone is NOT dead. Require set-piece context as well.
    stationary = sustained(visible & np.isfinite(speed)
                           & (speed < config.stationary_speed_mps),
                           max(1, round(config.stationary_seconds * fps)))
    dead_context = stationary & set_piece
    scores[dead_context, 2] += config.control_weight + config.tag_weight
    # Collective stillness is only a small supporting signal, never standalone.
    scores[dead_context & (player_speed < 1.0), 2] += 0.25
    return scores / fps, {'ignored_tag_codes': ignored,
                         'visible_ball_frames': int(visible.sum()),
                         'outside_ball_frames': int(outside.sum()),
                         'stationary_set_piece_frames': int(dead_context.sum())}


def viterbi(scores, transition_cost):
    """Maximum-score H/A/D path with a soft cost for each state change."""
    if scores.ndim != 2 or scores.shape[1] != 3 or not np.isfinite(scores).all():
        raise ValueError('Expected finite n x 3 emission scores')
    if not np.isfinite(transition_cost) or transition_cost < 0:
        raise ValueError('Transition cost must be finite and nonnegative')
    if not len(scores):
        return np.empty(0, dtype=np.int8)
    penalty = np.full((3, 3), transition_cost)
    np.fill_diagonal(penalty, 0)
    back = np.zeros((len(scores), 3), dtype=np.int8)
    best = scores[0].copy()
    for frame in range(1, len(scores)):
        candidates = best[:, None] - penalty
        back[frame] = candidates.argmax(axis=0)
        best = candidates.max(axis=0) + scores[frame]
        best -= best.max()
    path = np.empty(len(scores), dtype=np.int8)
    path[-1] = best.argmax()
    for frame in range(len(scores) - 1, 0, -1):
        path[frame - 1] = back[frame, path[frame]]
    return path


def decode(scores, blocks, config):
    """Independent halves; entirely unsupported blocks abstain, outside is X."""
    labels = np.full(len(scores), 'X')
    for _, first, last in blocks:
        block = scores[first:last + 1]
        if np.any(block):
            labels[first:last + 1] = LABELS[viterbi(block, config.transition_cost)]
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data-dir')
    parser.add_argument('--home-tag', help='Exact home club name used in Smart Tagging')
    parser.add_argument('--away-tag', help='Exact away club name used in Smart Tagging')
    parser.add_argument('--output', default=str(ROOT / 'evaluation/results/team_temporal.csv'))
    args = parser.parse_args()
    config = Config()
    try:
        meta, raw, _ = require(['atd_meta', 'atd_raw', 'smart_tagging'], data_dir=args.data_dir)
        names = team_names(meta)
        names = (args.home_tag or names[0], args.away_tag or names[1])
        game = GAMES[GAME]
        features = read_tracking(raw, team_of_each_track(meta), game['n_video_frames'],
                                 np.array(game['pitch']), game['fps'], config)
        scores, diagnostics = emissions(*features, load_tags(data_dir=args.data_dir), names,
                                        game['pitch'], game['fps'], config)
        labels = decode(scores, half_video_blocks(GAME), config)
        write_submission(args.output, labels)
        digest = hashlib.sha256(Path(args.output).read_bytes()).hexdigest()
        report = {'method': 'temporal_v1', 'config': asdict(config), 'diagnostics': diagnostics,
                  'sha256': digest, 'labels': {x: int((labels == x).sum()) for x in 'HADX'},
                  'note': 'Untuned experimental method. No ground truth read during inference.'}
        Path(args.output).with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
        print('Written {} (SHA-256 {})'.format(args.output, digest))
        print('Next: python evaluation/validate_submission.py {}'.format(args.output))
    except (OSError, ValueError) as error:
        parser.exit(2, '{}\n'.format(error))


if __name__ == '__main__':
    main()
