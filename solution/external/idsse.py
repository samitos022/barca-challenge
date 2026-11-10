"""IDSSE / DFL Bundesliga open data: TRACAB tracking in the shape metrica.load() returns, plus
H/A/D labels from the tracking's ball-alive flag and the DFL event data. A second calibration set
next to the Metrica sample games (RULES.md: labelled data from OTHER games is allowed).

Source: Bassek, Rein, Weber, Memmert (2025), "An integrated dataset of spatiotemporal and event
data in elite soccer", Scientific Data 12:195, https://doi.org/10.6084/m9.figshare.28196177
(mirror: huggingface.co/datasets/pysport/idsse-data). 7 full matches, Bundesliga 1 and 2,
2022/23, data (c) Deutsche Fussball Liga (DFL), CC-BY 4.0. TRACAB optical tracking at 25 fps,
every player and the ball, centred coordinates (moved here to 0..105 x 0..68 like Metrica).
Download (about 2.6 GB of XML) with

    .venv/bin/python solution/external/idsse.py --download

into data/external/idsse/ (git-ignored). The first load parses the XML (about a minute per match)
and caches the arrays in data/derived/idsse_<id>.npz.

Labels (an algorithm applied to DFL's annotation, documented here). Three variants:
  - "flag" (default): D where the tracking's BallStatus flag says the ball is dead; the team from
    the events. This mirrors how the challenge's ground truth is built (CHALLENGE.md: dead where
    the corrected tracking marks the ball dead), with the team chosen by our own reading of
    "controlled possession", the same as metrica.py:
      * a Play (pass, cross), a ShotAtGoal, an OtherBallAction (not a clearance) or a set piece
        (kick-off, throw-in, free kick, goal kick, corner, penalty) gives the ball to its team;
      * a BallClaiming (interception, ball claimed / held), a TacklingGame with PossessionChange
        and a clearance only PROPOSE a change: the ball changes hands, dated to that event, if
        the same team then plays it before the other one does. A challenge or a loose ball stays
        with the last team ("contested or loose, possession stays with the last team");
      * each alive run starts with the team of the first owning event at or just after (within
        RESTART_TOL_S before) its start: the restart taker, not the team that put the ball out.
  - "events": the same team, with D from the events alone: from each set piece's
    DecisionTimestamp (the referee's decision, i.e. the ball out / the foul) to its EventTime
    (the kick), from a goal to the next kick-off, and from the last event to a RefereeBall.
  - "tracab": D from BallStatus, team from the tracking's own BallPossession flag. This is the
    literal mechanical derivation of CHALLENGE.md ("otherwise takes the team the tracking names as
    owning it"), so the closest in spirit to the challenge GT; DFL's possession flag is not ours,
    though, and we do not know how its "control" is defined.
  Graded frames: from the first to the last alive frame of each half (X elsewhere).

Caveats:
  - Event times are DFL live annotations on the wall clock; the tracking has its own timestamps.
    load_events() shifts every event by the per-match median lag between a set piece's EventTime
    and the nearest BallStatus dead -> alive switch (reported by running this file).
  - BallStatus is TRACAB's flag (operator assisted), not the challenge annotators': D from it is
    a silver standard, like the Metrica labels. Its D share is printed per match.
  - The ball is tracked almost always and the players never get lost or swap teams: use
    degrade.degrade() and degrade.degrade_ball() to make it look like BAR-ATH.
"""

import os
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DIR = os.path.join(ROOT, "data", "external", "idsse")
FPS = 25
HOME, AWAY = 0, 1
FIGSHARE = "https://ndownloader.figshare.com/files/"
# match id -> (competition, figshare ids of matchinformation, events, positions)
MATCHES = {
    "J03WMX": ("000001", 51643475, 51643493, 51643514),
    "J03WN1": ("000001", 51643472, 51643496, 51643517),
    "J03WOH": ("000002", 51643478, 51643499, 51643520),
    "J03WOY": ("000002", 51643481, 51643502, 51643523),
    "J03WPY": ("000002", 51643487, 51643505, 51643526),
    "J03WQQ": ("000002", 51643484, 51643508, 51643529),
    "J03WR9": ("000002", 51643490, 51643511, 51643532),
}
GAMES = tuple(MATCHES)
KINDS = ("02_01_matchinformation", "03_02_events_raw", "04_03_positions_raw_observed")
RESTART_TOL_S = 2.0      # s: the restart event may be annotated a little before the flag switches
SET_PIECES = ("KickOff", "ThrowIn", "FreeKick", "GoalKick", "CornerKick", "Penalty")


def _name(game, k):
    return "DFL_{}_DFL-COM-{}_DFL-MAT-{}.xml".format(KINDS[k], MATCHES[game][0], game)


def download():
    os.makedirs(DIR, exist_ok=True)
    for game, (_, *ids) in MATCHES.items():
        for k, fid in enumerate(ids):
            dst = os.path.join(DIR, _name(game, k))
            if os.path.exists(dst):
                continue
            print("downloading", os.path.basename(dst), flush=True)
            urllib.request.urlretrieve(FIGSHARE + str(fid), dst + ".part")
            os.replace(dst + ".part", dst)


def _path(game, k):
    p = os.path.join(DIR, _name(game, k))
    if not os.path.exists(p):
        raise FileNotFoundError("{} missing: run solution/external/idsse.py --download".format(p))
    return p


def _ts(s):
    return datetime.fromisoformat(s).timestamp()


def match_info(game):
    """(home team id, away team id, (length, width))."""
    root = ET.parse(_path(game, 0)).getroot()
    g = root.find(".//General")
    env = root.find(".//Environment")
    return g.get("HomeTeamId"), g.get("GuestTeamId"), (float(env.get("PitchX")), float(env.get("PitchY")))


# ---- tracking --------------------------------------------------------------------------------

_SET = re.compile(r'GameSection="([^"]+)".*?TeamId="([^"]+)" PersonId="([^"]+)"')
_FRAME = re.compile(r'N="(\d+)" T="([^"]+)" X="([^"]+)" Y="([^"]+)"')
_POSS = re.compile(r'BallPossession="(\d)"')
_STATUS = re.compile(r'BallStatus="(\d)"')


def _parse_positions(game):
    """{(section, team, person): (N, X, Y, possession, status)} streamed from the XML."""
    sets, cur, rows = {}, None, None
    t_first = {}
    with open(_path(game, 2)) as h:
        for line in h:
            if line.startswith("<Frame "):
                m = _FRAME.search(line)
                b = _STATUS.search(line) if cur[1] == "BALL" else None
                if b is not None:
                    o = _POSS.search(line)
                    rows.append((int(m.group(1)), float(m.group(3)), float(m.group(4)),
                                 int(o.group(1)) if o else 0, int(b.group(1))))
                    if cur[0] not in t_first:
                        t_first[cur[0]] = (int(m.group(1)), _ts(m.group(2)))
                else:
                    rows.append((int(m.group(1)), float(m.group(3)), float(m.group(4)), 0, 0))
            elif line.startswith("<FrameSet"):
                m = _SET.search(line)
                cur = (m.group(1), m.group(2), m.group(3))
                rows = sets.setdefault(cur, [])
    return {k: np.array(v, dtype=np.float64) for k, v in sets.items()}, t_first


def _build(game):
    home, away, pitch = match_info(game)
    sets, t_first = _parse_positions(game)
    sections = [s for s in ("firstHalf", "secondHalf", "firstHalfExtra", "secondHalfExtra")
                if any(k[0] == s and k[1] == "BALL" for k in sets)]
    persons = sorted({(0 if k[1] == home else 1, k[2]) for k in sets if k[1] in (home, away)})
    slot = {p: i for i, (_, p) in enumerate(persons)}
    teams = np.array([t for t, _ in persons], dtype=np.int8)
    spans, n = [], 0
    for s in sections:
        bn = sets[next(k for k in sets if k[0] == s and k[1] == "BALL")][:, 0]
        n0, n1 = int(bn.min()), int(bn.max())
        spans.append((s, n0, n1, n))
        n += n1 - n0 + 1
    players = np.full((n, len(persons), 2), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    status = np.zeros(n, dtype=np.int8)
    owner = np.zeros(n, dtype=np.int8)
    period = np.zeros(n, dtype=np.int8)
    t0 = np.zeros(len(spans))
    off = np.array([pitch[0] / 2, pitch[1] / 2], dtype=np.float32)
    for h, (s, n0, n1, base) in enumerate(spans):
        period[base:base + n1 - n0 + 1] = h + 1
        fn, ft = t_first[s]
        t0[h] = ft - (fn - n0) / FPS
        for (sec, team, person), a in sets.items():
            if sec != s:
                continue
            r = base + (a[:, 0].astype(np.int64) - n0)
            ok = (r >= base) & (r <= base + n1 - n0)
            xy = a[ok, 1:3].astype(np.float32) + off
            if team == "BALL":
                ball[r[ok]] = xy
                status[r[ok]] = a[ok, 4]
                owner[r[ok]] = a[ok, 3]
            elif team in (home, away):
                players[r[ok], slot[person]] = xy
    return dict(period=period, teams=teams, players=players, ball=ball, status=status,
                owner=owner, t0=t0, pitch=np.array(pitch))


def _cache(game):
    return os.path.join(ROOT, "data", "derived", "idsse_{}.npz".format(game))


def build_cache(game):
    if not os.path.exists(_cache(game)):
        z = _build(game)
        os.makedirs(os.path.dirname(_cache(game)), exist_ok=True)
        np.savez_compressed(_cache(game), **z)
    return game


def raw(game):
    """Every cached array: period, teams, players, ball, status (1 alive), owner (1 home, 2 away),
    t0 (unix time of each half's first row), pitch."""
    build_cache(game)
    return dict(np.load(_cache(game)))


def load(game):
    """(teams, players, ball, ball_speed, periods) in metres, like metrica.load (pitch: PITCH())."""
    z = raw(game)
    periods = []
    for k in np.unique(z["period"]):
        idx = np.flatnonzero(z["period"] == k)
        periods.append((int(k), int(idx[0]), int(idx[-1])))
    ball_speed = np.full(len(z["ball"]), np.nan, dtype=np.float32)
    return z["teams"], z["players"], z["ball"], ball_speed, periods


def pitch(game):
    return match_info(game)[2]


# ---- events -----------------------------------------------------------------------------------

def _row(t, t0, period):
    """Row of wall-clock time t (unix s), in the half whose clock it falls in, or None."""
    starts = np.concatenate(([0], np.flatnonzero(np.diff(period)) + 1))
    ends = np.concatenate((starts[1:], [len(period)])) - 1
    for h, (a, b) in enumerate(zip(starts, ends)):
        r = a + int(round((t - t0[h]) * FPS))
        if a - 60 * FPS <= r <= b + 60 * FPS:
            return int(min(max(r, a), b))
    return None


def _raw_events(game):
    """[(time, kind, team, decision time or None)] in time order.

    kind: 'own' (the team plays the ball), 'propose' (a recovery to be confirmed), 'restart' (a
    set piece), 'goal', 'referee_ball', 'end' (final whistle)."""
    home, away, _ = match_info(game)
    side = {home: HOME, away: AWAY}
    out = []
    for e in ET.parse(_path(game, 1)).getroot().iter("Event"):
        ch = list(e)
        if not ch:
            continue
        c, t = ch[0], _ts(e.get("EventTime"))
        tag = c.tag
        team = side.get(c.get("Team"))
        if tag in SET_PIECES:
            if team is None:
                team = next((side.get(d.get("Team")) for d in c.iter() if d.get("Team") in side), None)
            dec = c.get("DecisionTimestamp")
            out.append((t, "restart", team, _ts(dec) if dec else None, tag))
            if any(d.tag == "SuccessfulShot" for d in c.iter()):
                out.append((t, "goal", team, None, tag))
        elif tag == "Play" or tag == "ShotAtGoal":
            out.append((t, "own", team, None, tag))
            if any(d.tag == "SuccessfulShot" for d in c.iter()):
                out.append((t, "goal", team, None, tag))
        elif tag == "OtherBallAction":
            out.append((t, "propose" if c.get("DefensiveClearance") == "true" else "own", team, None, tag))
        elif tag == "BallClaiming":
            out.append((t, "propose", team, None, tag))
        elif tag == "TacklingGame" and c.get("PossessionChange") == "true":
            out.append((t, "propose", side.get(c.get("WinnerTeam")), None, tag))
        elif tag == "RefereeBall":
            out.append((t, "referee_ball", None, None, tag))
        elif tag == "FinalWhistle":
            out.append((t, "end", None, None, tag))
    out = [x for x in out if x[2] is not None or x[1] in ("referee_ball", "end")]
    out.sort(key=lambda x: x[0])
    return out


def _onsets(status):
    """Rows where BallStatus switches dead -> alive."""
    return np.flatnonzero((status[1:] == 1) & (status[:-1] == 0)) + 1


def event_lag(game, z=None):
    """(median, n): rows from each set piece's EventTime to the nearest dead -> alive switch of
    the flag within 5 s. Positive = the flag switches after the annotated time."""
    z = z or raw(game)
    on = _onsets(z["status"])
    lags = []
    for t, kind, _, _, tag in _raw_events(game):
        if kind != "restart":
            continue
        r = _row(t, z["t0"], z["period"])
        if r is None or not len(on):
            continue
        j = on[np.argmin(np.abs(on - r))]
        if abs(j - r) <= 5 * FPS:
            lags.append(j - r)
    return (int(np.median(lags)) if lags else 0), len(lags)


def load_events(game, z=None):
    """[(row, kind, team, decision row or None)], shifted by event_lag()."""
    z = z or raw(game)
    lag, _ = event_lag(game, z)
    out = []
    for t, kind, team, dec, _ in _raw_events(game):
        r = _row(t, z["t0"], z["period"])
        if r is None:
            continue
        d = _row(dec, z["t0"], z["period"]) if dec is not None else None
        out.append((r + lag, kind, team, d + lag if d is not None else None))
    return out


def dead_from_events(ev, n):
    """D mask from the events alone (variant 'events')."""
    dead = np.zeros(n, dtype=bool)
    last_goal, prev = None, None
    for r, kind, _, dec in ev:
        if kind == "restart":
            if dec is not None and dec < r:
                dead[max(dec, 0):r] = True
            if last_goal is not None:
                dead[last_goal:r] = True
                last_goal = None
        elif kind == "goal":
            last_goal = r
        elif kind == "referee_ball" and prev is not None:
            dead[prev:r] = True
        prev = r
    return dead


def _team_runs(ev, alive, lab):
    """Fill lab on every alive run with the events' team (see the module docstring)."""
    idx = np.flatnonzero(np.diff(np.concatenate(([0], alive.astype(np.int8), [0]))))
    rows = np.array([e[0] for e in ev])
    tol = int(RESTART_TOL_S * FPS)
    cur = HOME
    for a, b in zip(idx[::2], idx[1::2]):
        lo = np.searchsorted(rows, a - tol)
        hi = np.searchsorted(rows, b)
        first = next((ev[i][2] for i in range(lo, hi) if ev[i][1] in ("own", "restart")), None)
        cur = first if first is not None else cur
        t0, pending = a, None
        for i in range(np.searchsorted(rows, a), hi):
            r, kind, team, _ = ev[i]
            if kind == "propose":
                if team != cur and (pending is None or pending[0] != team):
                    pending = (team, r)
                elif team == cur:
                    pending = None
                continue
            if kind not in ("own", "restart"):
                continue
            if team == cur:
                pending = None
                continue
            at = pending[1] if pending is not None and pending[0] == team else r
            at = max(at, a)
            lab[t0:at] = "H" if cur == HOME else "A"
            cur, t0, pending = team, at, None
        lab[t0:b] = "H" if cur == HOME else "A"


def graded_mask(z):
    """From the first to the last alive frame of each half."""
    g = np.zeros(len(z["status"]), dtype=bool)
    for k in np.unique(z["period"]):
        idx = np.flatnonzero((z["period"] == k) & (z["status"] == 1))
        if len(idx):
            g[idx[0]:idx[-1] + 1] = True
    return g


def labels(game, variant="flag"):
    """H/A/D/X per row: 'flag' (default), 'events' or 'tracab' (see the module docstring)."""
    z = raw(game)
    n = len(z["status"])
    g = graded_mask(z)
    lab = np.full(n, "X", dtype="<U1")
    if variant == "tracab":
        lab[g] = np.where(z["status"][g] == 0, "D", np.where(z["owner"][g] == 1, "H", "A"))
        return lab
    ev = load_events(game, z)
    dead = (z["status"] == 0) if variant == "flag" else dead_from_events(ev, n)
    alive = g & ~dead
    lab[g] = "D"
    _team_runs(ev, alive, lab)
    lab[~g] = "X"
    return lab


def blocks_from_labels(lab):
    """Graded blocks [(k, first, last)]: the non-X runs, one per period."""
    idx = np.flatnonzero(np.diff(np.concatenate(([0], (lab != "X").astype(np.int8), [0]))))
    return [(k + 1, int(a), int(b) - 1) for k, (a, b) in enumerate(zip(idx[::2], idx[1::2]))]


def _summary(lab):
    seg = lab[lab != "X"]
    cut = np.flatnonzero(seg[1:] != seg[:-1]) + 1
    starts = np.concatenate(([0], cut))
    kinds = seg[starts]
    return len(seg), 100 * np.mean(seg == "D"), int(np.sum(kinds != "D"))


if __name__ == "__main__":
    if "--download" in sys.argv:
        download()
    with Pool(min(7, os.cpu_count())) as pool:
        pool.map(build_cache, GAMES)
    tot = np.zeros(4)
    for g in GAMES:
        z = raw(g)
        lag, nl = event_lag(g, z)
        lf, le, lt = labels(g, "flag"), labels(g, "events"), labels(g, "tracab")
        gm = lf != "X"
        df, de = lf[gm] == "D", le[gm] == "D"
        alive = ~df & (le[gm] != "D")
        team_ok = np.mean(lf[gm][~df] == lt[gm][~df])
        n, dshare, npos = _summary(lf)
        _, dshare_e, npos_e = _summary(le)
        _, _, npos_t = _summary(lt)
        print("{}: {:.0f} min graded, ball seen {:.1f}%, lag events->flag {:+.2f} s (n {}); D flag {:.1f}% "
              "events {:.1f}%; agree {:.1f}% (flag D also event D {:.0f}%, event D also flag D {:.0f}%); "
              "possessions flag {} events {} tracab {}; team events vs BallPossession {:.1f}%".format(
                  g, n / FPS / 60, 100 * np.mean(~np.isnan(z["ball"][gm, 0])), lag / FPS, nl, dshare,
                  dshare_e, 100 * np.mean(df == de), 100 * np.mean(de[df]), 100 * np.mean(df[de]),
                  npos, npos_e, npos_t, 100 * team_ok), flush=True)
