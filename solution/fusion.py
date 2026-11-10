"""Merge the broadcast tracking into the tactical timeline (needs sync_broadcast.py first).

On every tactical frame with a LIVE broadcast frame synchronised to it:
  - players: each team's tactical and broadcast players are paired greedily by distance (closest
    pair first, at most MATCH_R apart). A pair becomes one player at the weighted mean
    (W_TACTICAL / 1 - W_TACTICAL, 75/25 by default). A tactical player with no partner stays as
    he is; a broadcast player with no partner is added in an extra slot, after the 46 tactical
    ones, only if he is farther than GHOST_R from every tactical player of either team: closer
    than that he is almost always a double of a player the tactical camera already tracks.
  - ball (BALL_MODE "fill", the default): the tactical ball wherever the tactical camera sees it;
    where only the broadcast sees it, the broadcast ball shifted by the offset between the two
    cameras measured on the nearest frames both see it (within OFFSET_S, interpolated between
    the one before and the one after), so the track does not jump when the source changes.
    Measured without labels: the offset is a slow bias (3 cm from one frame to the next, 0.5 m
    over a second), and where the cameras are more than 2 m apart (a quarter of the frames both
    see) the gap grows and shrinks smoothly over about a second with both tracks smooth and the
    ball faster: a ball in the air, which each camera projects onto the ground from its own point
    of view. Neither is wrong, so there is nothing to average. The weighted mean (BALL_MODE
    "mean", the first version) made the ball jump 0.8 m (median) every time the source changed,
    which the touch detector read as touches: +24% touches and +8% possessions on the match.
Elsewhere (replays, close-ups, cuts, halftime) the tactical tracking is kept unchanged.

Writes data/derived/<game>_fused_tracking.npz in the same shape tracking.load() returns, so any
script can switch with tracking.load(source="fused"), plus per-frame provenance.

    .venv/bin/python solution/fusion.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, half_video_blocks  # noqa: E402
import tracking  # noqa: E402
import sync_broadcast  # noqa: E402

OUT = os.path.join(ROOT, "data", "derived", "{}_fused_tracking.npz".format(GAME))
W_TACTICAL = 0.75    # weight of the tactical position in a pair (broadcast gets the rest): the
                     # tactical camera is the reference; the broadcast nudges it and fills its gaps
MATCH_R = 3.0        # m: farther apart than this, two tracks are not the same player
GHOST_R = 3.0        # m: a broadcast-only player this close to ANY tactical player is a ghost
                     # double (a tactical player the pairing missed, or on the other team's
                     # side of a mix-up) and is not added

BALL_MODE = "fill"   # "fill": tactical ball, broadcast only in its gaps, offset-corrected;
                     # "mean": weighted mean where both see it (first version, kept for comparison)
OFFSET_S = 1.0       # s: a both-seen frame this close gives the offset for a broadcast-only frame

BALL_NONE, BALL_TACTICAL, BALL_BROADCAST, BALL_BOTH = 0, 1, 2, 3


def pair(a, b, r):
    """Greedy closest-first pairing of two point sets: [(i, j)] with distance <= r."""
    if not len(a) or not len(b):
        return []
    d = np.linalg.norm(a[:, None] - b[None], axis=2)
    order = np.argsort(d, axis=None)
    used_a, used_b, out = set(), set(), []
    for k in order:
        i, j = divmod(int(k), d.shape[1])
        if d[i, j] > r:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        out.append((i, j))
    return out


def shift_broadcast_only(ball, ball_src, b_off, max_back):
    """Move every run of broadcast-only balls by the camera offset measured at its two ends (the
    nearest both-seen frame at most max_back frames before its first frame and after its last),
    linear in time between the two, constant if only one end has it, unshifted if neither."""
    n = len(ball)
    has = ~np.isnan(b_off[:, 0])
    idx = np.arange(n)
    prev = np.maximum.accumulate(np.where(has, idx, -1))
    nxt = np.minimum.accumulate(np.where(has, idx, n)[::-1])[::-1]
    e = np.flatnonzero(np.diff(np.concatenate(([0], (ball_src == BALL_BROADCAST).astype(np.int8), [0]))))
    shifted = 0
    for s0, e0 in zip(e[::2], e[1::2]):  # run s0 .. e0 - 1
        p, q = prev[s0], nxt[e0 - 1]
        p_ok, q_ok = p >= 0 and s0 - p <= max_back, q < n and q - (e0 - 1) <= max_back
        if not (p_ok or q_ok):
            continue
        t = np.arange(s0, e0)
        if p_ok and q_ok:
            a = ((t - p) / (q - p))[:, None]
            off = (1 - a) * b_off[p] + a * b_off[q]
        else:
            off = np.broadcast_to(b_off[p if p_ok else q], (len(t), 2))
        ball[s0:e0] -= off
        shifted += e0 - s0
    return shifted


def main(ball_mode=BALL_MODE):
    t_teams, t_pl, t_ball, t_bs, names = tracking.load(GAME)
    b_teams, b_pl, b_ball, b_bs, _ = tracking.load(GAME, "broadcast")
    sync = np.load(sync_broadcast.OUT)
    b_live = sync["b_live"]
    n, nt, nb = len(t_pl), t_pl.shape[1], b_pl.shape[1]

    players = np.full((n, nt + nb, 2), np.nan, np.float32)
    players[:, :nt] = t_pl
    teams = np.concatenate([t_teams, b_teams]).astype(np.int8)
    ball = t_ball.copy()
    ball_speed = t_bs.copy()
    t_vis = ~np.isnan(t_ball[:, 0])
    ball_src = np.where(t_vis, BALL_TACTICAL, BALL_NONE).astype(np.int8)
    paired = np.zeros(n, np.int16)
    added = np.zeros(n, np.int16)
    ball_gap = np.full(n, np.nan, np.float32)  # distance between the two balls when both are seen
    b_off = np.full((n, 2), np.nan, np.float32)  # broadcast - tactical ball when both are seen

    w = W_TACTICAL
    ghosts = np.zeros(n, np.int16)
    for t in np.flatnonzero(b_live >= 0):
        b = b_live[t]
        t_all = t_pl[t][~np.isnan(t_pl[t, :, 0])]
        for team in (0, 1):
            ti = np.flatnonzero((t_teams == team) & ~np.isnan(t_pl[t, :, 0]))
            bi = np.flatnonzero((b_teams == team) & ~np.isnan(b_pl[b, :, 0]))
            pairs = pair(t_pl[t, ti], b_pl[b, bi], MATCH_R)
            got = set()
            for i, j in pairs:
                players[t, ti[i]] = w * t_pl[t, ti[i]] + (1 - w) * b_pl[b, bi[j]]
                got.add(j)
            for j in range(len(bi)):
                if j in got:
                    continue
                q = b_pl[b, bi[j]]
                if len(t_all) and np.min(np.linalg.norm(t_all - q, axis=1)) <= GHOST_R:
                    ghosts[t] += 1
                    continue
                players[t, nt + bi[j]] = q
                added[t] += 1
            paired[t] += len(pairs)
        bv = not np.isnan(b_ball[b, 0])
        if bv and t_vis[t]:
            ball_gap[t] = np.linalg.norm(t_ball[t] - b_ball[b])
            b_off[t] = b_ball[b] - t_ball[t]
            if ball_mode == "mean":
                ball[t] = w * t_ball[t] + (1 - w) * b_ball[b]
                if np.isfinite(t_bs[t]) and np.isfinite(b_bs[b]):
                    ball_speed[t] = w * t_bs[t] + (1 - w) * b_bs[b]
            ball_src[t] = BALL_BOTH
        elif bv:
            ball[t] = b_ball[b]
            ball_speed[t] = b_bs[b]
            ball_src[t] = BALL_BROADCAST

    shifted = shift_broadcast_only(ball, ball_src, b_off, int(OFFSET_S * 25)) \
        if ball_mode == "fill" else 0

    np.savez(OUT, teams=teams, players=players, ball=ball, ball_speed=ball_speed,
             ball_src=ball_src, paired=paired, added=added, ball_gap=ball_gap)

    graded = np.zeros(n, bool)
    for _, a, z in half_video_blocks(GAME):
        graded[a:z + 1] = True
    live = b_live >= 0
    g = graded
    print("graded frames with a live broadcast frame: {:.1f}%".format(100 * live[g].mean()))
    print("ball seen on graded frames: tactical {:.1f}% -> fused {:.1f}%  (broadcast only {:.1f}%, both {:.1f}%)".format(
        100 * t_vis[g].mean(), 100 * (ball_src[g] != BALL_NONE).mean(),
        100 * (ball_src[g] == BALL_BROADCAST).mean(), 100 * (ball_src[g] == BALL_BOTH).mean()))
    both = g & np.isfinite(ball_gap)
    if both.any():
        print("where both see the ball, they are {:.2f} m apart (median), {:.2f} m (p90)".format(
            np.median(ball_gap[both]), np.percentile(ball_gap[both], 90)))
    lg = g & live
    print("on live frames: {:.1f} player pairs merged, {:.2f} broadcast-only players added, {:.2f} ghost "
          "doubles dropped (mean per frame)".format(paired[lg].mean(), added[lg].mean(), ghosts[lg].mean()))
    if ball_mode == "fill":
        bo = g & (ball_src == BALL_BROADCAST)
        print("ball mode fill: broadcast-only frames shifted by the camera offset {:.0f}%".format(
            100 * shifted / max(int((ball_src == BALL_BROADCAST).sum()), 1)))
        step = np.linalg.norm(np.diff(ball, axis=0), axis=1)
        sw = (ball_src[1:] != ball_src[:-1]) & (ball_src[1:] > 0) & (ball_src[:-1] > 0) & g[1:]
        print("ball jump where the source changes: median {:.2f} m, p90 {:.2f} m ({} changes, {} "
              "graded broadcast-only frames)".format(np.nanmedian(step[sw]), np.nanpercentile(step[sw], 90),
                                                     int(sw.sum()), int(bo.sum())))
    print("-> {}".format(os.path.relpath(OUT, ROOT)))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ball", choices=("fill", "mean"), default=BALL_MODE,
                    help="fill (default): tactical ball, broadcast only in its gaps; mean: 75/25 mean")
    main(ap.parse_args().ball)
