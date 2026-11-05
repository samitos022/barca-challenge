"""Merge the broadcast tracking into the tactical timeline (needs sync_broadcast.py first).

On every tactical frame with a LIVE broadcast frame synchronised to it:
  - players: each team's tactical and broadcast players are paired greedily by distance (closest
    pair first, at most MATCH_R apart). A pair becomes one player at the weighted mean
    (W_TACTICAL / 1 - W_TACTICAL, 50/50 by default). A tactical player with no partner stays as
    he is; a broadcast player with no partner (the tactical tracker missed him) is added in an
    extra slot, after the 46 tactical ones.
  - ball: seen by both, the weighted mean; seen by one, that one; seen by none, missing.
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
W_TACTICAL = 0.5     # weight of the tactical position in a pair (broadcast gets the rest)
MATCH_R = 3.0        # m: farther apart than this, two tracks are not the same player

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


def main():
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

    w = W_TACTICAL
    for t in np.flatnonzero(b_live >= 0):
        b = b_live[t]
        for team in (0, 1):
            ti = np.flatnonzero((t_teams == team) & ~np.isnan(t_pl[t, :, 0]))
            bi = np.flatnonzero((b_teams == team) & ~np.isnan(b_pl[b, :, 0]))
            pairs = pair(t_pl[t, ti], b_pl[b, bi], MATCH_R)
            got = set()
            for i, j in pairs:
                players[t, ti[i]] = w * t_pl[t, ti[i]] + (1 - w) * b_pl[b, bi[j]]
                got.add(j)
            for j in range(len(bi)):
                if j not in got:
                    players[t, nt + bi[j]] = b_pl[b, bi[j]]
            paired[t] += len(pairs)
            added[t] += len(bi) - len(got)
        bv = not np.isnan(b_ball[b, 0])
        if bv and t_vis[t]:
            ball_gap[t] = np.linalg.norm(t_ball[t] - b_ball[b])
            ball[t] = w * t_ball[t] + (1 - w) * b_ball[b]
            if np.isfinite(t_bs[t]) and np.isfinite(b_bs[b]):
                ball_speed[t] = w * t_bs[t] + (1 - w) * b_bs[b]
            ball_src[t] = BALL_BOTH
        elif bv:
            ball[t] = b_ball[b]
            ball_speed[t] = b_bs[b]
            ball_src[t] = BALL_BROADCAST

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
    print("on live frames: {:.1f} player pairs merged, {:.1f} broadcast-only players added (mean per frame)".format(
        paired[lg].mean(), added[lg].mean()))
    print("-> {}".format(os.path.relpath(OUT, ROOT)))


if __name__ == "__main__":
    main()
