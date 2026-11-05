"""Make a clean external tracking look like the BAR-ATH one, so calibration sees the same problems.

The Metrica sample games already lose the ball like BAR-ATH does (59-68% visible vs 59%, ~97% of
missing frames in gaps of 5 s or more), so the ball is left alone. Their players, though, are
perfect: one slot per person, never lost, never on the wrong team. BAR-ATH has a team over 11 on
6.4% of team-frames (a player carried by the other team's slot) and under 9 on 8.8%.

Model: every player's track is cut into tracklets with BAR-ATH-like lengths (exponential, mean
TRACKLET_S); each tracklet is moved to the other team with probability SWAP, or dropped with
probability DROP. On top, per team, BURST_EVERY_S on average, BURST_K players are lost together
for about BURST_S (the tactical camera loses a group, e.g. at the far side). The defaults match
BAR-ATH's per-team count histogram (11: 57.6%, 10: 21.7%, 9: 5.5%, <=8: 4.6%, 12: 5.9%), printed
by running this file. Seeded: the same seed gives the same degraded game.
"""

import numpy as np

FPS = 25
TRACKLET_S = 35.0
SWAP = 0.009
DROP = 0.02
BURST_EVERY_S = 60.0
BURST_S = 4.0
BURST_K = (2, 6)


def degrade(teams, players, seed, swap=SWAP, drop=DROP, tracklet_s=TRACKLET_S,
            burst_every_s=BURST_EVERY_S, burst_s=BURST_S, burst_k=BURST_K):
    """(teams2, players2): twice the slots, second half = each player's copy on the other team."""
    rng = np.random.default_rng(seed)
    n, slots = players.shape[:2]
    out = np.full((n, 2 * slots, 2), np.nan, dtype=players.dtype)
    out[:, :slots] = players
    for s in range(slots):
        t = 0
        while t < n:
            length = max(1, int(rng.exponential(tracklet_s * FPS)))
            seg = slice(t, min(n, t + length))
            u = rng.random()
            if u < swap:
                out[seg, slots + s] = out[seg, s]
                out[seg, s] = np.nan
            elif u < swap + drop:
                out[seg, s] = np.nan
            t += length
    for k in (0, 1):
        cols = np.flatnonzero(teams == k)
        t = int(rng.exponential(burst_every_s * FPS))
        while t < n:
            length = max(1, int(rng.exponential(burst_s * FPS)))
            on = cols[~np.isnan(players[t, cols, 0])]
            m = min(len(on), int(rng.integers(burst_k[0], burst_k[1] + 1)))
            for s in rng.choice(on, size=m, replace=False) if m else []:
                out[t:t + length, s] = np.nan
                out[t:t + length, slots + s] = np.nan
            t += length + int(rng.exponential(burst_every_s * FPS))
    return np.concatenate([teams, 1 - teams]).astype(teams.dtype), out


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import metrica
    from clean_tracking import team_counts
    for g in metrica.GAMES:
        teams, players, ball, _, periods = metrica.load(g)
        graded = metrica.labels(g, len(ball), periods) != "X"
        t2, p2 = degrade(teams, players, seed=g)
        ok = ~np.isnan(p2[graded][..., 0])
        c = np.concatenate([ok[:, t2 == k].sum(1) for k in (0, 1)])
        print("game {}: 11 {:.1f}%  10 {:.1f}%  9 {:.1f}%  <=8 {:.1f}%  12+ {:.1f}%".format(
            g, *(100 * np.mean(x) for x in (c == 11, c == 10, c == 9, c <= 8, c >= 12))))
    print("BAR-ATH: 11 57.6%  10 21.7%  9 5.5%  <=8 4.6%  12+ 6.2%")
