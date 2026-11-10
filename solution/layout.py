"""Stoppages read from where the players stand, without the ball.

When the tracker loses the ball the players still tell what is going on. Four layouts, each a
per-frame boolean computed from the player positions only (so it works inside a ball gap):

  kickoff  every tracked player in his own half (the kickers may stand in the centre circle), at
           least KO_MIN players tracked, for KO_S: the teams are lined up for a kick-off, i.e. a goal
           was scored (or the half is starting). Also gives the team taking it (closest to the spot).
  corner   a slow player within CORNER_R of a corner flag with CORNER_BOX or more players inside the
           penalty area at that end, for CORNER_S: a corner kick is about to be taken.
  throw    a slow player on or beyond a touchline, for THROW_S: a throw-in is about to be taken.
  wall     WALL_N or more players of one team side by side, standing, for WALL_S: a free kick.

How often each one means a dead ball is measured on the external matches (run this file): the
thresholds are physical (metres, m/s, seconds), nothing comes from BAR-ATH.

    .venv/bin/python solution/layout.py        # precision of each layout on the external matches
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE, os.path.join(HERE, "external"), os.path.join(ROOT, "starter"), os.path.join(ROOT, "evaluation")):
    if p not in sys.path:
        sys.path.insert(0, p)

FPS = 25
NAMES = ("kickoff", "corner", "throw", "wall")

KO_MARGIN = 1.0      # m over the halfway line still "in his own half" (tracking noise)
KO_CIRCLE = 10.5     # m from the centre spot: the players taking the kick-off (circle 9.15 m + noise)
KO_MIN = 16          # tracked players needed to trust "everybody"
KO_S = 2.0           # s the formation must hold
CORNER_R = 3.0       # m from the corner flag
CORNER_V = 1.5       # m/s: the taker is walking to the ball or standing
CORNER_BOX = 6       # players inside the penalty area at that end
CORNER_S = 1.5
THROW_OUT = 0.3      # m inside the touchline: on the line or beyond it
THROW_V = 1.0        # m/s
THROW_S = 1.5
WALL_N = 3
WALL_GAP = 1.3       # m between neighbours in the wall
WALL_V = 0.7         # m/s: standing
WALL_S = 2.0


def _hold(mask, k):
    """True on the frames of runs of True at least k long."""
    out = np.zeros_like(mask)
    e = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    for a, b in zip(e[::2], e[1::2]):
        if b - a >= k:
            out[a:b] = True
    return out


def signals(teams, players, blocks, pitch):
    """{name: (n,) bool} for the four layouts, and "kick_team" (n,) int8: the team taking the
    kick-off (-1 outside a kick-off formation)."""
    L, W = pitch
    n = len(players)
    out = {k: np.zeros(n, dtype=bool) for k in NAMES}
    kick_team = np.full(n, -1, dtype=np.int8)
    ok = ~np.isnan(players[..., 0])
    spd = np.full(players.shape[:2], np.nan, dtype=np.float32)
    spd[2:-2] = np.linalg.norm(players[4:] - players[:-4], axis=2) * (FPS / 4.0)
    for _, a, z in blocks:
        P, O, V = players[a:z + 1], ok[a:z + 1], spd[a:z + 1]
        x, y = P[..., 0], P[..., 1]
        mx = [np.nanmean(x[:, teams == k]) for k in (0, 1)]
        gx = [0.0, L] if mx[0] < mx[1] else [L, 0.0]
        slow = np.nan_to_num(V, nan=np.inf)

        # kick-off: nobody beyond the halfway line except inside the centre circle
        dc = np.hypot(x - L / 2, y - W / 2)
        bad = np.zeros(len(P), dtype=bool)
        for k in (0, 1):
            m = teams == k
            over = O[:, m] & (np.abs(x[:, m] - gx[k]) > L / 2 + KO_MARGIN) & (dc[:, m] > KO_CIRCLE)
            bad |= over.any(axis=1)
        ko = _hold(~bad & (O.sum(axis=1) >= KO_MIN), int(KO_S * FPS))
        out["kickoff"][a:z + 1] = ko
        dmin = [np.where(O[:, teams == k], dc[:, teams == k], np.inf).min(axis=1) for k in (0, 1)]
        kick_team[a:z + 1] = np.where(ko, np.where(dmin[0] <= dmin[1], 0, 1), -1)

        # corner: a slow player at a flag, the box full
        cor = np.zeros(len(P), dtype=bool)
        for g in (0.0, L):
            inbox = (O & (np.abs(x - g) <= 16.5) & (np.abs(y - W / 2) <= 20.16)).sum(axis=1)
            for fy in (0.0, W):
                at = O & (np.hypot(x - g, y - fy) <= CORNER_R) & (slow <= CORNER_V)
                cor |= at.any(axis=1) & (inbox >= CORNER_BOX)
        out["corner"][a:z + 1] = _hold(cor, int(CORNER_S * FPS))

        # throw-in: a slow player on or over a touchline, away from the corners
        thr = O & ((y <= THROW_OUT) | (y >= W - THROW_OUT)) & (slow <= THROW_V) & \
            (np.minimum(x, L - x) > CORNER_R)
        out["throw"][a:z + 1] = _hold(thr.any(axis=1), int(THROW_S * FPS))

        # wall: WALL_N players of a team standing side by side (each has a team-mate within WALL_GAP
        # on either side along the chain: counted as a connected group of standing team-mates)
        wal = np.zeros(len(P), dtype=bool)
        for k in (0, 1):
            m = np.flatnonzero(teams == k)
            st = O[:, m] & (slow[:, m] <= WALL_V)
            xx = np.where(st, x[:, m], np.nan)
            yy = np.where(st, y[:, m], np.nan)
            with np.errstate(invalid="ignore"):
                near = np.hypot(xx[:, :, None] - xx[:, None, :], yy[:, :, None] - yy[:, None, :]) <= WALL_GAP
            deg = near.sum(axis=2) - st                     # standing team-mates within WALL_GAP
            # a chain of 3: a player with two such neighbours; longer walls contain one
            wal |= ((deg >= WALL_N - 1) & st).any(axis=1)
        out["wall"][a:z + 1] = _hold(wal, int(WALL_S * FPS))
    out["kick_team"] = kick_team
    return out


def _job(args):
    import calibrate
    run, labels, cond = args
    teams, players, ball, bs, blocks, gt = calibrate.case(run, cond, labels)
    s = signals(teams, players, blocks, calibrate.pitch_of(run))
    g = gt != 3
    res = {}
    for k in NAMES:
        m = s[k] & g
        e = np.flatnonzero(np.diff(np.concatenate(([0], m.astype(np.int8), [0]))))
        ep_dead = [np.mean(gt[a:b] == 2) >= 0.5 for a, b in zip(e[::2], e[1::2])]
        res[k] = (int(m.sum()), int((m & (gt == 2)).sum()), len(ep_dead), int(np.sum(ep_dead)))
    anyl = np.zeros(len(gt), bool)
    for k in NAMES:
        anyl |= s[k]
    nob = np.isnan(ball[:, 0]) & g
    res["dead frames"] = int((g & (gt == 2)).sum())
    res["dead covered"] = int((anyl & g & (gt == 2)).sum())
    res["dead no ball"] = int((nob & (gt == 2)).sum())
    res["dead no ball covered"] = int((anyl & nob & (gt == 2)).sum())
    return res


def main():
    from multiprocessing import Pool
    import idsse
    import metrica
    for cond in ("clean", "realistic"):
        runs = [((g, 1), "tracab", cond) for g in idsse.GAMES] + [((g, 1), None, cond) for g in metrica.GAMES]
        with Pool(5) as pool:
            R = pool.map(_job, runs)
        print("\n== {} tracking, {} external matches".format(cond, len(R)))
        for k in NAMES:
            fr, dead, ep, epd = (sum(r[k][i] for r in R) for i in range(4))
            print("  {:8s} {:6.1f} s per match, dead ball on {:5.1f}% of its frames; {:4d} episodes, {:5.1f}% mostly dead".format(
                k, fr / FPS / len(R), 100 * dead / max(fr, 1), ep, 100 * epd / max(ep, 1)))
        tot = {k: sum(r[k] for r in R) for k in ("dead frames", "dead covered", "dead no ball", "dead no ball covered")}
        print("  any layout covers {:.1f}% of the dead frames ({:.1f}% of those without the ball)".format(
            100 * tot["dead covered"] / tot["dead frames"], 100 * tot["dead no ball covered"] / max(tot["dead no ball"], 1)))


if __name__ == "__main__":
    main()
