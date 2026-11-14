"""Set pieces read from the players alone, and WHICH team will take them.

After a stoppage "there is a clear possessor" (CHALLENGE.md): the team that takes the restart owns
the ball from its first touch, even if the other team clears it a second later. layout.py says
that a set piece is coming; these signals also say for whom, per team (..._home / ..._away = "this
team will restart", team-symmetric; prefix "sp_"), and work without the ball and without the taker
(at a corner the flag is often outside the tactical camera's image):

  corner   a slow player of the team within CORNER_R of a corner flag at the end it attacks, with
           the box at that end full (layout.py's corner, with the taker's team)
  crowd    the box the team attacks full and slow: CROWD_N or more players inside it, at least
           CROWD_SIDE of each team, median pace under CROWD_V, for CROWD_S. A corner or a free kick
           into the box, waiting to be taken, seen even when the taker is not tracked
  penalty  one player of the team within PEN_ARC of the penalty spot of the goal it attacks (the
           kicker: before the kick he stands back for the run-up, not on the spot); nobody else
           inside the box or the arc (PEN_IN_M of tracking noise at their lines) but at most one
           player on the goal line (the keeper); PEN_EDGE or more players gathered just outside the
           box; slow. Or the same with the kicker untracked: nobody in the box and the arc but the
           other team's keeper on his line, PEN_SIDE of each team at the edge of the box (13/11).
           A goal kick, the layout it is most like, has the keeper forward with the ball and
           defenders inside the box, nobody on the spot
  throw    the slow player of the team furthest out on a touchline (layout.py's throw-in, with the
           thrower's team)
  wall     a wall of WALL_N standing players of the OTHER team (layout.py's wall): a free kick for
           this team

For each one: sp_<name>_<team> (1 on the frames of the layout) and sp_<name>_since_s_<team>
(seconds since the layout was last seen, capped at CAP_S). For all of them together:
sp_any_c125_<team> (share of the last 5 s with any) and sp_any_until_s_<team>.

Positions only, no label and no ball: the same code runs on any match. How often each layout
means a dead ball, and how often its team takes the next restart, is measured on the external
matches by analisi/calci_piazzati.py check (thresholds are physical, nothing from BAR-ATH).
"""

import warnings

import numpy as np

FPS = 25
BOX_X, BOX_Y = 16.5, 20.16
SPOT = 11.0
CORNER_R, CORNER_V, CORNER_BOX, CORNER_S = 3.0, 1.5, 6, 1.5
CROWD_N, CROWD_SIDE, CROWD_V, CROWD_S = 9, 3, 1.3, 2.0
PEN_EDGE, PEN_EDGE_M, PEN_V, PEN_S, PEN_IN_M = 7, 10.0, 1.2, 1.5, 1.0
PEN_ARC = 9.15        # m from the spot: the arc everybody but the kicker must stay out of
PEN_GK_DEPTH, PEN_GK_Y = 2.0, 4.7    # the keeper on his line, between the posts (+1 m)
PEN_SIDE = 3          # players of each team at the edge of the box when the kicker is untracked
THROW_OUT, THROW_V, THROW_S = 0.3, 1.0, 1.5
WALL_N, WALL_GAP, WALL_V, WALL_S = 3, 1.3, 0.7, 2.0
CAP_S = 30.0
NAMES = ("corner", "crowd", "penalty", "throw", "wall")


def _hold(mask, k):
    out = np.zeros_like(mask)
    e = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    for a, b in zip(e[::2], e[1::2]):
        if b - a >= k:
            out[a:b] = True
    return out


def _since(mask, cap=CAP_S):
    n = len(mask)
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(mask, idx, -10 ** 9))
    return np.clip((idx - last) / FPS, 0, cap).astype(np.float32)


def _until(mask, cap=CAP_S):
    n = len(mask)
    idx = np.arange(n)
    nxt = np.minimum.accumulate(np.where(mask, idx, 10 ** 9)[::-1])[::-1]
    return np.clip((nxt - idx) / FPS, 0, cap).astype(np.float32)


def _window_mean(x, past):
    c = np.concatenate(([0.0], np.cumsum(x.astype(np.float64))))
    i = np.arange(len(x))
    lo = np.maximum(i - past, 0)
    return ((c[i + 1] - c[lo]) / (i + 1 - lo)).astype(np.float32)


def layouts(teams, players, blocks, pitch):
    """{name: (n, 2) bool}: column k = the layout says team k will restart."""
    L, W = pitch
    n = len(players)
    out = {k: np.zeros((n, 2), dtype=bool) for k in NAMES}
    ok = ~np.isnan(players[..., 0])
    spd = np.full(players.shape[:2], np.inf, dtype=np.float32)
    spd[2:-2] = np.linalg.norm(players[4:] - players[:-4], axis=2) * (FPS / 4.0)
    spd[~ok] = np.inf
    spd = np.nan_to_num(spd, nan=np.inf)
    for _, a, z in blocks:
        P, O, V = players[a:z + 1], ok[a:z + 1], spd[a:z + 1]
        x, y = P[..., 0], P[..., 1]
        mx = [np.nanmean(x[:, teams == k]) for k in (0, 1)]
        own = [0.0, L] if mx[0] < mx[1] else [L, 0.0]      # each team's own goal line in this half
        m = len(P)
        for k in (0, 1):
            g = own[1 - k]                                    # the goal team k attacks
            sgn = 1.0 if g == 0.0 else -1.0                   # pitch direction pointing away from g
            depth = (x - g) * sgn                             # metres in front of that goal line
            inbox = O & (depth >= -1.0) & (depth <= BOX_X) & (np.abs(y - W / 2) <= BOX_Y)
            mine, theirs = teams == k, teams != k

            # corner, taker of team k at a flag of that end
            cor = np.zeros(m, dtype=bool)
            for fy in (0.0, W):
                at = O[:, mine] & (np.hypot(x[:, mine] - g, y[:, mine] - fy) <= CORNER_R) & (V[:, mine] <= CORNER_V)
                cor |= at.any(axis=1)
            cor &= inbox.sum(axis=1) >= CORNER_BOX
            out["corner"][a:z + 1, k] = _hold(cor, int(CORNER_S * FPS))

            # crowd: that box full of both teams, slow
            nk, no = inbox[:, mine].sum(axis=1), inbox[:, theirs].sum(axis=1)
            vb = np.where(inbox, V, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                vmed = np.nanmedian(np.where(np.isfinite(vb), vb, np.nan), axis=1)
            crowd = (nk + no >= CROWD_N) & (nk >= CROWD_SIDE) & (no >= CROWD_SIDE) & (np.nan_to_num(vmed, nan=9.0) <= CROWD_V)
            out["crowd"][a:z + 1, k] = _hold(crowd, int(CROWD_S * FPS))

            # penalty: the kicker on the spot, the keeper on his line, everybody else outside the box
            # and outside the arc, gathered at the edge of the box. The arc gets the same PEN_IN_M of
            # tracking noise as the box lines (players stand right on it). The kicker may be
            # untracked (he is often the only player outside a camera's framing): then the box and
            # the arc are empty but for the other team's keeper, alone on his line, and both teams
            # line the box (PEN_SIDE each: after a goal, before the kick-off, the box is empty too)
            d_spot = np.hypot(depth - SPOT, y - W / 2)
            keeper = inbox & (depth <= PEN_GK_DEPTH) & (np.abs(y - W / 2) <= PEN_GK_Y)
            in_arc = O & ~keeper & (d_spot < PEN_ARC - PEN_IN_M)
            kicker_k = (in_arc & mine[None, :]).sum(axis=1) == 1
            no_kicker = (in_arc.sum(axis=1) == 0) & (keeper.sum(axis=1) == 1) & ((keeper & theirs[None, :]).sum(axis=1) == 1)
            inner = O & ~keeper & ~in_arc & (depth >= -1.0) & (depth <= BOX_X - PEN_IN_M) & \
                (np.abs(y - W / 2) <= BOX_Y - PEN_IN_M)
            edge = O & ~keeper & ~in_arc & (depth > BOX_X - PEN_IN_M) & (depth <= BOX_X + PEN_EDGE_M) & \
                (np.abs(y - W / 2) <= BOX_Y + PEN_EDGE_M)
            vall = np.where(O, V, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                vmean = np.nanmean(np.where(np.isfinite(vall), vall, np.nan), axis=1)
            no_kicker &= ((edge & mine[None, :]).sum(axis=1) >= PEN_SIDE) & ((edge & theirs[None, :]).sum(axis=1) >= PEN_SIDE)
            pen = ((kicker_k & (in_arc.sum(axis=1) == 1) & (keeper.sum(axis=1) <= 1)) | no_kicker) & \
                ~inner.any(axis=1) & (edge.sum(axis=1) >= PEN_EDGE) & (np.nan_to_num(vmean, nan=9.0) <= PEN_V)
            out["penalty"][a:z + 1, k] = _hold(pen, int(PEN_S * FPS))

            # wall of the other team (team k takes the free kick)
            j = np.flatnonzero(theirs)
            st = O[:, j] & (V[:, j] <= WALL_V)
            xx, yy = np.where(st, x[:, j], np.nan), np.where(st, y[:, j], np.nan)
            with np.errstate(invalid="ignore"):
                near = np.hypot(xx[:, :, None] - xx[:, None, :], yy[:, :, None] - yy[:, None, :]) <= WALL_GAP
            deg = near.sum(axis=2) - st
            out["wall"][a:z + 1, k] = _hold(((deg >= WALL_N - 1) & st).any(axis=1), int(WALL_S * FPS))

        # throw-in: the slow player furthest out on a touchline, away from the corners; his team
        outward = np.where(O & (V <= THROW_V) & (np.minimum(x, L - x) > CORNER_R),
                           np.maximum(THROW_OUT - y, y - (W - THROW_OUT)), -np.inf)
        best = np.argmax(outward, axis=1)
        has = outward[np.arange(m), best] >= 0
        for k in (0, 1):
            out["throw"][a:z + 1, k] = _hold(has & (teams[best] == k), int(THROW_S * FPS))
    return out


def features(teams, players, ball, blocks, pitch):
    lay = layouts(teams, players, blocks, pitch)
    names, cols = [], []
    for k, side in ((0, "_home"), (1, "_away")):
        anyl = np.zeros(len(players), dtype=bool)
        for nm in NAMES:
            m = lay[nm][:, k]
            anyl |= m
            names += ["sp_{}{}".format(nm, side), "sp_{}_since_s{}".format(nm, side)]
            cols += [m.astype(np.float32), _since(m)]
        names += ["sp_any_c125" + side, "sp_any_until_s" + side]
        cols += [_window_mean(anyl, 125), _until(anyl)]
    F = np.stack(cols, axis=1).astype(np.float32)
    return F, names
