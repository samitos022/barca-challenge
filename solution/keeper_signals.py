"""Signals for "the ball vanished because the goalkeeper has it in his hands".

When the tracker loses the ball in a keeper's hands, play goes on: the keeper holds it a few seconds,
walks towards the edge of his box, the opponents leave the box, then he throws or kicks it and the
ball reappears moving fast away from him. A stoppage looks different: nobody holds the ball, the
players stop, it reappears still on a line or a spot. These signals describe the ball gap the frame
is in from the keepers' point of view, per team (..._home / ..._away, team-symmetric; prefix "kp_"):

  kp_vanish_d      distance between the last seen ball and the team's keeper (its deepest player) at
                   that moment
  kp_vanish_box    1 if the ball was last seen inside the team's own penalty area
  kp_keeper_goal   keeper's distance from his goal line now
  kp_keeper_adv    how far the keeper has moved away from his goal line since the ball vanished
  kp_keeper_speed  keeper's speed now (median over 1 s, from his own positions)
  kp_opp_box_now / kp_opp_box_drop   opponents inside the team's box now, and how many fewer than
                   when the ball vanished (attackers leave the box after a catch)
  kp_reappear_d    distance between where the ball reappears and the keeper at that moment
  kp_reappear_v    speed of the ball over the first 0.5 s after it reappears (a throw or a kick)
  kp_reappear_away component of that velocity away from the team's own goal (m/s)
Scalars: kp_in_gap, kp_gap_elapsed_s, kp_gap_left_s.

Every value is NaN outside a ball gap (and the reappearance ones in a gap with no reappearance).
Positions only; no label is used, so the same code runs on any match.
"""

import numpy as np

FPS = 25
BOX_X, BOX_Y = 16.5, 20.16
MIN_GAP = 5            # frames: shorter gaps are flicker, not a lost ball


def _goal_x(teams, players, blocks, pitch, n):
    L = pitch[0]
    out = np.full((n, 2), np.nan, dtype=np.float32)
    for _, a, z in blocks:
        mx = [np.nanmean(players[a:z + 1][:, teams == k, 0]) for k in (0, 1)]
        out[a:z + 1] = [0.0, L] if mx[0] < mx[1] else [L, 0.0]
    ok = ~np.isnan(out[:, 0])
    if ok.any():
        idx = np.where(ok, np.arange(n), -1)
        last = np.maximum.accumulate(idx)
        last[last < 0] = np.flatnonzero(ok)[0]
        out = out[last]
    return out


def features(teams, players, ball, blocks, pitch):
    L, W = pitch
    n = len(ball)
    P = players.astype(np.float32)
    gx = _goal_x(teams, P, blocks, pitch, n)
    seen = ~np.isnan(ball[:, 0])
    names, cols = [], {}

    def col(name, k=None):
        nm = "kp_" + name + ("" if k is None else ("_home" if k == 0 else "_away"))
        if nm not in cols:
            names.append(nm)
            cols[nm] = np.full(n, np.nan, dtype=np.float32)
        return cols[nm]

    # each team's keeper per frame: its deepest tracked player, and his own speed
    kpos = np.full((n, 2, 2), np.nan, dtype=np.float32)
    kspeed = np.full((n, 2), np.nan, dtype=np.float32)
    for k in (0, 1):
        m = np.flatnonzero(teams == k)
        dep = np.abs(P[:, m, 0] - gx[:, k][:, None])
        dep = np.where(np.isnan(dep), np.inf, dep)
        j = np.argmin(dep, axis=1)
        has = np.isfinite(dep[np.arange(n), j])
        slot = m[j]
        kpos[has, k] = P[np.flatnonzero(has), slot[has]]
        # speed of that same slot over +-2 frames
        a, b = np.clip(np.arange(n) - 2, 0, n - 1), np.clip(np.arange(n) + 2, 0, n - 1)
        d = np.hypot(*(P[b, slot] - P[a, slot]).T) * FPS / 4.0
        kspeed[has, k] = np.where(d[has] < 12.0, d[has], np.nan)
    ks1 = np.stack([np.convolve(np.nan_to_num(kspeed[:, k], nan=0.0), np.ones(FPS) / FPS, mode="same") for k in (0, 1)], 1)
    in_box = np.zeros((n, 2), dtype=np.float32)        # opponents inside each team's box
    for k in (0, 1):
        m = teams == 1 - k
        x, y = P[:, m, 0], P[:, m, 1]
        in_box[:, k] = np.sum((np.abs(x - gx[:, k][:, None]) <= BOX_X) & (np.abs(y - W / 2) <= BOX_Y), axis=1)
    e = np.flatnonzero(np.diff(np.concatenate(([0], (~seen).astype(np.int8), [0]))))
    for a, b in zip(e[::2], e[1::2]):
        if b - a < MIN_GAP or a == 0:
            continue
        last = a - 1
        sl = slice(a, b)
        col("in_gap")[sl] = 1.0
        col("gap_elapsed_s")[sl] = (np.arange(a, b) - a) / FPS
        col("gap_left_s")[sl] = (b - np.arange(a, b)) / FPS
        bl = ball[last]
        re = b if b < n else None
        for k in (0, 1):
            kp0 = kpos[last, k]
            col("vanish_d", k)[sl] = np.hypot(*(bl - kp0))
            col("vanish_box", k)[sl] = float(abs(bl[0] - gx[last, k]) <= BOX_X and abs(bl[1] - W / 2) <= BOX_Y)
            g_now = np.abs(kpos[sl, k, 0] - gx[sl, k])
            col("keeper_goal", k)[sl] = g_now
            col("keeper_adv", k)[sl] = g_now - abs(kp0[0] - gx[last, k])
            col("keeper_speed", k)[sl] = ks1[sl, k]
            col("opp_box_now", k)[sl] = in_box[sl, k]
            col("opp_box_drop", k)[sl] = in_box[last, k] - in_box[sl, k]
            if re is not None:
                col("reappear_d", k)[sl] = np.hypot(*(ball[re] - kpos[re, k]))
                e2 = min(n - 1, re + FPS // 2)
                if seen[e2]:
                    v = (ball[e2] - ball[re]) * FPS / max(e2 - re, 1)
                    col("reappear_v", k)[sl] = np.hypot(*v)
                    away = 1.0 if gx[re, k] < L / 2 else -1.0
                    col("reappear_away", k)[sl] = v[0] * away
    F = np.stack([cols[nm] for nm in names], axis=1).astype(np.float32)
    F[~np.isfinite(F)] = np.nan
    return F, names
