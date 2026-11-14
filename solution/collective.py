"""Collective signals of the two teams, per frame: what the players do together, read so that a
player who disappears, reappears or swaps identity does not look like a tactical movement.

Robustness rules:
  - every MOVEMENT is measured per player, from his own position before and after (central
    difference over VEL_HALF frames, both ends tracked, faster than MAX_SPEED = a tracking jump,
    dropped), and only then summarised over the team with a median or a share. A centroid that
    moves because one player vanished therefore does not move here.
  - every SHAPE is a robust statistic (median, interquartile range) of the players tracked now.
  - every group of signals comes with its confidence: how many players it was computed on, and how
    many of the team's tracks appear or disappear around the frame ("churn").

Features are team-symmetric: per-team quantities are named ..._home / ..._away (possession_ml.
swap_teams exchanges them), and everything with a direction is measured towards the team's own
attacking goal, so the same football gives the same numbers for either team. Prefix "col_".

Groups (per team unless said otherwise):
  speeds       median, 90th percentile, share still (< STILL m/s), share running (> RUN m/s),
               change of pace next 1 s vs past 1 s (paired players), synchronous braking (share of
               players whose speed halves), pace against the match's own last 30 s and 120 s
  centroid     velocity of the team (median of the players' velocities): speed, component towards
               the attacking goal; distance between the two centroids; centroid relative to the ball
  coordination polarisation (how aligned the moving players are), share moving towards the ball,
               share retreating towards their own goal, radial speed towards the ball (median)
  shape        width and depth (interquartile ranges), compactness, height of the last line and of
               the front line from their own goal, players in each penalty area
  pressure     distance of the nearest opponent to the team's player closest to the ball, opponents
               within 5 m of him, and its change over the last second (pressure released)
  ball/players ball displacement over the last and next second against the change of the players'
               pace: a ball moved about while nobody reacts
  set pieces   continuous scores rather than yes / no (layout.py has the binary ones): slow player
               nearest a corner flag and players in that box; slow player nearest a touchline;
               ball near a goal area; largest group of standing team-mates; share of players in
               their own half and in the centre circle (kick-off)
  windows      means over 2 s and 5 s and next-2 s minus past-2 s for the main signals, so the model
               sees sequences of changes and not only a still picture.
"""

import numpy as np

FPS = 25
VEL_HALF = 3          # frames each side for a player's velocity (0.24 s)
MAX_SPEED = 12.0      # m/s, faster is a tracking jump
STILL, RUN, MOVING = 0.5, 4.0, 1.0
PRESS_R = 5.0
BOX_X, BOX_Y = 16.5, 20.16
CHUNK = 4000


def _nanmean_window(x, half):
    v = np.nan_to_num(x, nan=0.0)
    c = (~np.isnan(x)).astype(np.float64)
    cv = np.concatenate(([0.0], np.cumsum(v)))
    cc = np.concatenate(([0.0], np.cumsum(c)))
    n = len(x)
    lo = np.clip(np.arange(n) - half, 0, n)
    hi = np.clip(np.arange(n) + half + 1, 0, n)
    cnt = cc[hi] - cc[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (cv[hi] - cv[lo]) / cnt, np.nan)


def _shifted_mean(x, start, stop):
    v = np.nan_to_num(x, nan=0.0)
    c = (~np.isnan(x)).astype(np.float64)
    cv = np.concatenate(([0.0], np.cumsum(v)))
    cc = np.concatenate(([0.0], np.cumsum(c)))
    n = len(x)
    lo = np.clip(np.arange(n) + start, 0, n)
    hi = np.clip(np.arange(n) + stop, 0, n)
    cnt = cc[hi] - cc[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (cv[hi] - cv[lo]) / cnt, np.nan)


def _per_slot_window(a, start, stop):
    """(n, S) mean of a over frames t+start .. t+stop-1 per slot, NaN-aware."""
    v = np.nan_to_num(a, nan=0.0)
    c = (~np.isnan(a)).astype(np.float32)
    cv = np.concatenate((np.zeros((1, a.shape[1]), np.float64), np.cumsum(v, axis=0)))
    cc = np.concatenate((np.zeros((1, a.shape[1]), np.float32), np.cumsum(c, axis=0)))
    n = len(a)
    lo = np.clip(np.arange(n) + start, 0, n)
    hi = np.clip(np.arange(n) + stop, 0, n)
    cnt = cc[hi] - cc[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (cv[hi] - cv[lo]) / cnt, np.nan).astype(np.float32)


def _rolling_past_mean(x, span):
    return _shifted_mean(x, -span, 0)


def goal_x(teams, players, blocks, pitch, n):
    """(n, 2) x of the goal each team defends."""
    L = pitch[0]
    out = np.full((n, 2), np.nan, dtype=np.float32)
    for _, a, z in blocks:
        mx = [np.nanmean(players[a:z + 1][:, teams == k, 0]) for k in (0, 1)]
        out[a:z + 1] = [0.0, L] if mx[0] < mx[1] else [L, 0.0]
    # outside the blocks: the nearest block's sides
    ok = ~np.isnan(out[:, 0])
    if ok.any():
        idx = np.where(ok, np.arange(n), -1)
        last = np.maximum.accumulate(idx)
        first = np.flatnonzero(ok)[0]
        last[last < 0] = first
        out = out[last]
    return out


def features(teams, players, ball, blocks, pitch):
    """(F (n, k) float32, names)."""
    L, W = pitch
    n, S = players.shape[:2]
    P = players.astype(np.float32)
    ok = ~np.isnan(P[..., 0])
    # per-player velocity, both ends tracked, jumps dropped
    V = np.full((n, S, 2), np.nan, dtype=np.float32)
    h = VEL_HALF
    V[h:-h] = (P[2 * h:] - P[:-2 * h]) * (FPS / (2.0 * h))
    sp = np.linalg.norm(V, axis=2)
    bad = sp > MAX_SPEED
    V[bad] = np.nan
    sp[bad] = np.nan
    gx = goal_x(teams, P, blocks, pitch, n)
    att = np.sign(np.where(gx[:, 0] < L / 2, 1.0, -1.0)).astype(np.float32)   # home attacks towards +x if it defends x=0
    dirk = np.stack([att, -att], axis=1)                                         # (n, 2) attacking direction of each team
    sp_past = _per_slot_window(sp, -FPS, 0)
    sp_next = _per_slot_window(sp, 1, FPS + 1)
    present_past = _per_slot_window(ok.astype(np.float32), -FPS, -FPS + 1)
    present_next = _per_slot_window(ok.astype(np.float32), FPS, FPS + 1)
    cols, names = {}, []

    def put(name, arr=None, k=None):
        nm = "col_" + name + ("" if k is None else ("_home" if k == 0 else "_away"))
        if nm not in cols:
            names.append(nm)
            cols[nm] = np.full(n, np.nan, dtype=np.float32)
        return cols[nm]

    bvis = ~np.isnan(ball[:, 0])
    with np.errstate(invalid="ignore", divide="ignore"):
        for c0 in range(0, n, CHUNK):
            sl = slice(c0, min(n, c0 + CHUNK))
            x, y = P[sl, :, 0], P[sl, :, 1]
            okc, vx, vy, s = ok[sl], V[sl, :, 0], V[sl, :, 1], sp[sl]
            bx, by = ball[sl, 0], ball[sl, 1]
            gxc, dc = gx[sl], dirk[sl]
            cen = []
            for k in (0, 1):
                m = okc & (teams == k)[None]
                xm, ym = np.where(m, x, np.nan), np.where(m, y, np.nan)
                nk = m.sum(1).astype(np.float32)
                put("n", None, k)[sl] = nk
                mv = m & ~np.isnan(s)
                put("n_vel", None, k)[sl] = mv.sum(1)
                sk = np.where(mv, s, np.nan)
                put("speed_med", None, k)[sl] = np.nanmedian(sk, 1)
                put("speed_p90", None, k)[sl] = np.nanpercentile(sk, 90, axis=1)
                cnt = np.maximum(mv.sum(1), 1)
                put("still", None, k)[sl] = np.where(mv.sum(1) > 0, np.sum(mv & (s < STILL), 1) / cnt, np.nan)
                put("running", None, k)[sl] = np.where(mv.sum(1) > 0, np.sum(mv & (s > RUN), 1) / cnt, np.nan)
                # pace change and synchronous braking, paired per player
                pp, pn = sp_past[sl], sp_next[sl]
                pair = (teams == k)[None] & ~np.isnan(pp) & ~np.isnan(pn)
                put("pace_change", None, k)[sl] = np.nanmedian(np.where(pair, pn - pp, np.nan), 1)
                brake = pair & (pp > 2.0) & (pn < 0.5 * pp)
                put("brake_share", None, k)[sl] = np.where(pair.sum(1) > 0, brake.sum(1) / np.maximum(pair.sum(1), 1), np.nan)
                # churn: tracks of this team appearing / disappearing within +-1 s
                ch = (teams == k)[None] & (present_past[sl] != present_next[sl]) & ~np.isnan(present_past[sl]) & ~np.isnan(present_next[sl])
                put("churn", None, k)[sl] = ch.sum(1)
                # centroid and its velocity (median of the players' velocities)
                cx, cy = np.nanmedian(xm, 1), np.nanmedian(ym, 1)
                cen.append((cx, cy))
                mvx, mvy = np.nanmedian(np.where(mv, vx, np.nan), 1), np.nanmedian(np.where(mv, vy, np.nan), 1)
                put("cen_speed", None, k)[sl] = np.hypot(mvx, mvy)
                put("cen_v_attack", None, k)[sl] = mvx * dc[:, k]
                put("cen_ball_attack", None, k)[sl] = (cx - bx) * dc[:, k]
                # coordination among the moving players
                movp = mv & (s > MOVING)
                ux, uy = np.where(movp, vx / s, np.nan), np.where(movp, vy / s, np.nan)
                nm_ = movp.sum(1)
                put("n_moving", None, k)[sl] = nm_
                put("polar", None, k)[sl] = np.where(nm_ >= 3, np.hypot(np.nanmean(ux, 1), np.nanmean(uy, 1)), np.nan)
                dbx, dby = bx[:, None] - x, by[:, None] - y
                db = np.hypot(dbx, dby)
                cosb = (ux * dbx + uy * dby) / np.maximum(db, 0.1)
                put("to_ball", None, k)[sl] = np.where(nm_ >= 3, np.sum(movp & (cosb > 0.5), 1) / np.maximum(nm_, 1), np.nan)
                retreat = (ux * (-dc[:, k][:, None])) > 0.5
                put("retreat", None, k)[sl] = np.where(nm_ >= 3, np.sum(movp & retreat, 1) / np.maximum(nm_, 1), np.nan)
                radial = -(vx * dbx + vy * dby) / np.maximum(db, 0.1)       # > 0: moving away from the ball
                put("radial_ball", None, k)[sl] = np.nanmedian(np.where(mv, radial, np.nan), 1)
                # shape
                put("width", None, k)[sl] = np.nanpercentile(ym, 75, axis=1) - np.nanpercentile(ym, 25, axis=1)
                put("depth", None, k)[sl] = np.nanpercentile(xm, 75, axis=1) - np.nanpercentile(xm, 25, axis=1)
                put("compact", None, k)[sl] = np.nanmedian(np.hypot(xm - cx[:, None], ym - cy[:, None]), 1)
                dg = np.abs(xm - gxc[:, k][:, None])
                dgs = np.sort(np.where(np.isnan(dg), np.inf, dg), axis=1)
                put("last_line", None, k)[sl] = np.where(nk >= 3, dgs[:, 1], np.nan)
                put("front_line", None, k)[sl] = np.nanmax(dg, 1)
                own_box = m & (np.abs(x - gxc[:, k][:, None]) <= BOX_X) & (np.abs(y - W / 2) <= BOX_Y)
                opp_box = m & (np.abs(x - gxc[:, 1 - k][:, None]) <= BOX_X) & (np.abs(y - W / 2) <= BOX_Y)
                put("in_own_box", None, k)[sl] = own_box.sum(1)
                put("in_opp_box", None, k)[sl] = opp_box.sum(1)
                # pressure on the team's player closest to the ball
                dbm = np.where(m, db, np.inf)
                j = np.argmin(dbm, 1)
                has = np.isfinite(dbm[np.arange(len(j)), j]) & bvis[sl]
                px, py = x[np.arange(len(j)), j], y[np.arange(len(j)), j]
                mo = okc & (teams == 1 - k)[None]
                do = np.where(mo, np.hypot(x - px[:, None], y - py[:, None]), np.inf)
                put("press_d", None, k)[sl] = np.where(has & np.isfinite(do.min(1)), do.min(1), np.nan)
                put("press_n", None, k)[sl] = np.where(has, np.sum(do < PRESS_R, 1), np.nan)
                # own half share and centre circle (kick-off)
                half = m & ((x - L / 2) * (-dc[:, k][:, None]) >= -1.0)
                put("own_half", None, k)[sl] = np.where(nk > 0, half.sum(1) / np.maximum(nk, 1), np.nan)
                # largest group of standing team-mates (a wall): max over players of standing neighbours within 2 m
                st = m & (np.nan_to_num(s, nan=9.0) < 0.7)
                xs, ys_ = np.where(st, x, np.nan), np.where(st, y, np.nan)
                dd = np.hypot(xs[:, :, None] - xs[:, None, :], ys_[:, :, None] - ys_[:, None, :])
                put("stand_group", None, k)[sl] = np.max(np.sum(dd <= 2.0, axis=2), axis=1)
            put("cen_dist")[sl] = np.hypot(cen[0][0] - cen[1][0], cen[0][1] - cen[1][1])
            allok = okc
            put("n_all")[sl] = allok.sum(1)
            put("centre_circle")[sl] = np.sum(allok & (np.hypot(x - L / 2, y - W / 2) <= 10.0), 1)
            slow = allok & (np.nan_to_num(s, nan=9.0) < 1.5)
            dflag = np.full(x.shape[0], np.inf)
            boxn = np.full(x.shape[0], np.nan)
            for gx0 in (0.0, L):
                inb = np.sum(allok & (np.abs(x - gx0) <= BOX_X) & (np.abs(y - W / 2) <= BOX_Y), 1)
                for fy in (0.0, W):
                    d = np.min(np.where(slow, np.hypot(x - gx0, y - fy), np.inf), 1)
                    better = d < dflag
                    dflag = np.where(better, d, dflag)
                    boxn = np.where(better, inb, boxn)
            put("corner_d")[sl] = np.where(np.isfinite(dflag), np.minimum(dflag, 60.0), np.nan)
            put("corner_box")[sl] = boxn
            dtl = np.min(np.where(slow, np.minimum(y, W - y), np.inf), 1)
            put("touch_d")[sl] = np.where(np.isfinite(dtl), np.clip(dtl, -3.0, 40.0), np.nan)
            ga = np.minimum(np.abs(bx - 0.0), np.abs(bx - L))
            put("ball_goal_area")[sl] = np.where(bvis[sl], np.hypot(np.maximum(ga - 5.5, 0), np.maximum(np.abs(by - W / 2) - 9.16, 0)), np.nan)
    # match context and ball / players reaction (whole-array)
    pace = np.nanmean(np.stack([cols["col_speed_med_home"], cols["col_speed_med_away"]]), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        put("pace_vs_30s")[:] = pace / np.maximum(_rolling_past_mean(pace, 30 * FPS), 0.3)
        put("pace_vs_120s")[:] = pace / np.maximum(_rolling_past_mean(pace, 120 * FPS), 0.3)
    bd_past = np.full(n, np.nan, dtype=np.float32)
    bd_next = np.full(n, np.nan, dtype=np.float32)
    bd_past[FPS:] = np.hypot(*(ball[FPS:] - ball[:-FPS]).T)
    bd_next[:-FPS] = bd_past[FPS:]
    put("ball_move_past1s")[:] = bd_past
    put("ball_move_next1s")[:] = bd_next
    put("react")[:] = np.nanmean(np.stack([cols["col_pace_change_home"], cols["col_pace_change_away"]]), axis=0)
    # windows: sequences of changes
    for base in ("speed_med", "still", "running", "brake_share", "cen_speed", "cen_v_attack", "polar", "to_ball",
                 "retreat", "width", "depth", "compact", "last_line", "press_d", "own_half", "stand_group"):
        for k in (0, 1):
            nm = "col_{}_{}".format(base, "home" if k == 0 else "away")
            x_ = cols[nm].astype(np.float64)
            put("{}_c2s".format(base), None, k)[:] = _nanmean_window(x_, FPS)
            put("{}_c5s".format(base), None, k)[:] = _nanmean_window(x_, int(2.5 * FPS))
            with np.errstate(invalid="ignore"):
                put("{}_trend".format(base), None, k)[:] = _shifted_mean(x_, 1, 2 * FPS + 1) - _shifted_mean(x_, -2 * FPS, 0)
    for base in ("corner_d", "corner_box", "touch_d", "ball_goal_area", "centre_circle", "cen_dist", "pace_vs_30s", "react"):
        x_ = cols["col_" + base].astype(np.float64)
        put(base + "_c2s")[:] = _nanmean_window(x_, FPS)
        with np.errstate(invalid="ignore"):
            put(base + "_trend")[:] = _shifted_mean(x_, 1, 2 * FPS + 1) - _shifted_mean(x_, -2 * FPS, 0)
    F = np.stack([cols[nm] for nm in names], axis=1).astype(np.float32)
    F[~np.isfinite(F)] = np.nan
    return F, names
