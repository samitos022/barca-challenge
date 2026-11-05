"""Possession v1: who has the ball, decided by an evidence score with a threshold.

This first version only decides possession CHANGES (H -> A, A -> H). It has no dead ball yet:
inside the graded halves every frame is H or A, outside them X. Dead ball (H/A -> D, D -> H/A)
comes next, with its own rules.

How a change is decided
-----------------------
At every frame the team NOT in possession (the challenger) collects points; the team in possession
takes points away from it. When the challenger reaches THRESHOLD, possession changes, and the change
is dated back to the frame where the challenger's evidence started (the steal, not the moment we
became sure of it). Four rules feed the score:

  1. Control (slow ball).  The challenger's player is the closest to the ball, close enough to play
     it, clearly closer than the nearest opponent, and the ball is slow. Points per second of
     control, scaled by how well each of the three conditions holds.
  2. Touch (velocity change).  The ball's velocity VECTOR changes sharply (speed, direction or both)
     and the challenger has the player closest to it at that instant: a fixed bonus.
  3. Fast ball, no deviation.  A fast ball that keeps its line is a pass or a shot in flight: rule 1
     is switched off, so running past opponents does not count as losing the ball. Only a touch
     (rule 2) can take possession while the ball is fast.
  4. Look ahead.  At a challenger touch, the next LOOK_S seconds are checked: if the challenger
     then controls the ball, the touch was a steal (bonus); if the team in possession gets it back,
     it was a deflection (malus that cancels the touch).

The team in possession erases the challenger's score with its own control and touches, and the
score decays when nothing happens, so evidence from ten seconds ago cannot add up with today's.

Every parameter is a physical quantity (metres, m/s, seconds) set from how football works, not fitted
to any labels: the public 5 minutes are a correctness check, not a tuning set (RULES.md).

    .venv/bin/python solution/possession_v1.py

Writes evaluation/results/v1_possession.csv (the submission) and
data/derived/v1_diagnostics.npz (per-frame scores, used by viz/v1_side_by_side.py).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
sys.path.insert(0, HERE)
from games import GAME, half_video_blocks, source_frames  # noqa: E402
from baseline import write_submission  # noqa: E402
import tracking  # noqa: E402

FPS = 25
OUT_CSV = os.path.join(ROOT, "evaluation", "results", "v1_possession.csv")
OUT_DIAG = os.path.join(ROOT, "data", "derived", "v1_diagnostics.npz")

PARAMS = {
    # Rule 1, control
    "R_NEAR": 1.0,       # m: a player this close can play the ball (full weight)
    "R_FAR": 2.5,        # m: farther than this he cannot (zero weight); tracking noise sits between
    "MARGIN": 1.0,       # m: the challenger must be this much closer than the nearest opponent
    "V_SLOW": 4.0,       # m/s: a ball under control moves at most at running pace
    "V_FAST": 8.0,       # m/s: above this the ball is travelling (pass, shot, clearance)
    "SLOW_UNKNOWN": 0.5,  # weight of the speed condition when the speed cannot be measured
    "RATE_CTRL": 2.0,    # points per second of full control: half a second of clear control = 1
    "RATE_OWN": 2.0,     # points per second removed by the possessing team's own control
    # Rule 2, touch
    "DV_TOUCH": 3.5,     # m/s: change of the velocity vector (over 2 * DV_LAG frames) = a touch
    "DV_LAG": 3,         # frames each side of the touch used to compare velocities
    "VEL_HALF": 2,       # frames each side used to measure a velocity
    "R_TOUCH": 2.0,      # m: the toucher is the closest player, if within this distance
    "TOUCH_NMS": 4,      # frames: one touch per local maximum of the velocity change
    "W_TOUCH": 0.5,      # points for a challenger touch
    "W_TOUCH_OWN": 0.5,  # points removed by a touch of the possessing team
    # Rule 4, look ahead
    "LOOK_S": 2.0,       # s checked after a challenger touch
    "W_LOOK": 0.5,       # points: +W_LOOK if the challenger then owns the ball, -W_LOOK if not
    "LOOK_TOUCH_EQ": 0.4,  # s of control one touch is worth inside the look-ahead window
    "LOOK_MIN": 0.2,     # s of evidence needed in the window, else the look-ahead says nothing
    # Score
    "HALF_LIFE": 1.0,    # s: the challenger's score halves in this time if nothing happens
    "THRESHOLD": 1.0,    # points to change possession
    "GAP_MIN": 5,        # frames: if the evidence starts right after a ball gap this long,
                         # the change is dated to the middle of the gap
    "EPISODE_GAP": 0.5,  # s without challenger points that closes an episode: the change is
                         # dated to the start of the latest episode, not to stale evidence
}

HOME, AWAY, NONE = 0, 1, -1


def _shift(a, k):
    """out[t] = a[t + k], NaN where t + k falls outside."""
    out = np.full_like(a, np.nan)
    if k > 0:
        out[:-k] = a[k:]
    elif k < 0:
        out[-k:] = a[:k]
    else:
        out[:] = a
    return out


def ball_kinematics(ball, ball_speed, p):
    """Velocity, speed, velocity change and direction change of the ball, NaN where unknown."""
    pos = ball.astype(np.float64)
    smooth = (_shift(pos, -1) + pos + _shift(pos, 1)) / 3.0
    smooth = np.where(np.isnan(smooth), pos, smooth)  # run edges keep the raw point

    k = p["VEL_HALF"]
    vel = (_shift(smooth, k) - _shift(smooth, -k)) * FPS / (2.0 * k)
    speed = np.linalg.norm(vel, axis=1)
    measured = ~np.isnan(speed)
    fallback = ~measured & ~np.isnan(ball_speed)
    speed[fallback] = ball_speed[fallback]

    lag = p["DV_LAG"]
    before, after = _shift(vel, -lag), _shift(vel, lag)
    dv = np.linalg.norm(after - before, axis=1)
    nb, na = np.linalg.norm(before, axis=1), np.linalg.norm(after, axis=1)
    cos = np.sum(before * after, axis=1) / np.maximum(nb * na, 1e-9)
    angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    angle[(nb < 1.0) | (na < 1.0)] = np.nan  # direction is meaningless for a ball at rest
    return vel, speed, dv, angle


def team_distances(players, ball, teams):
    """Per team: distance of its closest player to the ball (m, NaN if none) and that slot."""
    d = np.linalg.norm(players - ball[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    dist = np.full((2, len(ball)), np.nan, dtype=np.float32)
    slot = np.full((2, len(ball)), -1, dtype=np.int16)
    for team in (HOME, AWAY):
        cols = np.flatnonzero(teams == team)
        sub = d[:, cols]
        j = np.argmin(sub, axis=1)
        best = sub[np.arange(len(ball)), j]
        ok = np.isfinite(best)
        dist[team, ok] = best[ok]
        slot[team, ok] = cols[j[ok]]
    return dist, slot


def control(dist, speed, visible, p):
    """Rule 1 and rule 3: per team, 0..1 how much it controls the ball at each frame."""
    near = np.clip((p["R_FAR"] - dist) / (p["R_FAR"] - p["R_NEAR"]), 0, 1)
    near = np.nan_to_num(near, nan=0.0)
    slow = np.clip((p["V_FAST"] - speed) / (p["V_FAST"] - p["V_SLOW"]), 0, 1)
    slow = np.where(np.isnan(speed), p["SLOW_UNKNOWN"], slow)
    ctrl = np.zeros_like(near)
    for team in (HOME, AWAY):
        other = np.nan_to_num(dist[1 - team], nan=np.inf)
        mine = np.nan_to_num(dist[team], nan=np.inf)
        with np.errstate(invalid="ignore"):
            margin = np.nan_to_num(np.clip((other - mine) / p["MARGIN"], 0, 1), nan=0.0)
        ctrl[team] = near[team] * slow * margin
    ctrl[:, ~visible] = 0
    return ctrl, slow


def touches(dv, dist, p):
    """Rule 2: frames where the velocity vector jumps, and the team that touched (NONE if unclear)."""
    x = np.nan_to_num(dv, nan=0.0)
    w = p["TOUCH_NMS"]
    padded = np.pad(x, w)
    local_max = np.lib.stride_tricks.sliding_window_view(padded, 2 * w + 1).max(axis=1)
    is_touch = (x >= p["DV_TOUCH"]) & (x == local_max)
    team = np.full(len(dv), NONE, dtype=np.int8)
    d = np.nan_to_num(dist, nan=np.inf)
    closest = np.argmin(d, axis=0)
    close_enough = np.min(d, axis=0) <= p["R_TOUCH"]
    hit = is_touch & close_enough
    team[hit] = closest[hit]
    return team, is_touch


def look_ahead(ctrl, touch_team, p):
    """Rule 4: for each frame, -1..1 how much the next LOOK_S seconds belong to each team."""
    n = ctrl.shape[1]
    w = int(round(p["LOOK_S"] * FPS))
    eq = p["LOOK_TOUCH_EQ"] * FPS
    conf = np.zeros((2, n), dtype=np.float32)
    evidence = []
    for team in (HOME, AWAY):
        e = ctrl[team] + eq * (touch_team == team)
        evidence.append(np.concatenate(([0.0], np.cumsum(e))))
    t = np.arange(n)
    hi = np.minimum(n, t + 1 + w)
    lo = t + 1
    lo = np.minimum(lo, n)
    sums = [ev[hi] - ev[lo] for ev in evidence]
    total = sums[0] + sums[1]
    ok = total >= p["LOOK_MIN"] * FPS
    for team in (HOME, AWAY):
        c = np.zeros(n)
        c[ok] = (sums[team][ok] - sums[1 - team][ok]) / total[ok]
        conf[team] = c
    return conf


def decide(ctrl, touch_team, conf, visible, blocks, n, p):
    """The score state machine. Returns per-frame possessor, scores, point breakdown, switches."""
    decay = 0.5 ** (1.0 / (p["HALF_LIFE"] * FPS))
    rate_ctrl, rate_own = p["RATE_CTRL"] / FPS, p["RATE_OWN"] / FPS
    w_touch, w_own, w_look, thr = p["W_TOUCH"], p["W_TOUCH_OWN"], p["W_LOOK"], p["THRESHOLD"]

    possessor = np.full(n, NONE, dtype=np.int8)
    live = np.full(n, NONE, dtype=np.int8)  # owner as the machine knew it at t, before back-dating
    score = np.zeros((2, n), dtype=np.float32)
    pts = {k: np.zeros((2, n), dtype=np.float32) for k in ("ctrl", "touch", "look", "own")}
    switches = []

    # last visible frame strictly before t, for dating changes that follow a gap
    idx = np.where(visible, np.arange(n), -1)
    last_seen = np.maximum.accumulate(idx)
    prev_seen = np.concatenate(([-1], last_seen[:-1]))

    ctrl_l = [ctrl[0].tolist(), ctrl[1].tolist()]
    conf_l = [conf[0].tolist(), conf[1].tolist()]
    touch_l = touch_team.tolist()

    for _, first, last in blocks:
        owner = NONE
        s = [0.0, 0.0]
        onset = [first, first]
        last_gain = [first - 10 ** 6, first - 10 ** 6]
        episode_gap = int(round(p["EPISODE_GAP"] * FPS))
        acc = [[0.0, 0.0], [0.0, 0.0]]  # per team: decayed points from control, from touch+look
        last_switch = first
        for t in range(first, last + 1):
            live[t] = owner
            for c in (HOME, AWAY):
                if c == owner:
                    s[c] = 0.0
                    continue
                g_ctrl = rate_ctrl * ctrl_l[c][t]
                g_touch = g_look = loss = 0.0
                if touch_l[t] == c:
                    g_touch = w_touch
                    g_look = w_look * conf_l[c][t]
                if owner != NONE:
                    loss = rate_own * ctrl_l[owner][t]
                    if touch_l[t] == owner:
                        loss += w_own
                new = s[c] * decay + g_ctrl + g_touch + g_look - loss
                if new <= 0.0:
                    new = 0.0
                    acc[c] = [0.0, 0.0]
                else:
                    acc[c] = [acc[c][0] * decay + g_ctrl, acc[c][1] * decay + g_touch + g_look]
                gain = g_ctrl + g_touch + g_look
                if gain > 0.0:
                    if s[c] < 0.02 or t - last_gain[c] > episode_gap:
                        onset[c] = t
                    last_gain[c] = t
                s[c] = new
                pts["ctrl"][c, t], pts["touch"][c, t] = g_ctrl, g_touch
                pts["look"][c, t], pts["own"][c, t] = g_look, loss
                score[c, t] = new
            winner = NONE
            if s[HOME] >= thr or s[AWAY] >= thr:
                winner = HOME if s[HOME] >= s[AWAY] else AWAY
            if winner != NONE:
                start = onset[winner]
                if start - prev_seen[start] > p["GAP_MIN"] and prev_seen[start] >= first:
                    start = (prev_seen[start] + start) // 2
                start = max(start, last_switch + 1 if owner != NONE else first)
                reason = "touch" if acc[winner][1] >= acc[winner][0] else "control"
                switches.append((t, start, winner, reason))
                possessor[start:t + 1] = winner
                if owner == NONE:
                    possessor[first:t + 1] = winner  # the half opens with this team
                owner, last_switch = winner, t
                s = [0.0, 0.0]
                acc = [[0.0, 0.0], [0.0, 0.0]]
            possessor[t] = owner
    return possessor, live, score, pts, switches


def main():
    p = PARAMS
    n = source_frames("tactical", GAME)
    teams, players, ball, ball_speed, (home, away) = tracking.load(GAME)
    visible = ~np.isnan(ball[:, 0])
    blocks = half_video_blocks(GAME)

    vel, speed, dv, angle = ball_kinematics(ball, ball_speed, p)
    dist, slot = team_distances(players, ball, teams)
    ctrl, slow = control(dist, speed, visible, p)
    touch_team, is_touch = touches(dv, dist, p)
    conf = look_ahead(ctrl, touch_team, p)
    possessor, live, score, pts, switches = decide(ctrl, touch_team, conf, visible, blocks, n, p)

    labels = np.full(n, "X", dtype="<U1")
    for _, first, last in blocks:
        seg = possessor[first:last + 1]
        labels[first:last + 1] = np.where(seg == HOME, "H", np.where(seg == AWAY, "A", "X"))
    write_submission(OUT_CSV, labels)

    sw = np.array([(a, b, c) for a, b, c, _ in switches], dtype=np.int64).reshape(-1, 3)
    reasons = np.array([r for *_, r in switches])
    np.savez_compressed(
        OUT_DIAG, possessor=possessor, live=live, score=score, ctrl=ctrl, slow=slow, speed=speed,
        dv=dv, angle=angle, vel=vel, dist=dist, slot=slot, touch_team=touch_team,
        is_touch=is_touch, conf=conf, pts_ctrl=pts["ctrl"], pts_touch=pts["touch"],
        pts_look=pts["look"], pts_own=pts["own"], switches=sw, reasons=reasons,
        threshold=p["THRESHOLD"])

    # Aggregate summary only: counts and distributions, nothing frame-level.
    graded = np.zeros(n, dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    print("teams: H = {}, A = {}".format(home, away))
    print("ball visible on {:.1f}% of graded frames".format(100 * visible[graded].mean()))
    v = dv[graded & ~np.isnan(dv)]
    print("velocity change |dv| m/s, graded frames: p50 {:.2f}  p90 {:.2f}  p99 {:.2f}".format(
        *np.percentile(v, [50, 90, 99])))
    print("touch events: {} detected, {} attributed (H {}, A {})".format(
        int(is_touch[graded].sum()), int((touch_team[graded] != NONE).sum()),
        int((touch_team[graded] == HOME).sum()), int((touch_team[graded] == AWAY).sum())))
    for half, first, last in blocks:
        mine = [s for s in switches if first <= s[0] <= last]
        seg = labels[first:last + 1]
        runs = 1 + int(np.sum(seg[1:] != seg[:-1]))
        print("half {}: {} possession changes ({} by touch, {} by control), {} possessions, "
              "mean {:.1f} s, H {:.1f}%".format(
                  half, len(mine), sum(r == "touch" for *_, r in mine),
                  sum(r == "control" for *_, r in mine), runs, len(seg) / runs / FPS,
                  100 * np.mean(seg == "H")))
    lags = sw[:, 0] - sw[:, 1]
    print("decision delay after the dated change: median {:.2f} s, p90 {:.2f} s".format(
        np.median(lags) / FPS, np.percentile(lags, 90) / FPS))
    print("-> {}".format(os.path.relpath(OUT_CSV, ROOT)))


if __name__ == "__main__":
    main()
