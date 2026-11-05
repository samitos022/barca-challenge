"""Possession v2: v1 (H <-> A) plus the dead ball (H/A -> D and back).

Possession changes work exactly as in v1 (possession_v1.py). On top of them a second score, the
dead-ball score, decides when play stops. Each half starts dead (until the kick-off).

Entering the dead ball (H/A -> D)
---------------------------------
  1. Ball out.  The ball is beyond a line by more than OUT_MARGIN for OUT_MIN_FRAMES frames: dead at
     once, dated to the frame it crossed the line.
  2. Players far.  Mean distance of the K players closest to the ball: at a stoppage they stand back
     (free kicks, throw-ins, goal kicks) while in play someone is always on the ball.
  3. Sudden change of rhythm.  Mean speed of every tracked player over the next second, divided by
     the mean over the RHYTHM_BEFORE_S seconds before: at a stoppage the players slow down
     abruptly. The absolute speed is not used to enter a dead ball, because a team keeping the
     ball in its own half can be just as slow while play goes on; what it does not do is brake.
     Measured on this match with classes chosen by the algorithm: players already slow in the
     previous 5 s brake (ratio under 0.7) on 0-2% of frames, against about a third when the ball
     has just gone out.
     2 and 3 average into a 0..1 "deadness". Above 0.5 the dead-ball score grows, below it shrinks;
     a fast ball or a touch in play resets it.
  4. Whistle.  A whistle-like tone in the audio adds points only if the players then brake (rule 3)
     in the next WHISTLE_LOOK_S seconds (whistle.py explains why the audio alone is not trusted).
  5. Corner kick.  An attacker standing at a corner flag (within R_CORNER, at least
     CORNER_STAND_S) and then leaving it is a corner kick, a restart: the dead ball is extended
     back over the time the ball was taken to the corner undisturbed (no opponent within PRESS_R,
     ball within CORNER_AREA of that corner), back to where the ball went out. The tracking almost
     never sees the ball at the flag, so the taker is what is detected; a player who is not
     exactly at the flag, or who defends that goal, does not count.
  6. Untracked ball.  A missing ball is never dead by position: it is dead by position only if it
     is out when last seen before the gap or first seen after it. Elsewhere a gap only holds the
     state, and the other rules decide alone: rule 3 (braking) counts in full.
  7. A reason to stop.  The referee stops play inside the pitch for a foul or an offside. A dead
     ball reached only through rules 2-4 is granted only if, in the CAUSE_S seconds before it
     started, there was a clear contest near the ball (a player of each team within DUEL_R of each
     other, both within DUEL_BALL_R of the ball) or a pass received in an offside position (the
     receiver, at the moment of the pass, in the opponents' half and beyond both the ball and the
     second-last opponent by OFFSIDE_MARGIN), or the ball, untracked, comes back into view out of
     the pitch at the end of the gap (it went out unseen). Otherwise it is refused: a player
     holding the ball slowly, unchallenged, is not a stoppage, whatever the crowd is whistling.
     Ball gaps here are long (99.7% of the missing frames are in gaps of 5 s or more), so the ball
     cannot be interpolated across them; where it reappears is still known.

A dead ball entered only on rules 2-4 (players or whistle) must last D_MIN_S in all and
D_MIN_AFTER_S after it was decided: if play resumes earlier it was not a stoppage, and it is cancelled (the team that had the ball keeps it). A ball
out and a corner kick are certain and are never cancelled.

Leaving the dead ball (D -> H/A)
--------------------------------
  Here the absolute speed of the players is used (deadness with rule 3 replaced by "players
  slow"): during a stoppage everybody walks, and the restart shows as players running again.
  - Kick-off: the first restart of each half only counts with the ball within KICKOFF_R of the
    centre spot (balls knocked about before the kick-off are not a restart).
  - Restart touch: the first touch of a player (velocity change, player within reach) with the ball
    on the pitch or just outside a line (throw-ins), at least MIN_DEAD_S after the stoppage began.
    The toucher's team gets the ball.
  - Fallback when the touch is not seen (ball missing at the restart, or a kick-off taken on the
    first graded frame): ALIVE_RATE points per second of play that looks alive (ball on the pitch,
    deadness under 0.5), threshold 1. The ball goes to the team with the player closest to it
    when play resumed (the taker), else to the team that controlled it since, else to the team
    that had it before the stoppage.

    .venv/bin/python solution/possession_v2.py                    # tactical tracking
    .venv/bin/python solution/possession_v2.py --tracking fused   # tactical + broadcast

Writes evaluation/results/v2_possession.csv and data/derived/v2_diagnostics.npz (v2f_* when fused).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
sys.path.insert(0, HERE)
from games import GAME, GAMES, half_video_blocks, source_frames  # noqa: E402
from baseline import write_submission  # noqa: E402
import possession_v1 as v1  # noqa: E402
import tracking  # noqa: E402
import whistle  # noqa: E402

FPS = 25
OUT_CSV = os.path.join(ROOT, "evaluation", "results", "v2_possession.csv")
OUT_DIAG = os.path.join(ROOT, "data", "derived", "v2_diagnostics.npz")


def outputs(source="tactical"):
    """(submission csv, diagnostics npz) for a tracking source: v2_* tactical, v2f_* fused."""
    if source == "tactical":
        return OUT_CSV, OUT_DIAG
    return (os.path.join(ROOT, "evaluation", "results", "v2f_possession.csv"),
            os.path.join(ROOT, "data", "derived", "v2f_diagnostics.npz"))

PARAMS = dict(v1.PARAMS)
PARAMS.update({
    # 1, ball out
    "OUT_MARGIN": 1.0,      # m beyond the line: tracking noise plus the ball's radius
    "OUT_MIN_FRAMES": 3,    # frames: one stray detection outside is not a ball out
    "NEAR_LINE": 3.0,       # m: a ball last seen this close to a line can have gone out in a gap
    # 2, players far
    "K_NEAR": 4,            # players closest to the ball
    "DIST_PLAY": 7.0,       # m, mean distance of the K: typical of play (deadness 0)
    "DIST_DEAD": 12.0,      # m, typical of a stoppage (deadness 1)
    # 3, players slow
    "SPEED_LAG": 5,         # frames each side to measure a player's speed
    "SPEED_MAX": 12.0,      # m/s, faster is a tracking jump, ignored
    "MIN_PLAYERS": 10,      # tracked players needed for a mean speed
    "SPD_PLAY": 2.0,        # m/s, mean player speed typical of play (exit deadness 0)
    "SPD_DEAD": 1.1,        # m/s, typical of a stoppage (exit deadness 1)
    "RHYTHM_BEFORE_S": 4.0,  # s of play the current pace is compared with (ending 1 s ago)
    "RHYTHM_AFTER_S": 1.0,  # s of pace measured from now on
    "R_PLAY": 0.9,          # pace ratio still normal in play: fast play stays above it 3 times in 4
    "R_DEAD": 0.5,          # pace ratio of a stoppage: the pace halves (deadness 1)
    "D_MIN_S": 3.0,         # s: a dead ball from rules 2-4 shorter than this is cancelled ...
    # 7, a reason to stop
    "CAUSE_S": 3.0,         # s before the stoppage in which a contest or an offside is looked for
    "DUEL_R": 1.2,          # m between two opponents: a contest (tracking puts bodies ~0.5-1 m apart)
    "DUEL_BALL_R": 5.0,     # m from the ball (last seen within BALL_HOLD_S) for both of them
    "BALL_HOLD_S": 3.0,     # s a ball position is carried over a gap, for rule 7 only: the ball is
                            # lost most often precisely in duels (DATA.md). Measured: a contest in
                            # the previous 3 s on 17-21% of in-play frames, so still selective
    "PASS_SPEED": 6.0,      # m/s, a touch that sends the ball this fast is a pass
    "RECEIVE_S": 4.0,       # s within which the passer's team must touch / control it again
    "OFFSIDE_MARGIN": 0.5,  # m beyond the second-last opponent: tracking noise
    "D_MIN_AFTER_S": 2.0,   # s: ... and so is one where play resumes this soon after the decision
                            # (back-dating alone must not make a stoppage look long enough)
    # dead-ball score
    "RATE_DEAD": 0.5,       # points per second at deadness 1: two seconds of a clear stoppage
    "GAP_WEIGHT": 1.0,      # weight of rule 3 alone while the ball is untracked (no penalty)
    "W_RHYTHM": 0.6,        # share of rule 3 (braking) in the deadness; rule 2 (distance) the rest
    "THRESHOLD_DEAD": 1.0,
    # 4, whistle
    "W_WHISTLE": 0.3,       # points if the players then brake completely (crowds whistle too)
    "WHISTLE_LOOK_S": 2.0,
    "WHISTLE_LEAD_S": 0.3,  # the foul comes just before the whistle
    # 5, corner kick
    "R_CORNER": 2.0,        # m from the corner flag: the arc is 1 m, plus tracking noise
    "CORNER_STAND_S": 2.0,  # s the taker stands at the flag before kicking
    "PRESS_R": 3.0,         # m: an opponent this close means the attacker is not undisturbed
    "CORNER_AREA": 25.0,    # m from the corner: farther than this the walk had not begun
    "CORNER_BACK_S": 40.0,  # s, longest walk looked back over
    # restart
    "RESTART_MARGIN": 2.0,  # m outside the line where a restart touch still counts (throw-in)
    "MIN_DEAD_S": 1.0,      # s, no restart is quicker than this
    "ALIVE_RATE": 1.0,      # points per second of clearly alive play while dead
    "KICKOFF_R": 4.0,       # m from the centre spot: a half starts with a kick-off from there
    "KICKOFF_MAX_S": 300.0,  # s: if no kick-off is seen by then, any restart counts
})

HOME, AWAY, NONE, DEAD = 0, 1, -1, 2
RESTART_LOOK = 12  # frames each side of the restart in which the taker is looked for (0.5 s)
P_KEYS = ("ctrl", "touch", "look", "own", "decay")   # possession score breakdown
D_KEYS = ("dist", "rhythm", "whistle")               # dead-ball score breakdown
A_KEYS = ("alive", "fast", "decay")                  # restart score breakdown
RESET_FAST, RESET_TOUCH, RESET_NO_CAUSE, RESET_OUT, RESET_KICKOFF = 1, 2, 3, 4, 5


def ball_out_distance(ball, pitch):
    """How far beyond the nearest line the ball is, in metres (0 inside, NaN if untracked)."""
    length, width = pitch
    x, y = ball[:, 0].astype(np.float64), ball[:, 1].astype(np.float64)
    out = np.maximum.reduce([-x, x - length, -y, y - width, np.zeros_like(x)])
    line = np.minimum.reduce([x, length - x, y, width - y])  # distance to nearest line, inside
    return out, line


def runs_at_least(mask, k):
    """mask, keeping only runs of at least k consecutive True."""
    out = np.zeros_like(mask)
    idx = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    for s, e in zip(idx[::2], idx[1::2]):
        if e - s >= k:
            out[s:e] = True
    return out


def players_far(players, ball, k):
    d = np.linalg.norm(players - ball[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    part = np.partition(d, k - 1, axis=1)[:, :k]
    m = part.mean(axis=1)
    m[~np.isfinite(m)] = np.nan
    return m.astype(np.float32)


def players_speed(players, p):
    k = p["SPEED_LAG"]
    a = np.full_like(players, np.nan)
    b = np.full_like(players, np.nan)
    a[k:-k], b[k:-k] = players[2 * k:], players[:-2 * k]
    s = np.linalg.norm(a - b, axis=2) * FPS / (2 * k)
    s[s > p["SPEED_MAX"]] = np.nan
    cnt = np.sum(~np.isnan(s), axis=1)
    with np.errstate(invalid="ignore"):
        m = np.nansum(s, axis=1) / np.maximum(cnt, 1)
    m[cnt < p["MIN_PLAYERS"]] = np.nan
    return m.astype(np.float32)


def attack_sides(players, teams, blocks):
    """Per half, the team defending the x = 0 goal: the one whose players stand lower on average."""
    sides = {}
    for half, first, last in blocks:
        mx = [np.nanmean(players[first:last + 1, teams == team, 0]) for team in (HOME, AWAY)]
        sides[half] = HOME if mx[0] <= mx[1] else AWAY
    return sides


def pace_ratio(pspeed, p):
    """Mean player speed over the next RHYTHM_AFTER_S / mean over the RHYTHM_BEFORE_S before the
    last second. Under 1 the players are braking. NaN where either window has no speed."""
    n = len(pspeed)
    x = np.nan_to_num(pspeed.astype(np.float64), nan=0.0)
    c = np.concatenate(([0.0], np.cumsum(x)))
    k = np.concatenate(([0], np.cumsum(~np.isnan(pspeed))))
    t = np.arange(n)

    def mean(lo, hi):
        lo, hi = np.clip(lo, 0, n), np.clip(hi, 0, n)
        cnt = k[hi] - k[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(cnt > 0, (c[hi] - c[lo]) / cnt, np.nan)

    before = mean(t - int((p["RHYTHM_BEFORE_S"] + 1) * FPS), t - FPS)
    after = mean(t, t + int(p["RHYTHM_AFTER_S"] * FPS))
    return (after / np.maximum(before, 0.3)).astype(np.float32)


def carried_ball(ball, p):
    """Ball position, carried forward over gaps up to BALL_HOLD_S (rule 7 only)."""
    n = len(ball)
    idx = np.where(~np.isnan(ball[:, 0]), np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    ok = (last >= 0) & (np.arange(n) - last <= int(p["BALL_HOLD_S"] * FPS))
    out = np.full_like(ball, np.nan)
    out[ok] = ball[last[ok]]
    return out


def contests(players, teams, ball, p):
    """Rule 7a: per frame, True when a player of each team are within DUEL_R of each other and both
    within DUEL_BALL_R of the ball."""
    n = len(players)
    b = carried_ball(ball, p)
    home, away = np.flatnonzero(teams == HOME), np.flatnonzero(teams == AWAY)
    out = np.zeros(n, dtype=bool)
    for a in range(0, n, 20000):
        z = slice(a, min(n, a + 20000))
        ph, pa, bb = players[z][:, home], players[z][:, away], b[z]
        near_h = np.linalg.norm(ph - bb[:, None], axis=2) <= p["DUEL_BALL_R"]
        near_a = np.linalg.norm(pa - bb[:, None], axis=2) <= p["DUEL_BALL_R"]
        d = np.linalg.norm(ph[:, :, None] - pa[:, None], axis=3)
        close = (d <= p["DUEL_R"]) & near_h[:, :, None] & near_a[:, None, :]
        out[z] = close.any(axis=(1, 2))
    return out


def offsides(players, teams, ball, speed, touch_team, ctrl, slot, blocks, pitch, p):
    """Rule 7b: [(receive frame, pass frame, team)] for passes received in an offside position."""
    length = pitch[0]
    sides = attack_sides(players, teams, blocks)
    recv = int(p["RECEIVE_S"] * FPS)
    found = []
    for half, first, last in blocks:
        touches = [t for t in np.flatnonzero(touch_team[first:last + 1] >= 0) + first]
        for t in touches:
            team = int(touch_team[t])
            after = speed[t + 2:t + 6]
            after = after[np.isfinite(after)]
            if not len(after) or after.max() < p["PASS_SPEED"]:
                continue
            # the receiver: the passer's team next touches or controls the ball
            r = None
            for u in range(t + 6, min(last, t + recv) + 1):
                if touch_team[u] >= 0 or ctrl[:, u].max() > 0.5:
                    who = int(touch_team[u]) if touch_team[u] >= 0 else int(np.argmax(ctrl[:, u]))
                    r = u if who == team else None
                    break
            if r is None or slot[team, r] < 0:
                continue
            rx = players[t, slot[team, r], 0]  # the receiver where he stood at the pass
            if np.isnan(rx) or np.isnan(ball[t, 0]):
                continue
            opp = players[t, teams == 1 - team, 0]
            opp = opp[~np.isnan(opp)]
            if len(opp) < 2:
                continue
            forward = sides[half] == team  # this team defends x = 0, so it attacks towards +x
            sign = 1.0 if forward else -1.0
            second_last = np.sort(sign * opp)[-2]
            line = max(sign * ball[t, 0], second_last, sign * length / 2 if forward else -length / 2)
            if sign * rx > line + p["OFFSIDE_MARGIN"]:
                found.append((int(r), int(t), team))
    return found


def corner_kicks(players, teams, ball, dist, out, blocks, pitch, p):
    """Rule 5: {kick frame: (first dead frame, kicker team)}.

    The tracking almost never sees the ball at a corner flag (the tactical camera's edge), so the
    kick is found from the taker: an attacker standing within R_CORNER of the flag for at least
    CORNER_STAND_S, the kick being the moment he leaves it. A defender at his own corner flag, or
    a player who only runs past it, is not a corner kick.
    """
    length, width = pitch
    corners = np.array([(0, 0), (0, width), (length, 0), (length, width)], dtype=np.float32)
    n = len(players)
    d = np.linalg.norm(players[:, :, None, :] - corners[None, None], axis=3)
    flat = np.where(np.isnan(d), np.inf, d).reshape(n, -1)
    j = np.argmin(flat, axis=1)
    dmin = flat[np.arange(n), j]
    slot_at, corner_at = j // 4, j % 4
    sides = attack_sides(players, teams, blocks)
    stand = int(p["CORNER_STAND_S"] * FPS)
    back = int(p["CORNER_BACK_S"] * FPS)
    found = {}
    for half, first, last in blocks:
        near = np.zeros(n, dtype=bool)
        near[first:last + 1] = dmin[first:last + 1] <= p["R_CORNER"]
        idx = np.flatnonzero(np.diff(np.concatenate(([0], near.astype(np.int8), [0]))))
        runs = []
        for a, b in zip(idx[::2], idx[1::2]):
            if runs and a - runs[-1][1] < FPS:  # the taker flickers in and out of the circle
                runs[-1][1] = b
            else:
                runs.append([a, b])
        for a, b in runs:
            if b - a < stand:
                continue
            seg = slice(a, b)
            c = int(np.bincount(corner_at[seg][near[seg]], minlength=4).argmax())
            team = int(np.bincount(teams[slot_at[seg][near[seg]]], minlength=2).argmax())
            attacker = (1 - sides[half]) if corners[c, 0] == 0 else sides[half]
            if team != attacker:
                continue
            kick = b - 1
            start = a
            for u in range(a - 1, max(first, a - back) - 1, -1):
                if np.isnan(ball[u, 0]):
                    continue
                if out[u]:
                    start = u
                    break
                opp = dist[1 - team, u]
                if (np.isfinite(opp) and opp < p["PRESS_R"]) or \
                        np.linalg.norm(ball[u] - corners[c]) > p["CORNER_AREA"]:
                    start = min(a, u + 1)
                    break
            found[int(kick)] = (int(start), team)
    return found


def decide(f, blocks, n, p):
    """The three-state machine. f: dict of per-frame features. Returns labels and diagnostics."""
    decay = 0.5 ** (1.0 / (p["HALF_LIFE"] * FPS))
    rate_ctrl, rate_own = p["RATE_CTRL"] / FPS, p["RATE_OWN"] / FPS
    w_touch, w_own, w_look, thr = p["W_TOUCH"], p["W_TOUCH_OWN"], p["W_LOOK"], p["THRESHOLD"]
    rate_dead, thr_dead = p["RATE_DEAD"] / FPS, p["THRESHOLD_DEAD"]
    alive_rate = p["ALIVE_RATE"] / FPS
    episode_gap = int(round(p["EPISODE_GAP"] * FPS))
    min_dead = int(p["MIN_DEAD_S"] * FPS)
    lead = int(round(p["WHISTLE_LEAD_S"] * FPS))

    ctrl = [f["ctrl"][0].tolist(), f["ctrl"][1].tolist()]
    conf = [f["conf"][0].tolist(), f["conf"][1].tolist()]
    touch = f["touch_team"].tolist()
    f_dist_l = f["f_dist"].tolist()
    f_rhythm_l = f["f_rhythm"].tolist()
    exit_ness = f["deadness_exit"].tolist()
    d_min = int(p["D_MIN_S"] * FPS)
    d_min_after = int(p["D_MIN_AFTER_S"] * FPS)
    w_rhythm = p["W_RHYTHM"]
    visible = f["visible"].tolist()
    fast = f["fast"].tolist()
    in_zone = f["restart_zone"].tolist()
    at_centre = f["at_centre"].tolist()
    kickoff_max = int(p["KICKOFF_MAX_S"] * FPS)
    on_pitch = f["on_pitch"].tolist()
    out_start = f["out_start"]
    whistle_pts = f["whistle_pts"].tolist()
    cause_cum = f["cause_cum"]
    cause_back = int(p["CAUSE_S"] * FPS)
    refused = []
    corners = f["corners"]

    near = f["dist"]

    def restart_team(onset, t, before):
        """Who restarted play when the restart touch was not seen: the team closest to the ball
        when play resumed (the taker), then the team that controlled it since, then the team that
        had it before the stoppage."""
        lo, hi = max(0, onset - RESTART_LOOK), min(n, onset + RESTART_LOOK + 1)
        d = np.nan_to_num(near[:, lo:hi], nan=np.inf).min(axis=1)
        if np.isfinite(d).any():
            return HOME if d[HOME] <= d[AWAY] else AWAY
        w = [sum(ctrl[c][onset:t + 1]) for c in (HOME, AWAY)]
        if max(w) > 0:
            return HOME if w[0] >= w[1] else AWAY
        return before if before in (HOME, AWAY) else HOME

    state = np.full(n, NONE, dtype=np.int8)
    live = np.full(n, NONE, dtype=np.int8)
    score = np.zeros((2, n), dtype=np.float32)
    dead_score = np.zeros(n, dtype=np.float32)
    alive_score = np.zeros(n, dtype=np.float32)
    changes = []  # (decided frame, dated frame, new state, reason)
    # Per-parameter breakdown of every score, for the review video: at each frame, the points each
    # parameter has added (or removed) since the score last started from zero. The totals of one
    # score always add up to that score, decay included.
    ptot = {k: np.zeros((2, n), dtype=np.float32) for k in P_KEYS}
    dtot = {k: np.zeros(n, dtype=np.float32) for k in D_KEYS}
    atot = {k: np.zeros(n, dtype=np.float32) for k in A_KEYS}
    d_reset = np.zeros(n, dtype=np.int8)  # why the dead-ball score went to 0 (RESET_* codes)
    a_reset = np.zeros(n, dtype=np.int8)

    for _, first, last in blocks:
        owner = DEAD
        before_dead = NONE
        dead_since = first
        dead_reason = "half start"
        s = [0.0, 0.0]
        onset = [first, first]
        last_gain = [first - 10 ** 6, first - 10 ** 6]
        acc = [[0.0, 0.0], [0.0, 0.0]]
        sd, sd_onset, sd_last, sd_whistle = 0.0, first, first - 10 ** 6, False
        sa, sa_onset, sa_last = 0.0, first, first - 10 ** 6
        pt = [[0.0] * len(P_KEYS), [0.0] * len(P_KEYS)]
        dt = [0.0] * len(D_KEYS)
        at = [0.0] * len(A_KEYS)
        last_change = first
        kickoff = True  # the first restart of a half is the kick-off, from the centre spot

        def change(t, start, new, reason):
            nonlocal owner, last_change, s, acc, sd, sa, dead_since, before_dead, dead_reason, pt, dt, at
            if reason != "dead ball too short":  # a cancellation rewrites the whole stoppage
                start = max(start, last_change + 1 if t > first else first)
            start = min(start, t)
            state[start:t + 1] = new
            changes.append((t, start, new, reason))
            if new == DEAD:
                before_dead = owner
                dead_since = start
                dead_reason = reason
            owner, last_change = new, t
            s, acc = [0.0, 0.0], [[0.0, 0.0], [0.0, 0.0]]
            sd = sa = 0.0
            pt = [[0.0] * len(P_KEYS), [0.0] * len(P_KEYS)]
            dt, at = [0.0] * len(D_KEYS), [0.0] * len(A_KEYS)

        for t in range(first, last + 1):
            live[t] = owner
            # ---- corner kick (rule 5): a restart, whatever the machine thought -------------
            if t in corners and t - last_change >= min_dead:
                start, kicker = corners[t]
                if owner != DEAD:
                    change(t - 1, start, DEAD, "corner")
                change(t, t, kicker, "corner kick")
                state[t] = owner
                continue
            if owner == DEAD:
                # ---- leaving the dead ball ----------------------------------------------
                if kickoff and t - first > kickoff_max:
                    kickoff = False
                centre_ok = at_centre[t] or not kickoff
                soft = dead_reason in ("whistle", "players stopped")
                too_short = soft and before_dead in (HOME, AWAY) and \
                    (t - dead_since < d_min or t - last_change < d_min_after)
                if touch[t] >= 0 and in_zone[t] and centre_ok and (t - dead_since >= min_dead or soft):
                    kickoff = False
                    if too_short:
                        change(t, dead_since, before_dead, "dead ball too short")
                    else:
                        change(t, t, touch[t], "restart touch")
                    state[t] = owner
                    continue
                if visible[t] and on_pitch[t]:
                    # a fast ball is alive play only if the players are moving too: a ball kicked
                    # back to the centre while everyone stands still is not a restart
                    alive = max(0.0, 1.0 - 2.0 * exit_ness[t])
                    ga = alive_rate * alive
                    gf = alive_rate if fast[t] and alive > 0.0 else 0.0
                    g = ga + gf
                    if g > 0.0 and (sa < 0.02 or t - sa_last > episode_gap):
                        sa_onset = t
                    if g > 0.0:
                        sa_last = t
                    dl = sa * (1.0 - decay)
                    sa = sa - dl + g
                    at = [at[0] + ga, at[1] + gf, at[2] - dl]
                elif visible[t]:
                    sa = 0.0  # ball out: nothing is alive
                    at = [0.0] * len(A_KEYS)
                    a_reset[t] = RESET_OUT
                else:
                    dl = sa * (1.0 - decay)
                    sa -= dl
                    at[2] -= dl
                if sa >= 1.0 and kickoff and not any(
                        at_centre[max(first, sa_onset - FPS):sa_onset + FPS // 2 + 1]):
                    sa = 0.0  # before the kick-off, play can only resume from the centre spot
                    at = [0.0] * len(A_KEYS)
                    a_reset[t] = RESET_KICKOFF
                alive_score[t] = sa
                for k, v in zip(A_KEYS, at):
                    atot[k][t] = v
                if sa >= 1.0 and too_short:
                    change(t, dead_since, before_dead, "dead ball too short")
                elif sa >= 1.0 and t - dead_since >= min_dead:
                    kickoff = False
                    change(t, sa_onset, restart_team(sa_onset, t, before_dead), "alive play")
                state[t] = owner
                continue

            # ---- entering the dead ball ---------------------------------------------------
            if out_start[t] >= 0:
                change(t, int(out_start[t]), DEAD, "ball out")
                state[t] = owner
                continue
            if fast[t] or (touch[t] >= 0 and on_pitch[t]):
                sd = 0.0
                dt = [0.0] * len(D_KEYS)
                d_reset[t] = RESET_FAST if fast[t] else RESET_TOUCH
            else:
                # deadness = mean of rule 2 (distance) and rule 3 (braking); each parameter gets
                # its share of the points, all of them when the other cannot be measured
                rw = rate_dead * (1.0 if visible[t] else p["GAP_WEIGHT"])
                fd, fr = f_dist_l[t], f_rhythm_l[t]
                okd, okr = visible[t] and fd == fd, fr == fr
                if okd and okr:  # deadness = (1 - W_RHYTHM) * fd + W_RHYTHM * fr
                    gd = 2.0 * (1.0 - w_rhythm) * rw * (fd - 0.5)
                    gr = 2.0 * w_rhythm * rw * (fr - 0.5)
                elif okd:
                    gd, gr = rw * (2.0 * fd - 1.0), 0.0
                elif okr:
                    gd, gr = 0.0, rw * (2.0 * fr - 1.0)
                else:
                    gd = gr = 0.0
                g = gd + gr
                gw = whistle_pts[t]
                if g + gw > 0.0 and (sd < 0.02 or t - sd_last > episode_gap):
                    sd_onset, sd_whistle = t, False
                if gw > 0.0:
                    sd_onset, sd_whistle = min(sd_onset, t - lead), True
                if g + gw > 0.0:
                    sd_last = t
                # no decay here: the gain is already negative below deadness 0.5, and a decay would
                # cap a steady stoppage below the threshold (at deadness 1 it settled at 0.73)
                sd = sd + g + gw
                if sd <= 0.0:
                    sd, dt = 0.0, [0.0] * len(D_KEYS)
                else:
                    dt = [dt[0] + gd, dt[1] + gr, dt[2] + gw]
            dead_score[t] = sd
            for k, v in zip(D_KEYS, dt):
                dtot[k][t] = v
            if sd >= thr_dead:
                lo = max(first, sd_onset - cause_back)
                if cause_cum[t + 1] - cause_cum[lo] == 0:
                    # rule 7: no contest and no offside before it, so not a stoppage
                    refused.append((t, sd_onset))
                    sd = 0.0
                    dt = [0.0] * len(D_KEYS)
                    d_reset[t] = RESET_NO_CAUSE
                    dead_score[t] = sd
                    for k in D_KEYS:
                        dtot[k][t] = 0.0
                    state[t] = owner
                    continue
                change(t, sd_onset, DEAD, "whistle" if sd_whistle else "players stopped")
                state[t] = owner
                continue

            # ---- possession change, as in v1 ----------------------------------------------
            for c in (HOME, AWAY):
                if c == owner:
                    s[c] = 0.0
                    pt[c] = [0.0] * len(P_KEYS)
                    continue
                g_ctrl = rate_ctrl * ctrl[c][t]
                g_touch = g_look = loss = 0.0
                if touch[t] == c:
                    g_touch = w_touch
                    g_look = w_look * conf[c][t]
                loss = rate_own * ctrl[owner][t]
                if touch[t] == owner:
                    loss += w_own
                dl = s[c] * (1.0 - decay)
                new = s[c] - dl + g_ctrl + g_touch + g_look - loss
                if new <= 0.0:
                    new = 0.0
                    acc[c] = [0.0, 0.0]
                    pt[c] = [0.0] * len(P_KEYS)
                else:
                    acc[c] = [acc[c][0] * decay + g_ctrl, acc[c][1] * decay + g_touch + g_look]
                    q = pt[c]
                    pt[c] = [q[0] + g_ctrl, q[1] + g_touch, q[2] + g_look, q[3] - loss, q[4] - dl]
                for k, v in zip(P_KEYS, pt[c]):
                    ptot[k][c, t] = v
                gain = g_ctrl + g_touch + g_look
                if gain > 0.0:
                    if s[c] < 0.02 or t - last_gain[c] > episode_gap:
                        onset[c] = t
                    last_gain[c] = t
                s[c] = new
                score[c, t] = new
            winner = HOME if s[HOME] >= thr else (AWAY if s[AWAY] >= thr else NONE)
            if winner != NONE:
                start = onset[winner]
                prev = f["prev_seen"][start]
                if start - prev > p["GAP_MIN"] and prev >= first:
                    start = (prev + start) // 2
                c_acc = acc[winner]
                change(t, start, winner, "touch" if c_acc[1] >= c_acc[0] else "control")
            state[t] = owner
    breakdown = {"ptot": ptot, "dtot": dtot, "atot": atot, "d_reset": d_reset, "a_reset": a_reset}
    return state, live, score, dead_score, alive_score, changes, refused, breakdown


def main(source="tactical"):
    p = PARAMS
    out_csv, out_diag = outputs(source)
    n = source_frames("tactical", GAME)
    pitch = GAMES[GAME]["pitch"]
    teams, players, ball, ball_speed, (home, away) = tracking.load(GAME, source)
    visible = ~np.isnan(ball[:, 0])
    blocks = half_video_blocks(GAME)

    # v1 features
    vel, speed, dv, angle = v1.ball_kinematics(ball, ball_speed, p)
    dist, slot = v1.team_distances(players, ball, teams)
    ctrl, slow = v1.control(dist, speed, visible, p)
    touch_team, is_touch = v1.touches(dv, dist, p)
    conf = v1.look_ahead(ctrl, touch_team, p)

    # rule 1 and 6: ball out, with gaps handled
    out_dist, line_dist = ball_out_distance(ball, pitch)
    out = runs_at_least(np.nan_to_num(out_dist, nan=0.0) > p["OUT_MARGIN"], p["OUT_MIN_FRAMES"])
    idx = np.where(visible, np.arange(n), -1)
    prev_seen = np.concatenate(([-1], np.maximum.accumulate(idx)[:-1]))
    out_start = np.full(n, -1, dtype=np.int64)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], out.astype(np.int8)))) == 1)
    for e in edges:
        k = e + p["OUT_MIN_FRAMES"] - 1  # the frame the run is confirmed
        s0 = e
        while s0 - 1 >= 0 and visible[s0 - 1] and out_dist[s0 - 1] > 0:
            s0 -= 1  # back to the frame the ball crossed the line
        prev = prev_seen[s0]
        if prev >= 0 and s0 - prev > p["GAP_MIN"]:
            # it went out inside a gap: from the last sighting if that was near a line
            s0 = prev + 1 if line_dist[prev] <= p["NEAR_LINE"] else (prev + s0) // 2
        out_start[k] = s0
    on_pitch = visible & (np.nan_to_num(out_dist, nan=np.inf) == 0)
    restart_zone = visible & (np.nan_to_num(out_dist, nan=np.inf) <= p["RESTART_MARGIN"])
    centre = np.hypot(ball[:, 0] - pitch[0] / 2, ball[:, 1] - pitch[1] / 2)
    at_centre = np.nan_to_num(centre, nan=np.inf) <= p["KICKOFF_R"]

    # rules 2 and 3
    far = players_far(players, ball, p["K_NEAR"])
    pspeed = players_speed(players, p)
    f_dist = np.clip((far - p["DIST_PLAY"]) / (p["DIST_DEAD"] - p["DIST_PLAY"]), 0, 1)
    f_speed = np.clip((p["SPD_PLAY"] - pspeed) / (p["SPD_PLAY"] - p["SPD_DEAD"]), 0, 1)
    rhythm = pace_ratio(pspeed, p)
    f_rhythm = np.clip((p["R_PLAY"] - rhythm) / (p["R_PLAY"] - p["R_DEAD"]), 0, 1)

    def combine(f_play, w_play):
        both = np.where(np.isnan(f_dist), f_play,
                        np.where(np.isnan(f_play), f_dist, (1 - w_play) * f_dist + w_play * f_play))
        return np.nan_to_num(np.where(visible, both, f_play), nan=0.5).astype(np.float32)

    deadness = combine(f_rhythm, p["W_RHYTHM"])  # entering: distance + braking
    deadness_exit = combine(f_speed, 0.5)        # leaving: distance + absolute pace
    fast = visible & np.nan_to_num(speed > p["V_FAST"], nan=False)

    # rule 4: whistle, confirmed by the players stopping afterwards
    events, wlevel = whistle.load(GAME)
    look = int(p["WHISTLE_LOOK_S"] * FPS)
    fs = np.nan_to_num(f_rhythm, nan=0.0)
    cs = np.concatenate(([0.0], np.cumsum(fs)))
    whistle_pts = np.zeros(n, dtype=np.float32)
    whistle_conf = np.zeros(n, dtype=np.float32)
    for s0, e0, *_ in events:
        c = (cs[min(n, s0 + 1 + look)] - cs[min(n, s0 + 1)]) / look  # mean braking afterwards
        whistle_conf[s0:e0 + 1] = c
        whistle_pts[s0] = p["W_WHISTLE"] * c

    # rule 5
    corners = corner_kicks(players, teams, ball, dist, out, blocks, pitch, p)

    # rule 7
    duel = contests(players, teams, ball, p)
    offs = offsides(players, teams, ball, speed, touch_team, ctrl, slot, blocks, pitch, p)
    offside_at = np.zeros(n, dtype=bool)
    for r, _, _ in offs:
        offside_at[r] = True
    # ...or a gap that ends with the ball out of the pitch: it went out unseen
    next_seen = np.full(n, -1, dtype=np.int64)
    nxt = -1
    for t in range(n - 1, -1, -1):
        if visible[t]:
            nxt = t
        next_seen[t] = nxt
    gap_out = ~visible & (next_seen >= 0) & \
        (np.nan_to_num(out_dist[np.maximum(next_seen, 0)], nan=0.0) > p["OUT_MARGIN"])
    cause_cum = np.concatenate(([0], np.cumsum(duel | offside_at | gap_out)))

    feats = {"ctrl": ctrl, "conf": conf, "touch_team": touch_team, "deadness": deadness,
             "deadness_exit": deadness_exit, "f_dist": f_dist, "f_rhythm": f_rhythm,
             "visible": visible, "fast": fast, "restart_zone": restart_zone, "on_pitch": on_pitch,
             "out_start": out_start, "whistle_pts": whistle_pts, "corners": corners,
             "prev_seen": prev_seen, "dist": dist, "at_centre": at_centre, "cause_cum": cause_cum}
    state, live, score, dead_score, alive_score, changes, refused, bd = decide(feats, blocks, n, p)

    labels = np.full(n, "X", dtype="<U1")
    for _, first, last in blocks:
        seg = state[first:last + 1]
        labels[first:last + 1] = np.select([seg == HOME, seg == AWAY, seg == DEAD], ["H", "A", "D"], "X")
    write_submission(out_csv, labels)

    ch = np.array([(a, b, c) for a, b, c, _ in changes], dtype=np.int64).reshape(-1, 3)
    reasons = np.array([r for *_, r in changes])
    corner_arr = np.array([(t, s0, k) for t, (s0, k) in sorted(corners.items())], dtype=np.int64).reshape(-1, 3)
    np.savez_compressed(
        out_diag, state=state, live=live, score=score, dead_score=dead_score, alive_score=alive_score,
        ctrl=ctrl, slow=slow, speed=speed, dv=dv, angle=angle, vel=vel, dist=dist, slot=slot,
        touch_team=touch_team, is_touch=is_touch, conf=conf, out_dist=out_dist, far=far,
        pspeed=pspeed, f_dist=f_dist, f_speed=f_speed, deadness=deadness, rhythm=rhythm,
        f_rhythm=f_rhythm, deadness_exit=deadness_exit, whistle_level=wlevel,
        whistle_conf=whistle_conf, changes=ch, reasons=reasons, corners=corner_arr,
        threshold=p["THRESHOLD"], threshold_dead=p["THRESHOLD_DEAD"], duel=duel, gap_out=gap_out,
        offsides=np.array(offs, dtype=np.int64).reshape(-1, 3),
        refused=np.array(refused, dtype=np.int64).reshape(-1, 2),
        d_reset=bd["d_reset"], a_reset=bd["a_reset"],
        **{"ptot_" + k: v for k, v in bd["ptot"].items()},
        **{"dtot_" + k: v for k, v in bd["dtot"].items()},
        **{"atot_" + k: v for k, v in bd["atot"].items()})

    # aggregate summary
    graded = np.zeros(n, dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    print("teams: H = {}, A = {}".format(home, away))
    print("whistle-like events {}, confirmed by players stopping {}".format(
        len(events), int(np.sum(whistle_pts > 0))))
    print("corner kicks found: {}".format(len(corners)))
    graded_duel = duel[graded].mean() if graded.any() else 0
    print("contest near the ball on {:.1f}% of graded frames; passes received offside: {}".format(
        100 * graded_duel, len(offs)))
    print("dead balls refused for lack of a contest or offside: {}".format(len(refused)))
    from collections import Counter
    for half, first, last in blocks:
        mine = [c for c in changes if first <= c[0] <= last]
        seg = labels[first:last + 1]
        runs, cur = [], None
        for lab in seg:
            if lab != cur:
                runs.append([lab, 0])
                cur = lab
            runs[-1][1] += 1
        poss = [r for r in runs if r[0] in "HA"]
        dead = [r for r in runs if r[0] == "D"]
        print("half {}: H {:.1f}%  A {:.1f}%  D {:.1f}%   {} possessions (mean {:.1f} s), "
              "{} dead balls (mean {:.1f} s)".format(
                  half, 100 * np.mean(seg == "H"), 100 * np.mean(seg == "A"), 100 * np.mean(seg == "D"),
                  len(poss), np.mean([r[1] for r in poss]) / FPS, len(dead),
                  np.mean([r[1] for r in dead]) / FPS))
        print("   reasons: " + ", ".join("{} {}".format(k, v) for k, v in
                                         Counter(r for *_, r in mine).most_common()))
    print("-> {}".format(os.path.relpath(out_csv, ROOT)))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracking", choices=("tactical", "fused"), default="tactical",
                    help="fused = tactical + broadcast (run sync_broadcast.py and fusion.py first)")
    main(ap.parse_args().tracking)
