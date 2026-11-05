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
     the pitch at the end of the gap (it went out unseen). Without a cause the dead ball is not
     refused, it costs NO_CAUSE_PENALTY points more: a player holding the ball slowly,
     unchallenged, rarely gets that far, while a stoppage whose cause the tracking missed (the
     contest happened in a crowd, the ball went out unseen) still gets there if every other
     parameter says so.
     Ball gaps here are long (99.7% of the missing frames are in gaps of 5 s or more), so the ball
     cannot be interpolated across them; where it reappears is still known.

  8. Long gap, players walking, SUPPORTED.  The ball untracked for GAP_DEAD_S or more while the
     players' mean speed over the gap is under (SPD_PLAY + SPD_DEAD) / 2 (no new threshold). A gap
     alone is weak evidence (the ball is also lost in crowds), so it makes a dead ball only with at
     least one other signal: the ball last seen near or beyond a line, or seen again beyond a line
     or at a set-piece spot (goal area, centre spot); the players braking or the players far from
     the ball when it was lost; a whistle. Then dead from the first missing frame; play resumes when the ball is seen again on the pitch (or just outside a
     line), dated to that frame, and goes to the team closest to it. Measured, without labels:
     BAR-ATH's long gaps cover 41% of the graded frames with the players at 1.0 m/s against 1.9
     m/s while the ball is seen, and the ball reappears at a set-piece spot (touchline, goal area,
     centre, goal line) 39% of the time against 13% for a random frame. On the external games a
     long gap is a dead ball 93-98% of the time, and its ends match the stoppage's to the frame
     (stronger than BAR-ATH: the speed condition is there not to trust the gap alone).

  9. Restart dated to the ball's first movement.  However a restart is confirmed (restart touch,
     alive-play score, ball back after a gap), it is dated to the moment the ball, after being slow
     (under V_MOVE for STILL_S), moves off: the slowest frame of the last second before it
     crosses V_MOVE, if that happened within MOVE_BACK_S before the confirmation. A kick-off at the very first frame of a half is dated
     there too.
  10. Lightning touch.  A touch by the team without the ball that sends it at PASS_SPEED or more
     in a direction LIGHT_ANGLE or more away from where it was going is an interception. When that
     team then takes possession (score over the threshold, within LIGHT_BACK_S, with no touch or
     control of the old owner in between) the possession is dated to that touch.

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
    .venv/bin/python solution/possession_v2.py --no-whistle       # without rule 4 (no video needed)

Writes evaluation/results/v2_possession.csv and data/derived/v2_diagnostics.npz (v2f_* when fused,
*_nowhistle without the whistle). predict() is the whole method on bare arrays, with no file access:
calibration runs it on external games.
"""

import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
sys.path.insert(0, HERE)
from games import GAME, GAMES, half_video_blocks, source_frames  # noqa: E402
from baseline import write_submission  # noqa: E402
import clean_tracking  # noqa: E402
import possession_v1 as v1  # noqa: E402
import tracking  # noqa: E402
import whistle  # noqa: E402

FPS = 25
def outputs(source="tactical", use_whistle=True):
    """(submission csv, diagnostics npz): v2_* tactical, v2f_* fused, *_nowhistle without rule 4."""
    stem = "v2" if source == "tactical" else "v2f"
    tail = "" if use_whistle else "_nowhistle"
    return (os.path.join(ROOT, "evaluation", "results", "{}_possession{}.csv".format(stem, tail)),
            os.path.join(ROOT, "data", "derived", "{}_diagnostics{}.npz".format(stem, tail)))


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
    "NO_CAUSE_PENALTY": 0.25,  # points: without a cause the dead-ball threshold is 1.25, not 1
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
    # 8, long ball gap with the players walking
    "GAP_DEAD_S": 5.0,      # s: shorter gaps are occlusions (99.7% of BAR-ATH's missing frames
                            # are in gaps of 5 s or more, DATA.md)
    "GAP_RULE": 1,          # 0 switches rule 8 off (ablation)
    "GAP_SUPPORT": 1,       # 1: a long gap is a dead ball only with another signal (see rule 8)
    "GAP_SUPPORT_S": 2.0,   # s around the start of the gap where braking / whistle are looked for
    # 9, restart dated to the ball's first movement
    "RESTART_MOVE": 1,      # 0 switches it off (ablation)
    "STILL_S": 0.5,         # s the ball must have been slow (under V_MOVE) before it moves off
    "V_MOVE": 3.0,          # m/s: the restart sends the ball faster than a player walks with it.
                            # A ball "lying still" is not still in this tracking: while it is placed
                            # or held it reads 1-2.5 m/s, so slow means under V_MOVE
    "MOVE_BACK_S": 3.0,     # s: the movement must be at most this long before the confirmation
    # 10, lightning touch: a possession starts at the interception that began it
    "LIGHTNING": 1,         # 0 switches it off (ablation)
    "LIGHT_ANGLE": 60.0,    # deg: the ball's direction changes completely
    "LIGHT_BACK_S": 4.0,    # s: the touch must be at most this long before the change is decided
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
    "CORNER_RULE": 0,       # off since rule 8: on BAR-ATH 13 of the 14 corners found fall inside a
                            # long gap rule 8 already makes dead, and 10 of them restarted play
                            # 1.5-38.6 s before the ball came back; on the external games
                            # removing it gains +0.012 spellF1 on 6 runs of 6. Still computed
                            # for the review video.
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
    gap_start = f["gap_dead_start"].tolist()
    gap_end = f["gap_dead_end"].tolist()
    move_onset = f["move_onset"]
    move_back = int(p["MOVE_BACK_S"] * FPS)
    lightning = f["lightning"].tolist()
    light_back = int(p["LIGHT_BACK_S"] * FPS)

    def restart_date(t, fallback, dead_since, kickoff):
        """Rule 9: the ball's first movement after lying still, if recent and inside the stoppage
        (a kick-off may sit on the first frame of the half), else the rule's own date."""
        m = int(move_onset[t])
        if m >= 0 and t - m <= move_back and (m > dead_since or (kickoff and m >= dead_since)):
            return min(m, fallback)
        return fallback

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
    no_cause = np.zeros(n, dtype=bool)     # threshold reached but rule 7 found no cause yet
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
        last_light = [-10 ** 9, -10 ** 9]  # last lightning touch of each team
        last_own_ev = first                 # last touch / clear control of the team in possession

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
            if p["CORNER_RULE"] and t in corners and t - last_change >= min_dead:
                start, kicker = corners[t]
                if owner != DEAD:
                    change(t - 1, start, DEAD, "corner")
                change(t, t, kicker, "corner kick")
                state[t] = owner
                continue
            # ---- rule 8: a long gap with the players walking ------------------------------
            if gap_start[t] and owner != DEAD:
                change(t, t, DEAD, "ball lost, players walking")
                state[t] = owner
                continue
            if owner == DEAD:
                # ---- leaving the dead ball ----------------------------------------------
                if kickoff and t - first > kickoff_max:
                    kickoff = False
                centre_ok = at_centre[t] or not kickoff
                if gap_end[t] and in_zone[t] and centre_ok and t - dead_since >= min_dead:
                    d0 = restart_date(t, t, dead_since, kickoff)
                    kickoff = False
                    change(t, d0, restart_team(d0, t, before_dead), "ball back")
                    state[t] = owner
                    continue
                soft = dead_reason in ("whistle", "players stopped")
                too_short = soft and before_dead in (HOME, AWAY) and \
                    (t - dead_since < d_min or t - last_change < d_min_after)
                if touch[t] >= 0 and in_zone[t] and centre_ok and (t - dead_since >= min_dead or soft):
                    if too_short:
                        kickoff = False
                        change(t, dead_since, before_dead, "dead ball too short")
                    else:
                        d0 = restart_date(t, t, dead_since, kickoff)
                        kickoff = False
                        change(t, d0, touch[t], "restart touch")
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
                elif sa >= 1.0 and (t - dead_since >= min_dead or kickoff):
                    d0 = restart_date(t, sa_onset, dead_since, kickoff)
                    kickoff = False
                    change(t, d0, restart_team(d0, t, before_dead), "alive play")
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
                has_cause = cause_cum[t + 1] - cause_cum[lo] > 0
                no_cause[t] = not has_cause
                # rule 7: without a contest or an offside the stoppage needs NO_CAUSE_PENALTY more
                if has_cause or sd >= thr_dead + p["NO_CAUSE_PENALTY"]:
                    if not has_cause:
                        refused.append((t, sd_onset))  # entered without a cause
                    change(t, sd_onset, DEAD, "whistle" if sd_whistle else "players stopped")
                    state[t] = owner
                    continue

            # ---- possession change, as in v1 ----------------------------------------------
            if owner in (HOME, AWAY) and (touch[t] == owner or ctrl[owner][t] > 0.5):
                last_own_ev = t
            if lightning[t] and touch[t] in (HOME, AWAY) and touch[t] != owner:
                last_light[touch[t]] = t
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
                reason = "touch" if c_acc[1] >= c_acc[0] else "control"
                lt = last_light[winner]
                if p["LIGHTNING"] and lt > last_own_ev and t - lt <= light_back and lt < start:
                    start, reason = lt, "lightning touch"
                change(t, start, winner, reason)
            state[t] = owner
    breakdown = {"ptot": ptot, "dtot": dtot, "atot": atot, "d_reset": d_reset, "a_reset": a_reset,
                 "no_cause": no_cause}
    return state, live, score, dead_score, alive_score, changes, refused, breakdown


def ball_out(ball, visible, prev_seen, pitch, p):
    """Rules 1 and 6: (out_dist, line_dist, out, out_start).

    out_start[k] >= 0 at the frame k a ball out is confirmed, holding the frame it went out: the
    frame it crossed the line, or, when it went out inside a gap, the last sighting (if that was
    near a line) or the middle of the gap."""
    n = len(ball)
    out_dist, line_dist = ball_out_distance(ball, pitch)
    out = runs_at_least(np.nan_to_num(out_dist, nan=0.0) > p["OUT_MARGIN"], p["OUT_MIN_FRAMES"])
    out_start = np.full(n, -1, dtype=np.int64)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], out.astype(np.int8)))) == 1)
    for e in edges:
        k = e + p["OUT_MIN_FRAMES"] - 1  # the frame the run is confirmed
        s0 = e
        while s0 - 1 >= 0 and visible[s0 - 1] and out_dist[s0 - 1] > 0:
            s0 -= 1  # back to the frame the ball crossed the line
        prev = prev_seen[s0]
        if prev >= 0 and s0 - prev > p["GAP_MIN"]:
            s0 = prev + 1 if line_dist[prev] <= p["NEAR_LINE"] else (prev + s0) // 2
        out_start[k] = s0
    return out_dist, line_dist, out, out_start


def whistle_points(events, f_rhythm, n, p):
    """Rule 4: (points, confirmation) per frame. A whistle earns W_WHISTLE times the mean braking
    of the players over the next WHISTLE_LOOK_S seconds."""
    look = int(p["WHISTLE_LOOK_S"] * FPS)
    cs = np.concatenate(([0.0], np.cumsum(np.nan_to_num(f_rhythm, nan=0.0))))
    pts = np.zeros(n, dtype=np.float32)
    conf = np.zeros(n, dtype=np.float32)
    for s0, e0, *_ in events:
        c = (cs[min(n, s0 + 1 + look)] - cs[min(n, s0 + 1)]) / look
        conf[s0:e0 + 1] = c
        pts[s0] = p["W_WHISTLE"] * c
    return pts, conf


def compute_features(teams, players, ball, ball_speed, blocks, pitch, p=PARAMS, whistle_events=(),
                     whistle_level=None):
    """Every per-frame feature the state machine reads, plus the ones the review video shows.

    Pure function of the arrays: no file is read, so the same code runs on any game (the BAR-ATH
    tracking, the fused tracking, or an external game used for calibration)."""
    n = len(ball)
    visible = ~np.isnan(ball[:, 0])
    idx = np.where(visible, np.arange(n), -1)
    prev_seen = np.concatenate(([-1], np.maximum.accumulate(idx)[:-1]))
    nxt = np.minimum.accumulate(np.where(visible, np.arange(n), n)[::-1])[::-1]
    next_seen = np.where(nxt < n, nxt, -1)

    # v1 features: control, touches, look-ahead
    vel, speed, dv, angle = v1.ball_kinematics(ball, ball_speed, p)
    dist, slot = v1.team_distances(players, ball, teams)
    ctrl, slow = v1.control(dist, speed, visible, p)
    touch_team, is_touch = v1.touches(dv, dist, p)
    conf = v1.look_ahead(ctrl, touch_team, p)

    # rules 1 and 6: ball out, with gaps handled
    out_dist, line_dist, out, out_start = ball_out(ball, visible, prev_seen, pitch, p)
    on_pitch = visible & (np.nan_to_num(out_dist, nan=np.inf) == 0)
    restart_zone = visible & (np.nan_to_num(out_dist, nan=np.inf) <= p["RESTART_MARGIN"])
    centre = np.hypot(ball[:, 0] - pitch[0] / 2, ball[:, 1] - pitch[1] / 2)
    at_centre = np.nan_to_num(centre, nan=np.inf) <= p["KICKOFF_R"]

    # rules 2 and 3: players far from the ball, players braking (entry) or slow (exit)
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

    deadness = combine(f_rhythm, p["W_RHYTHM"])  # entering: distance + braking (review video only)
    deadness_exit = combine(f_speed, 0.5)        # leaving: distance + absolute pace
    fast = visible & np.nan_to_num(speed > p["V_FAST"], nan=False)

    # rule 4: whistle, confirmed by the players braking afterwards
    whistle_pts, whistle_conf = whistle_points(whistle_events, f_rhythm, n, p)
    if whistle_level is None:
        whistle_level = np.zeros(n, dtype=np.float32)

    # rule 5: corner kicks
    corners = corner_kicks(players, teams, ball, dist, out, blocks, pitch, p)

    # rule 7: a cause -- a contest, a pass received offside, or a gap ending with the ball out
    duel = contests(players, teams, ball, p)
    offs = offsides(players, teams, ball, speed, touch_team, ctrl, slot, blocks, pitch, p)
    offside_at = np.zeros(n, dtype=bool)
    for r, _, _ in offs:
        offside_at[r] = True
    gap_out = ~visible & (next_seen >= 0) & \
        (np.nan_to_num(out_dist[np.maximum(next_seen, 0)], nan=0.0) > p["OUT_MARGIN"])
    cause_cum = np.concatenate(([0], np.cumsum(duel | offside_at | gap_out)))

    # rule 8: long gaps with the players walking, supported by another signal
    gap_dead_start = np.zeros(n, dtype=bool)
    gap_dead_end = np.zeros(n, dtype=bool)
    gap_why = []
    if p["GAP_RULE"]:
        slow_gap = (p["SPD_PLAY"] + p["SPD_DEAD"]) / 2
        sup = int(p["GAP_SUPPORT_S"] * FPS)
        L, Wd = pitch
        miss = np.flatnonzero(np.diff(np.concatenate(([0], (~visible).astype(np.int8), [0]))))
        for a, b in zip(miss[::2], miss[1::2]):
            if b - a < p["GAP_DEAD_S"] * FPS or b >= n:
                continue
            sp = pspeed[a:b]
            if np.isnan(sp).all() or np.nanmean(sp) > slow_gap:
                continue
            why = []
            if p["GAP_SUPPORT"]:
                lo, hi = max(0, a - sup), min(n, a + sup)
                last = prev_seen[a + 1] if a + 1 < n else -1
                if last >= 0 and (line_dist[last] <= p["NEAR_LINE"] or out_dist[last] > 0):
                    why.append("lost near a line")
                bx, by = ball[b]
                spot = (min(bx, L - bx) <= 6.5 and abs(by - Wd / 2) <= 10.0) or at_centre[b]
                if out_dist[b] > 0 or line_dist[b] <= p["NEAR_LINE"] or spot:
                    why.append("back at a line / set-piece spot")
                if np.nanmax(np.concatenate(([0.0], f_rhythm[lo:hi]))) >= 0.5:
                    why.append("players braking")
                if last >= 0 and np.nan_to_num(f_dist[max(0, last - FPS):last + 1]).mean() >= 0.5:
                    why.append("players far from the ball")
                if whistle_pts[lo:hi].max() > 0:
                    why.append("whistle")
                if not why:
                    continue
            gap_dead_start[a], gap_dead_end[b] = True, True
            gap_why.append((int(a), int(b), why))

    # rule 9: restart = the ball moving again after lying still
    move_onset = np.full(n, -1, dtype=np.int64)
    if p["RESTART_MOVE"]:
        sp = np.where(visible, speed, np.nan)
        slow = np.nan_to_num(sp, nan=np.inf) < p["V_MOVE"]
        slow_run = runs_at_least(slow, max(1, int(p["STILL_S"] * FPS)))
        cross = np.flatnonzero(slow_run[:-1] & (np.nan_to_num(sp[1:], nan=0.0) >= p["V_MOVE"])) + 1
        for c in cross:
            lo = max(0, c - FPS)
            k = lo + int(np.nanargmin(sp[lo:c]))  # the ball at its slowest before moving off
            move_onset[k] = k
        move_onset = np.maximum.accumulate(move_onset)

    # rule 10: lightning touches (interceptions sent away in a new direction)
    lightning = np.zeros(n, dtype=bool)
    if p["LIGHTNING"]:
        for t in np.flatnonzero(touch_team >= 0):
            out_sp = speed[t + 2:t + 7]
            out_sp = out_sp[np.isfinite(out_sp)]
            if len(out_sp) and out_sp.max() >= p["PASS_SPEED"] and np.nan_to_num(angle[t]) >= p["LIGHT_ANGLE"]:
                lightning[t] = True

    return {
        # read by decide()
        "ctrl": ctrl, "conf": conf, "touch_team": touch_team, "f_dist": f_dist,
        "f_rhythm": f_rhythm, "deadness_exit": deadness_exit, "visible": visible, "fast": fast,
        "restart_zone": restart_zone, "on_pitch": on_pitch, "out_start": out_start,
        "whistle_pts": whistle_pts, "corners": corners, "prev_seen": prev_seen, "dist": dist,
        "at_centre": at_centre, "cause_cum": cause_cum, "gap_dead_start": gap_dead_start,
        "gap_dead_end": gap_dead_end, "move_onset": move_onset, "lightning": lightning,
        "gap_why": gap_why,
        # diagnostics only
        "vel": vel, "speed": speed, "dv": dv, "angle": angle, "slot": slot, "slow": slow,
        "is_touch": is_touch, "out_dist": out_dist, "far": far, "pspeed": pspeed,
        "f_speed": f_speed, "rhythm": rhythm, "deadness": deadness, "whistle_level": whistle_level,
        "whistle_conf": whistle_conf, "whistle_events": len(whistle_events), "duel": duel,
        "offs": offs, "gap_out": gap_out,
    }


def to_labels(state, blocks, n):
    """H/A/D inside the graded blocks, X everywhere else."""
    labels = np.full(n, "X", dtype="<U1")
    for _, first, last in blocks:
        seg = state[first:last + 1]
        labels[first:last + 1] = np.select([seg == HOME, seg == AWAY, seg == DEAD], ["H", "A", "D"], "X")
    return labels


def predict(teams, players, ball, ball_speed, blocks, pitch, p=PARAMS, whistle_events=(),
            whistle_level=None):
    """Labels for every frame, plus features and machine outputs (for diagnostics)."""
    n = len(ball)
    f = compute_features(teams, players, ball, ball_speed, blocks, pitch, p, whistle_events,
                         whistle_level)
    state, live, score, dead_score, alive_score, changes, refused, bd = decide(f, blocks, n, p)
    out = {"state": state, "live": live, "score": score, "dead_score": dead_score,
           "alive_score": alive_score, "changes": changes, "refused": refused, "breakdown": bd}
    return to_labels(state, blocks, n), f, out


def save_diagnostics(path, f, r, p):
    """Per-frame scores and features for viz/v2_side_by_side.py (data/derived/, never published)."""
    bd = r["breakdown"]
    ch = np.array([(a, b, c) for a, b, c, _ in r["changes"]], dtype=np.int64).reshape(-1, 3)
    corner_arr = np.array([(t, s0, k) for t, (s0, k) in sorted(f["corners"].items())],
                          dtype=np.int64).reshape(-1, 3)
    np.savez_compressed(
        path, state=r["state"], live=r["live"], score=r["score"], dead_score=r["dead_score"],
        alive_score=r["alive_score"], ctrl=f["ctrl"], slow=f["slow"], speed=f["speed"],
        dv=f["dv"], angle=f["angle"], vel=f["vel"], dist=f["dist"], slot=f["slot"],
        touch_team=f["touch_team"], is_touch=f["is_touch"], conf=f["conf"],
        out_dist=f["out_dist"], far=f["far"], pspeed=f["pspeed"], f_dist=f["f_dist"],
        f_speed=f["f_speed"], deadness=f["deadness"], rhythm=f["rhythm"], f_rhythm=f["f_rhythm"],
        deadness_exit=f["deadness_exit"], whistle_level=f["whistle_level"],
        whistle_conf=f["whistle_conf"], changes=ch, reasons=np.array([x for *_, x in r["changes"]]),
        corners=corner_arr, threshold=p["THRESHOLD"], threshold_dead=p["THRESHOLD_DEAD"],
        duel=f["duel"], gap_out=f["gap_out"],
        offsides=np.array(f["offs"], dtype=np.int64).reshape(-1, 3),
        refused=np.array(r["refused"], dtype=np.int64).reshape(-1, 2),
        d_reset=bd["d_reset"], a_reset=bd["a_reset"], no_cause=bd["no_cause"],
        no_cause_penalty=p["NO_CAUSE_PENALTY"],
        **{"ptot_" + k: v for k, v in bd["ptot"].items()},
        **{"dtot_" + k: v for k, v in bd["dtot"].items()},
        **{"atot_" + k: v for k, v in bd["atot"].items()})


def print_summary(labels, f, r, blocks, names):
    """Aggregate counts only, nothing frame-level."""
    n = len(labels)
    graded = np.zeros(n, dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    print("teams: H = {}, A = {}".format(*names))
    print("whistle-like events {}, confirmed by players stopping {}".format(
        f["whistle_events"], int(np.sum(f["whistle_pts"] > 0))))
    print("corner kicks found: {}".format(len(f["corners"])))
    print("contest near the ball on {:.1f}% of graded frames; passes received offside: {}".format(
        100 * (f["duel"][graded].mean() if graded.any() else 0), len(f["offs"])))
    print("dead balls entered without a contest or offside (penalty paid): {}".format(
        len(r["refused"])))
    for half, first, last in blocks:
        mine = [c for c in r["changes"] if first <= c[0] <= last]
        seg = labels[first:last + 1]
        cut = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        starts, ends = np.concatenate(([0], cut)), np.concatenate((cut, [len(seg)]))
        poss = [e - s for s, e in zip(starts, ends) if seg[s] in "HA"]
        dead = [e - s for s, e in zip(starts, ends) if seg[s] == "D"]
        print("half {}: H {:.1f}%  A {:.1f}%  D {:.1f}%   {} possessions (mean {:.1f} s), "
              "{} dead balls (mean {:.1f} s)".format(
                  half, 100 * np.mean(seg == "H"), 100 * np.mean(seg == "A"),
                  100 * np.mean(seg == "D"), len(poss), np.mean(poss) / FPS, len(dead),
                  np.mean(dead) / FPS))
        print("   reasons: " + ", ".join("{} {}".format(k, v) for k, v in
                                         Counter(x for *_, x in mine).most_common()))


def main(source="tactical", use_whistle=True, fix_teams=False):
    p = PARAMS
    out_csv, out_diag = outputs(source, use_whistle)
    teams, players, ball, ball_speed, names = tracking.load(GAME, source)
    if fix_teams:
        teams, players, flipped = clean_tracking.fix_team_swaps(teams, players)
        print("team swaps fixed: {} tracklets moved to the other team".format(len(flipped)))
    blocks = half_video_blocks(GAME)
    pitch = GAMES[GAME]["pitch"]
    events, level = whistle.load(GAME) if use_whistle else ((), None)

    labels, f, r = predict(teams, players, ball, ball_speed, blocks, pitch, p, events, level)
    write_submission(out_csv, labels)
    save_diagnostics(out_diag, f, r, p)
    print_summary(labels, f, r, blocks, names)
    print("-> {}".format(os.path.relpath(out_csv, ROOT)))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracking", choices=("tactical", "fused"), default="tactical",
                    help="fused = tactical + broadcast (run sync_broadcast.py and fusion.py first)")
    ap.add_argument("--no-whistle", action="store_true",
                    help="skip rule 4 (no tactical video needed); writes *_nowhistle outputs")
    ap.add_argument("--fix-teams", action="store_true",
                    help="move tracklets carried by the wrong team first (clean_tracking.py)")
    a = ap.parse_args()
    main(a.tracking, not a.no_whistle, a.fix_teams)
