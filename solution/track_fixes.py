"""Algorithmic fixes of a player tracking: goalkeepers on the wrong team, positions frozen out of view.

No labels, no frame-specific constants: every threshold is a physical quantity (metres, m/s, s).
Both fixes run unchanged on any game; the keeper fix needs only teams, players, ball and the
graded blocks, the frozen fix also needs the camera homography (pitch 0-1 -> pixels).

fix_keepers: in each half every team defends one goal (side of its mean x). With the ball far
from a goal (> BALL_FAR, i.e. in the other half), nobody but that goal's keeper stands within
R_GK of its centre: an outfield opponent there would be ~30 m beyond the defensive line, in an
offside position, for seconds. A tracklet (maximal run of one slot) labelled team k that, on the
frames where the ball is far from the goal team 1-k defends, stands within R_GK of that goal on
at least SHARE of them (and for at least MIN_FAR_S) is team 1-k's keeper: it moves to team 1-k,
unless team 1-k already has someone else in that zone on most of those frames (one keeper per
team). Output layout as clean_tracking.fix_team_swaps (twice the slots, copies on the other team),
so the two fixes chain; compact() merges the copies of a chain back to 2 x the original slots.

frozen_out_of_view: a position that projects outside the image cannot be measured by the camera;
the ATD keeps such players where they were last seen (median speed 0.06 m/s against 1.41 m/s in
view, analisi/audit_tracking.py). A held position is plausible for a short while only: after
HOLD_S a player moving at a typical in-view pace is already further than a control radius from
it. Positions out of view, still (< V_STILL), beyond HOLD_S of such a run, are masked. Optionally
(held_in_view) also positions IN view that do not move at all (step < HELD_STEP per frame, below
the ~2 mm jitter of a measured still player) for more than HOLD_S: the tracker is holding a lost
object there too.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from clean_tracking import MAX_PER_TEAM, tracklets  # noqa: E402

FPS = 25

# keepers
BALL_FAR = 40.0      # m from the goal line: ball beyond the penalty area by more than 20 m
R_GK = 25.0          # m from the goal centre: Bundesliga keepers stand 14-24 m out (median) with the ball far
MIN_FAR_S = 2.0      # s of ball-far frames as evidence
SHARE = 0.8          # of those frames inside R_GK

# frozen out of view
IMG_W, IMG_H, IMG_MARGIN = 1920, 1080, 10
V_STILL = 0.5        # m/s, slower than walking (1-1.5 m/s)
HOLD_S = 2.0         # s: a player at the median in-view pace (1.4 m/s) is ~3 m away by then
SPEED_WIN = 5        # frames, central difference for the speed
HELD_STEP = 0.001    # m per frame: a measured still player jitters ~2 mm per frame


def own_goals(teams, players, a, z, pitch):
    """x of the goal line each team defends in frames a..z: the team with the smaller mean x
    defends x = 0 (comparing the two means, not each mean with L/2: a team pinned in its half
    can have both means on the same side)."""
    L = pitch[0]
    mx = [np.nanmean(players[a:z + 1][:, teams == k, 0]) for k in (0, 1)]
    return [0.0, L] if mx[0] < mx[1] else [L, 0.0]


def keeper_candidates(teams, players, ball, blocks, pitch, ball_far=BALL_FAR, r_gk=R_GK, min_far_s=MIN_FAR_S):
    """[(slot, first, stop, far frames, share near, share deepest, share both)] for every tracklet
    with at least min_far_s of ball-far frames w.r.t. the goal the OTHER team defends."""
    W = pitch[1]
    ok = ~np.isnan(players[..., 0])
    bx = ball[:, 0]
    out = []
    for _, a, z in blocks:
        gx = own_goals(teams, players, a, z, pitch)
        P = players[a:z + 1]
        dg = [np.hypot(P[..., 0] - gx[j], P[..., 1] - W / 2) for j in (0, 1)]
        far = [np.abs(np.nan_to_num(bx[a:z + 1], nan=gx[j]) - gx[j]) > ball_far for j in (0, 1)]
        # distance of team j's deepest player to the goal j defends
        deep = [np.nanmin(np.where(np.isnan(dg[j][:, teams == j]), np.inf, dg[j][:, teams == j]), axis=1)
                for j in (0, 1)]
        for s, t0, t1 in tracklets(ok[a:z + 1]):
            j = 1 - int(teams[s])                      # the team whose goal it may be guarding
            fm = far[j][t0:t1]
            nf = int(fm.sum())
            if nf < min_far_s * FPS:
                continue
            d = dg[j][t0:t1, s][fm]
            near = d < r_gk
            deepest = d < deep[j][t0:t1][fm]
            out.append((s, a + t0, a + t1, nf, float(near.mean()), float(deepest.mean()),
                        float((near & deepest).mean())))
    return out


def fix_keepers(teams, players, ball, blocks, pitch, ball_far=BALL_FAR, r_gk=R_GK, min_far_s=MIN_FAR_S,
                share=SHARE, return_info=False):
    """(teams2, players2, flipped [(slot, first, stop)]) in the fix_team_swaps layout.

    Flip: on >= share of its ball-far frames the tracklet is within r_gk of the goal the other
    team defends AND closer to it than every player of that team (it is that team's last man:
    one keeper per team)."""
    n = len(players)
    team_of = np.repeat(teams[None].astype(np.int8), n, axis=0)
    info = keeper_candidates(teams, players, ball, blocks, pitch, ball_far, r_gk, min_far_s)
    flipped = []
    for s, a, b, nf, sn, sd, sb in info:
        if sb >= share:
            team_of[a:b, s] = 1 - teams[s]
            flipped.append((s, a, b))
    teams2, players2 = _materialise(teams, players, team_of)
    if return_info:
        return teams2, players2, flipped, info
    return teams2, players2, flipped


def _materialise(teams, players, team_of):
    n, slots = players.shape[:2]
    players2 = np.full((n, 2 * slots, 2), np.nan, dtype=players.dtype)
    moved = team_of != teams[None]
    players2[:, :slots][~moved] = players[~moved]
    players2[:, slots:][moved] = players[moved]
    return np.concatenate([teams, 1 - teams]).astype(teams.dtype), players2


def compact(teams2, players2, base_slots):
    """Merge the copies made by chained fixes (slot s, s + S, s + 2S, ... are one original slot
    S = base_slots) into one slot per (original slot, team): 2 * base_slots slots."""
    n, m = players2.shape[:2]
    out = np.full((n, 2 * base_slots, 2), np.nan, dtype=players2.dtype)
    t_out = np.concatenate([np.full(base_slots, -1), np.full(base_slots, -1)])
    for j in range(m):
        s = j % base_slots
        k = int(teams2[j])
        # slot s keeps its first team, s + base_slots holds the other
        if t_out[s] in (-1, k):
            d = s
        else:
            d = s + base_slots
        t_out[d] = k
        has = ~np.isnan(players2[:, j, 0])
        out[has, d] = players2[has, j]
    for s in range(base_slots):  # unused copies: the other team of the slot
        if t_out[s + base_slots] == -1:
            t_out[s + base_slots] = 1 - t_out[s]
    return t_out.astype(teams2.dtype), out


def team_count_stats(teams, players, graded):
    """{team: (share 12+, share 10-, share exactly 11)} on the graded frames."""
    ok = ~np.isnan(players[graded][..., 0])
    out = {}
    for k in (0, 1):
        c = ok[:, teams == k].sum(axis=1)
        out[k] = (float(np.mean(c > MAX_PER_TEAM)), float(np.mean(c < MAX_PER_TEAM)),
                  float(np.mean(c == MAX_PER_TEAM)))
    return out


# ---- frozen out of view ------------------------------------------------------------------------
def in_view(H, x, y, pitch):
    """H (n, 3, 3) pitch(0-1) -> pixels; x, y metres (n, k) -> bool (n, k), inside the image."""
    L, W = pitch
    u, v = x / L, y / W
    X = H[:, 0, 0, None] * u + H[:, 0, 1, None] * v + H[:, 0, 2, None]
    Y = H[:, 1, 0, None] * u + H[:, 1, 1, None] * v + H[:, 1, 2, None]
    Z = H[:, 2, 0, None] * u + H[:, 2, 1, None] * v + H[:, 2, 2, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        px, py = X / Z, Y / Z
    return (Z > 0) & (px >= IMG_MARGIN) & (px < IMG_W - IMG_MARGIN) & (py >= IMG_MARGIN) & (py < IMG_H - IMG_MARGIN)


def view_mask(H, players, pitch, chunk=20000):
    """(n, slots) bool, True where a tracked position projects inside the image."""
    n = len(players)
    v = np.zeros(players.shape[:2], bool)
    for c in range(0, n, chunk):
        sl = slice(c, min(n, c + chunk))
        v[sl] = in_view(H[sl], np.nan_to_num(players[sl, :, 0]), np.nan_to_num(players[sl, :, 1]), pitch)
    return v & ~np.isnan(players[..., 0])


def _after_hold(mask, hold):
    out = np.zeros_like(mask)
    for s in range(mask.shape[1]):
        e = np.flatnonzero(np.diff(np.concatenate(([0], mask[:, s].astype(np.int8), [0]))))
        for a, b in zip(e[::2], e[1::2]):
            if b - a > hold:
                out[a + hold:b, s] = True
    return out


def frozen_out_of_view(players, H, pitch, v_still=V_STILL, hold_s=HOLD_S, held_in_view=False):
    """(n, slots) bool: out of view, still, after the first hold_s of such a run (plus, with
    held_in_view, motionless positions in view after hold_s)."""
    vis = view_mask(H, players, pitch)
    ok = ~np.isnan(players[..., 0])
    h = SPEED_WIN // 2
    spd = np.full(players.shape[:2], np.nan)
    spd[h:-h] = np.linalg.norm(players[2 * h:] - players[:-2 * h], axis=2) * FPS / (2 * h)
    still = ok & ~vis & (np.nan_to_num(spd, nan=0.0) < v_still)
    hold = int(round(hold_s * FPS))
    out = _after_hold(still, hold)
    if held_in_view:
        step = np.full(players.shape[:2], np.inf)
        step[1:] = np.nan_to_num(np.linalg.norm(np.diff(players, axis=0), axis=2), nan=np.inf)
        out |= _after_hold(ok & vis & (step < HELD_STEP), hold)
    return out


def drop_frozen(players, H, pitch, **kw):
    """(players with frozen out-of-view positions set to NaN, mask)."""
    m = frozen_out_of_view(players, H, pitch, **kw)
    p = players.copy()
    p[m] = np.nan
    return p, m
