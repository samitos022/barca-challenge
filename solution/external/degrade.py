"""Make a clean external tracking look like the BAR-ATH one, so calibration sees the same problems.

The Metrica sample games already lose the ball like BAR-ATH does (59-68% visible vs 59%, ~97% of
missing frames in gaps of 5 s or more), so the ball is left alone (for tracking that sees the ball
almost always, like IDSSE's TRACAB, see degrade_ball() below). Their players, though, are
perfect: one slot per person, never lost, never on the wrong team. BAR-ATH has a team over 11 on
6.4% of team-frames (a player carried by the other team's slot) and under 9 on 8.8%.

Model: every player's track is cut into tracklets with BAR-ATH-like lengths (exponential, mean
TRACKLET_S); each tracklet is moved to the other team with probability SWAP, or dropped with
probability DROP. On top, per team, BURST_EVERY_S on average, BURST_K players are lost together
for about BURST_S (the tactical camera loses a group, e.g. at the far side). The defaults match
BAR-ATH's per-team count histogram (11: 57.6%, 10: 21.7%, 9: 5.5%, <=8: 4.6%, 12: 5.9%), printed
by running this file. Seeded: the same seed gives the same degraded game.
"""

import os
import warnings

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


# ---- ball ------------------------------------------------------------------------------------
# For tracking that sees the ball almost always (IDSSE / TRACAB). BAR-ATH's ball, measured without
# labels by analisi/audit_tracking.py: missing on 41% of the graded frames, all in gaps of 5 s or
# more, median gap 10.7 s, 44% of the gap frames in gaps of 30 s or more, about 132 gaps per 106
# graded minutes; the players' mean pace is 1.24 m/s inside gaps (median over gaps) against 2.10
# m/s with the ball seen; last sighting before a gap within 3 m of a line 35%, inside a penalty
# area 21%, elsewhere 44%. degrade_ball() reproduces these from observables only (player pace,
# ball position), never from the labels: it does not know where the ball is dead, it only prefers
# slow phases, as BAR-ATH's detector seems to.

BALL_TARGETS = {"missing": 0.41, "median_gap_s": 10.7, "share_30s": 0.44, "gaps_per_106min": 132,
                "pace_gap": 1.24, "pace_seen": 2.10, "last_line": 0.35, "last_box": 0.21,
                "last_else": 0.44}
GAP_MU, GAP_SIGMA = 1.34, 1.30   # log-normal gap length (s) kept above GAP_MIN_S: median 10.9 s,
GAP_MIN_S, GAP_MAX_S = 5.0, 240.0  # 45% of the frames in gaps >= 30 s, mean 18.4 s
MIN_SEEN_S = 1.0                 # s of ball seen between two gaps
FIT_ROUNDS = 6


def pace(players):
    """Per-frame mean speed of the tracked players (m/s), as analisi/audit_tracking.py measures it."""
    vel = np.linalg.norm(np.gradient(players.astype(np.float64), axis=0), axis=2) * FPS
    with np.errstate(invalid="ignore"):
        return np.nanmean(vel, axis=1)


def last_sighting_class(ball, pitch):
    """0 within 3 m of a line, 1 inside a penalty area, 2 elsewhere, -1 ball unseen."""
    L, W = pitch
    x, y = ball[:, 0], ball[:, 1]
    dline = np.minimum.reduce([x, L - x, y, W - y])
    box = ((x < 16.5) | (x > L - 16.5)) & (np.abs(y - W / 2) < 20.16)
    c = np.where(dline < 3, 0, np.where(box, 1, 2))
    c[np.isnan(x)] = -1
    return c


def _runs(mask):
    idx = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return idx[::2], idx[1::2]


def ball_stats(ball, players, graded, pitch):
    """The BALL_TARGETS statistics of a ball track, on the graded frames."""
    seen = ~np.isnan(ball[:, 0])
    st, en = _runs(~seen & graded)
    gl = (en - st) / FPS
    pc = pace(players)
    longg = [(a, b) for a, b in zip(st, en) if b - a >= GAP_MIN_S * FPS]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # gaps where no player is tracked
        gp = np.array([np.nanmean(pc[a:b]) for a, b in longg]) if longg else np.array([np.nan])
    last = np.array([a - 1 for a, _ in longg if a > 0 and seen[a - 1]], dtype=np.int64)
    cls = last_sighting_class(ball[last], pitch) if len(last) else np.array([-1])
    minutes = graded.sum() / FPS / 60
    return {"missing": float(np.mean(~seen[graded])),
            "median_gap_s": float(np.median(gl)) if len(gl) else np.nan,
            "share_30s": float(gl[gl >= 30].sum() / max(gl.sum(), 1e-9)),
            "share_5s": float(gl[gl >= 5].sum() / max(gl.sum(), 1e-9)),
            "gaps_per_106min": len(longg) / minutes * 106,
            "pace_gap": float(np.nanmedian(gp)) if np.any(~np.isnan(gp)) else np.nan, "pace_seen": float(np.nanmean(pc[graded & seen])),
            "last_line": float(np.mean(cls == 0)), "last_box": float(np.mean(cls == 1)),
            "last_else": float(np.mean(cls == 2))}


def _place(rng, lengths, pc, cls, blocks, tau, cw, n):
    """Gap starts, longest first, each drawn with weight exp(-mean pace / tau) x cw[class]."""
    taken = np.zeros(n, dtype=bool)
    inblock = np.zeros(n, dtype=bool)
    for _, a, b in blocks:
        inblock[a + 1:b + 1] = True   # never the first frame of a half
    pcf = np.where(np.isnan(pc), np.nanmean(pc), pc)
    cs = np.concatenate(([0.0], np.cumsum(pcf)))
    cin = np.concatenate(([0], np.cumsum(~inblock)))
    clsw = np.where(cls >= 0, np.asarray(cw)[np.maximum(cls, 0)], 0.0)
    clsw = np.concatenate(([0.0], clsw[:-1]))   # class of the frame before the start
    margin = int(MIN_SEEN_S * FPS)
    gaps = []
    for L in lengths:
        if L >= n:
            continue
        s = np.arange(n - L + 1)
        ct = np.concatenate(([0], np.cumsum(taken)))
        lo, hi = np.maximum(s - margin, 0), np.minimum(s + L + margin, n)
        free = (ct[hi] - ct[lo] == 0) & (cin[s + L] - cin[s] == 0)
        if not free.any():
            continue
        mp = (cs[s + L] - cs[s]) / L
        w = np.exp(-(mp - mp[free].min()) / tau) * clsw[s] * free
        tot = w.sum()
        if tot <= 0:
            continue
        a = int(rng.choice(s, p=w / tot))
        taken[a:a + L] = True
        gaps.append((a, a + L))
    return gaps


def _gap_lengths(rng, want):
    """Gap lengths in frames, longest first, summing to want: stratified quantiles of the
    truncated log-normal (one random point per stratum), so every game gets the target shape
    rather than a noisy draw of a heavy-tailed law."""
    from statistics import NormalDist
    nd = NormalDist()
    lo = nd.cdf((np.log(GAP_MIN_S) - GAP_MU) / GAP_SIGMA)
    hi = nd.cdf((np.log(GAP_MAX_S) - GAP_MU) / GAP_SIGMA)

    def draw(k):
        u = lo + (hi - lo) * (np.arange(k) + rng.random(k)) / k
        return np.exp(GAP_MU + GAP_SIGMA * np.array([nd.inv_cdf(x) for x in u])) * FPS

    k = max(1, int(round(want / draw(400).mean())))
    x = draw(k)
    x = x * want / x.sum()   # exact total; scales lengths by a few percent at most
    return sorted((max(int(GAP_MIN_S * FPS), int(round(v))) for v in x), reverse=True)


def degrade_ball(ball, players, blocks, pitch, seed, targets=BALL_TARGETS, rounds=FIT_ROUNDS):
    """(ball2, stats): the ball removed in BAR-ATH-like gaps. players: the tracking the method
    will see (pass the degraded players). The weights (pace temperature, one per last-sighting
    class) are fitted on this game, with this seed, to the targets in a few rounds."""
    rng = np.random.default_rng(seed)
    n = len(ball)
    graded = np.zeros(n, dtype=bool)
    for _, a, b in blocks:
        graded[a:b + 1] = True
    pc = pace(players)
    cls = last_sighting_class(ball, pitch)
    lengths = _gap_lengths(rng, targets["missing"] * graded.sum())
    tau, cw = 0.5, np.array([1.0, 1.0, 1.0])
    goal = np.array([targets["last_line"], targets["last_box"], targets["last_else"]])
    best = None
    for k in range(rounds + 1):
        gaps = _place(np.random.default_rng(seed * 1000 + k), lengths, pc, cls, blocks, tau, cw, n)
        out = ball.copy()
        for a, b in gaps:
            out[a:b] = np.nan
        st = ball_stats(out, players, graded, pitch)
        got = np.array([st["last_line"], st["last_box"], st["last_else"]])
        err = np.sum(np.abs(got - goal)) + abs(st["pace_gap"] / targets["pace_gap"] - 1)
        if best is None or err < best[0]:
            st["tau"], st["class_w"] = tau, cw.tolist()
            best = (err, out, st)
        cw = cw * (goal / np.maximum(got, 0.02)) ** 0.8
        cw = cw / cw.max()
        tau = float(np.clip(tau * (targets["pace_gap"] / st["pace_gap"]) ** 3, 0.03, 5.0))
    return best[1], best[2]



# ---- shape of the ball track ---------------------------------------------------------------------
# BAR-ATH's ATD ball is blurred and carries a slowly varying position error, unlike TRACAB's or
# Metrica's: v1 sees 16.7 touches per visible minute (30-43 on the clean external balls), only 56%
# of them with a player within R_TOUCH (88-93%), velocity changes p99 8.7 m/s (20-22), and the ball
# is 1.46 m from the nearest player at a velocity change (0.57 m on TRACAB). It is not late (the
# distance is smallest at zero time shift, analisi/ritardo_palla.py) and the error has no
# systematic direction. Fitted on those unlabelled numbers (analisi/fit_palla_realistica.py):
SHAPE = {"blur": 6.0, "amp": 0.7, "tau": 12.0, "jitter": 0.04}


def _smooth_runs(x, sigma):
    """Gaussian smoothing (sigma in frames) inside every run of non-NaN rows; gaps stay gaps."""
    r = int(3 * sigma) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    out = x.copy()
    ok = ~np.isnan(x[:, 0])
    e = np.flatnonzero(np.diff(np.concatenate(([0], ok.astype(np.int8), [0]))))
    for a, b in zip(e[::2], e[1::2]):
        pad = np.pad(x[a:b], ((r, r), (0, 0)), mode="edge")
        out[a:b] = np.stack([np.convolve(pad[:, j], k, mode="valid") for j in range(x.shape[1])], 1)
    return out


def shape_ball(ball, seed, shape=SHAPE):
    """Blur + slow position error + jitter, like BAR-ATH's ATD ball. NaN rows are kept."""
    rng = np.random.default_rng(seed)
    out = _smooth_runs(ball.astype(np.float64), shape["blur"])
    for amp, tau in ((shape["amp"], shape["tau"]), (shape["jitter"], 2.0)):
        z = rng.normal(0, 1, size=out.shape)
        z[np.isnan(out[:, 0])] = np.nan
        z = _smooth_runs(z, tau)
        z /= np.nanstd(z)
        out = out + amp * z
    return out.astype(ball.dtype)

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


# ---- keeper holding the ball ---------------------------------------------------------------------
# BAR-ATH's tracker loses the ball while a goalkeeper holds it in his hands (the body hides it):
# the gap starts around the catch and ends a little after the release. degrade_ball() places its
# gaps from the players' pace only, so it does not reproduce that. hide_keeper_holds() does, on a
# game whose truth is known: it finds the stretches where a team is in possession with the ball at
# its deepest player, inside its own penalty area, and removes the ball there.
HOLD_R = 1.5         # m between the ball and the deepest player: he has it in his hands or at his feet
HOLD_MIN_S = 1.5     # s, shorter is a touch
HOLD_BRIDGE = 10     # frames: the condition may drop out for this long inside one hold
HOLD_P = 0.85        # share of the holds hidden
HOLD_EARLY = (0, 12)   # frames before the hold where the ball is already lost (the catch itself)
HOLD_LATE = (5, 25)    # frames after the release before the ball is found again


def keeper_holds(teams, players, ball, gt, blocks, pitch):
    """[(first, stop, team)]: team (0 / 1, as in gt) in possession, ball within HOLD_R of its
    deepest player, inside its own penalty area, for HOLD_MIN_S or more. Clean tracking only."""
    L, W = pitch
    out = []
    for _, a, z in blocks:
        mx = [np.nanmean(players[a:z + 1][:, teams == k, 0]) for k in (0, 1)]
        gx = [0.0, L] if mx[0] < mx[1] else [L, 0.0]
        B = ball[a:z + 1]
        for k in (0, 1):
            P = players[a:z + 1][:, teams == k]
            dep = np.abs(P[..., 0] - gx[k])
            j = np.argmin(np.where(np.isnan(dep), np.inf, dep), axis=1)
            kp = P[np.arange(len(P)), j]
            with np.errstate(invalid="ignore"):
                cond = (gt[a:z + 1] == k) & (np.hypot(B[:, 0] - kp[:, 0], B[:, 1] - kp[:, 1]) <= HOLD_R) & \
                    (np.abs(B[:, 0] - gx[k]) <= 16.5) & (np.abs(B[:, 1] - W / 2) <= 20.16)
            st, en = _runs(cond)
            merged = []
            for s0, e0 in zip(st, en):
                if merged and s0 - merged[-1][1] <= HOLD_BRIDGE:
                    merged[-1][1] = e0
                else:
                    merged.append([s0, e0])
            out += [(a + int(s0), a + int(e0), k) for s0, e0 in merged if e0 - s0 >= HOLD_MIN_S * FPS]
    return sorted(out)


def hide_keeper_holds(ball, holds, seed):
    """(ball with HOLD_P of the holds removed, [(first, stop, team, hidden first, hidden stop)])."""
    rng = np.random.default_rng(seed + 4241)
    out = ball.copy()
    done = []
    for a, b, k in holds:
        if rng.random() >= HOLD_P:
            continue
        a2 = max(0, a - int(rng.integers(HOLD_EARLY[0], HOLD_EARLY[1] + 1)))
        b2 = min(len(ball), b + int(rng.integers(HOLD_LATE[0], HOLD_LATE[1] + 1)))
        out[a2:b2] = np.nan
        done.append((a, b, k, a2, b2))
    return out, done


# ---- the ball moved about during a stoppage --------------------------------------------------------
# In the TRACAB matches the ball stands still while play is dead (it moves faster than 2 m/s on
# 0.2% of the dead frames) and in the Metrica samples it is missing. A camera-based tracker like
# BAR-ATH's keeps following it: somebody fetches the ball, carries it to the spot and puts it down
# (in BAR-ATH's public five minutes the ball is tracked on 66% of the dead frames and moving on 60%
# of those). move_dead_ball() rewrites the ball inside the stoppages of a game whose truth is known:
# it stays where it went out for a moment, travels to the player who will take the restart (the one
# closest to the restart spot when play resumes), follows him while he walks there, and is put down
# on the spot shortly before the restart. Physical values only; nothing is fitted to BAR-ATH.
MOVE_P = 0.8             # share of the stoppages rewritten
MOVE_MIN_S = 3.0         # s: shorter stoppages keep the ball as it is
MOVE_REST_S = (0.5, 2.5)   # s the ball stays where it went out
MOVE_FETCH_V = (6.0, 14.0)  # m/s of the ball sent to the taker (a throw from a ball boy, a pass back)
MOVE_PLACE_S = (0.6, 2.0)  # s the ball lies on the spot before the restart
MOVE_HIDE_P = 0.3        # share of the rewritten stoppages where the ball is not seen until the taker has it
MOVE_CARRY_R = 0.5       # m from the taker while he carries it


def move_dead_ball(ball, players, gt, seed, dead=2):
    """Ball with the stoppages rewritten as above. gt: truth codes per frame (dead = 2)."""
    rng = np.random.default_rng(seed + 9133)
    out = ball.copy()
    n = len(ball)
    st, en = _runs(gt == dead)
    for a, b in zip(st, en):
        if b - a < MOVE_MIN_S * FPS or b >= n or a == 0 or rng.random() >= MOVE_P:
            continue
        spot = ball[b] if not np.isnan(ball[b, 0]) else None
        if spot is None:
            nxt = np.flatnonzero(~np.isnan(ball[b:b + 3 * FPS, 0]))
            if not len(nxt):
                continue
            spot = ball[b + nxt[0]]
        d = np.linalg.norm(players[b] - spot, axis=1)
        if np.isnan(d).all():
            continue
        taker = int(np.nanargmin(d))
        track = players[a:b, taker]
        if np.isnan(track[:, 0]).mean() > 0.3:
            continue
        ok = ~np.isnan(track[:, 0])
        idx = np.arange(b - a)
        track = np.stack([np.interp(idx, idx[ok], track[ok, k]) for k in (0, 1)], axis=1)
        m = b - a
        place = min(int(rng.uniform(*MOVE_PLACE_S) * FPS), m // 3)
        rest = min(int(rng.uniform(*MOVE_REST_S) * FPS), m // 3)
        start = ball[a - 1] if not np.isnan(ball[a - 1, 0]) else track[0]
        seg = np.empty((m, 2), dtype=np.float64)
        seg[:rest] = start
        # sent to the taker: it arrives where he is at that moment
        v = rng.uniform(*MOVE_FETCH_V)
        t1 = rest
        for _ in range(3):   # the flight time depends on where the taker is when it arrives
            t1 = min(m - place - 1, rest + max(2, int(np.linalg.norm(track[min(t1, m - 1)] - start) / v * FPS)))
        t1 = max(t1, rest + 1)
        w = np.linspace(0, 1, t1 - rest + 1)[1:, None]
        seg[rest:t1] = start + w * (track[t1 - 1] - start)
        ang = rng.uniform(0, 2 * np.pi)
        off = MOVE_CARRY_R * np.array([np.cos(ang), np.sin(ang)])
        seg[t1:m - place] = track[t1:m - place] + off
        # put down on the spot
        k0 = m - place
        w = np.clip(np.arange(place) / max(int(0.4 * FPS), 1), 0, 1)[:, None]
        seg[k0:] = seg[k0 - 1] + w * (spot - seg[k0 - 1])
        if rng.random() < MOVE_HIDE_P:
            seg[:t1] = np.nan
        out[a:b] = seg
    return out


# ---- the tactical camera's framing (12/11) -----------------------------------------------------
# BAR-ATH's tactical camera never shows the whole pitch (59% in median): the two corner flags on its
# far side are in the image on 2% of the frames, so at a corner from that side the taker is never
# tracked, and players out of the image are lost (tactical_clean removes the frozen copies the
# tracker leaves there). External tracking sees everybody. camera_view() keeps, on every external
# frame, only what BAR-ATH's camera would show: the homography of a BAR-ATH frame whose players
# stand in a similar place (players' centroid), chosen among the CAM_K nearest, a new choice every
# CAM_SEG_S. Positions only, no label: the framing of the camera is a property of the video.
CAM_BANK = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "data", "derived", "camera_bank.npz")
CAM_STEP = 5           # BAR-ATH frames between bank samples
CAM_K = 20
CAM_SEG_S = 5.0


def _camera_bank():
    if os.path.exists(CAM_BANK):
        z = np.load(CAM_BANK)
        return z["H"], z["c"], tuple(z["pitch"])
    import sys
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in (here, os.path.dirname(here)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import tracking
    from games import GAME, GAMES, asset_paths, half_video_blocks
    teams, players, _, _, _ = tracking.load(GAME, "tactical")
    Hm = np.loadtxt(asset_paths(GAME)["homography"], delimiter=",", skiprows=1,
                    usecols=range(1, 10)).reshape(-1, 3, 3)
    idx = np.concatenate([np.arange(a, z + 1, CAM_STEP) for _, a, z in half_video_blocks(GAME)])
    idx = idx[idx < len(Hm)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        c = np.stack([np.nanmean(players[idx, :, 0], axis=1), np.nanmean(players[idx, :, 1], axis=1)], axis=1)
    ok = np.isfinite(c).all(axis=1) & np.isfinite(Hm[idx]).all(axis=(1, 2))
    pitch = np.array(GAMES[GAME]["pitch"], dtype=np.float64)
    np.savez_compressed(CAM_BANK, H=Hm[idx][ok], c=c[ok] / pitch, pitch=pitch)
    return Hm[idx][ok], c[ok] / pitch, tuple(pitch)


def camera_view(players, ball, pitch, seed):
    """(players, ball) with what BAR-ATH's tactical camera would not see set to NaN."""
    from scipy.spatial import cKDTree
    import sys
    sol = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if sol not in sys.path:
        sys.path.insert(0, sol)
    import track_fixes as tf
    Hb, cb, _ = _camera_bank()
    L, W = pitch
    n = len(players)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        c = np.stack([np.nanmean(players[..., 0], axis=1) / L, np.nanmean(players[..., 1], axis=1) / W], axis=1)
    ok = np.isfinite(c).all(axis=1)
    rng = np.random.default_rng(seed)
    seg = int(CAM_SEG_S * FPS)
    rank = np.repeat(rng.integers(0, CAM_K, size=n // seg + 1), seg)[:n]
    j = np.zeros(n, dtype=np.int64)
    if ok.any():
        _, nn = cKDTree(cb).query(c[ok], k=CAM_K)
        j[ok] = nn[np.arange(ok.sum()), rank[ok]]
    H = Hb[j]
    p2, b2 = players.copy(), ball.copy()
    for a in range(0, n, 20000):
        sl = slice(a, min(n, a + 20000))
        vis = tf.in_view(H[sl], np.nan_to_num(players[sl, :, 0]), np.nan_to_num(players[sl, :, 1]), pitch)
        vis &= ok[sl, None]
        p2[sl][~vis & ok[sl, None]] = np.nan
        bv = tf.in_view(H[sl], np.nan_to_num(ball[sl, 0])[:, None], np.nan_to_num(ball[sl, 1])[:, None], pitch)[:, 0]
        b2[sl][~bv & ok[sl]] = np.nan
    return p2, b2
