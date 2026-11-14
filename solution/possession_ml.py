"""Who has the ball, learned on OTHER games: a frame classifier + a switching-cost decoder.

possession_v2 decides the dead ball well enough; what it gets wrong most is which team has the ball
while play is alive (on the external games a perfect team decision would take spellF1 from 0.32 to
0.76, a perfect dead ball only to 0.42: analisi/dove_si_perde.py). This module replaces that one
decision with a model trained on the external games (IDSSE Bundesliga, Metrica samples; RULES.md:
labels on other games are allowed), on tracking degraded to look like BAR-ATH's (calibrate
"realistic"). BAR-ATH is never used for training or for choosing anything.

1. Features, per frame, from the tracking only (what BAR-ATH has too), made team-symmetric: every
   team quantity appears as (home, away) in a fixed order, the pitch is seen from the home team's
   attacking direction, and training doubles every row with the teams swapped, so the model learns
   "a team" and not "the home team":
     ball: seen, speed, x towards the home attack, |y - centre|, velocity towards the home attack,
           seconds since / until seen;
     per team: closest and second closest player to the ball, players within 5 / 10 m, approach
           speed of the closest player, v1 control and touch, centroid x relative to the ball,
           width and length of the team, mean pace;
     context: means of the key differences over centred windows of 0.5 s and 2 s, and over the
           past / next second.
2. Model: gradient-boosted trees (scikit-learn HistGradientBoostingClassifier), P(home has the ball)
   on alive frames.
3. Decoder: inside every alive stretch of possession_v2, the most likely H/A sequence (Viterbi) under
   a cost SWITCH_COST (in log-likelihood) per change of possession. The cost is chosen on the
   external games, out of fold.

    .venv/bin/python solution/possession_ml.py cv          # leave-one-game-out on the external games
    .venv/bin/python solution/possession_ml.py train       # final model -> data/derived/possession_ml.pkl
    .venv/bin/python solution/possession_ml.py predict     # BAR-ATH -> evaluation/results/v2ml_possession*.csv
"""

import argparse
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE, os.path.join(HERE, "external"), os.path.join(ROOT, "evaluation"),
          os.path.join(ROOT, "starter")):
    sys.path.insert(0, p)
import possession_v1 as v1  # noqa: E402
import possession_v2 as v2  # noqa: E402

FPS = 25
H, A, D, X = 0, 1, 2, 3
MODEL = os.path.join(ROOT, "data", "derived", "possession_ml.pkl")
SWITCH_COST = 16.0         # log-likelihood per change of possession. Out of fold (cv, 6/11):
                           # IDSSE best at 4-8 (0.453), Metrica at 64 (0.678); 16 maximises the
                           # sum of the two (0.438 + 0.659): BAR-ATH's possessions (3.4 per
                           # minute in the public minutes) sit between IDSSE's (5.3) and
                           # Metrica's (about 2.8)
TRAIN_EVERY = 3            # frames between training rows (neighbours are nearly identical)
WINDOWS = (12, 50)         # frames: centred windows of 0.5 s and 2 s


# ---- features ---------------------------------------------------------------------------------
def _nanmean_window(x, half):
    """Centred moving mean of x over +-half frames, ignoring NaN (NaN if the window is empty)."""
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
    """Mean of x over frames t+start .. t+stop-1 (NaN-aware)."""
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


def attack_sign(teams, players, blocks, n):
    """+1 where the home team attacks towards x = L, -1 where towards x = 0 (per half)."""
    s = np.ones(n)
    for _, a, z in blocks:
        mx = np.nanmean(players[a:z + 1][:, teams == H, 0])
        s[a:z + 1] = 1.0 if mx < np.nanmean(players[a:z + 1][:, :, 0]) else -1.0
    return s


def features(teams, players, ball, ball_speed, blocks, pitch, p=None):
    """(n, F) float32 feature matrix and the feature names. Team columns are (home, away)."""
    p = p or v2.PARAMS
    L, W = pitch
    n = len(ball)
    vis = ~np.isnan(ball[:, 0])
    vel, speed, dv, _ = v1.ball_kinematics(ball, ball_speed, p)
    dist, _ = v1.team_distances(players, ball, teams)
    ctrl, _ = v1.control(dist, speed, vis, p)
    tt, _ = v1.touches(dv, dist, p)
    sgn = attack_sign(teams, players, blocks, n)
    bx = np.where(sgn > 0, ball[:, 0], L - ball[:, 0])          # towards the home attack
    by = np.abs(ball[:, 1] - W / 2)
    bvx = vel[:, 0] * sgn
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(vis, idx, -10 ** 9))
    nxt = np.minimum.accumulate(np.where(vis, idx, 10 ** 9)[::-1])[::-1]
    since = np.clip((idx - last) / FPS, 0, 60)
    until = np.clip((nxt - idx) / FPS, 0, 60)
    cols, names = [], []

    def add(name, x):
        cols.append(np.asarray(x, np.float32))
        names.append(name)

    add("ball_seen", vis)
    add("ball_speed", speed)
    add("ball_x_home_attack", bx)
    add("ball_y_off_centre", by)
    add("ball_vx_home_attack", bvx)
    add("since_seen_s", since)
    add("until_seen_s", until)
    seen = ~np.isnan(players[:, :, 0])
    pvel = np.linalg.norm(np.gradient(players, axis=0), axis=2) * FPS
    team_feats = {}
    for k, tag in ((H, "home"), (A, "away")):
        m = teams == k
        d = np.linalg.norm(players[:, m] - ball[:, None], axis=2)
        d = np.where(np.isnan(d), np.inf, d)
        ds = np.sort(d, axis=1)
        d1 = np.where(np.isfinite(ds[:, 0]), ds[:, 0], np.nan)
        d2 = np.where(np.isfinite(ds[:, 1]), ds[:, 1], np.nan) if ds.shape[1] > 1 else d1
        n5 = np.where(vis, np.sum(d < 5, axis=1), np.nan)
        n10 = np.where(vis, np.sum(d < 10, axis=1), np.nan)
        approach = -np.gradient(np.nan_to_num(d1, nan=np.nan)) * FPS
        px = players[:, m, 0]
        px = np.where(sgn[:, None] > 0, px, L - px)
        with np.errstate(invalid="ignore"):
            cx = np.nanmean(px, axis=1)
            width = np.nanstd(players[:, m, 1], axis=1)
            length = np.nanstd(px, axis=1)
            pace = np.nanmean(np.where(seen[:, m], pvel[:, m], np.nan), axis=1)
        team_feats[k] = dict(d1=d1, d2=d2, n5=n5, n10=n10, approach=approach, ctrl=ctrl[k],
                             touch=(tt == k).astype(np.float32), cx_rel=cx - bx, width=width,
                             length=length, pace=pace)
    for key in team_feats[H]:
        add(key + "_home", team_feats[H][key])
        add(key + "_away", team_feats[A][key])
    # context: differences home - away of the key quantities, over windows
    diffs = {
        "d1": team_feats[A]["d1"] - team_feats[H]["d1"],           # > 0: home closer
        "ctrl": team_feats[H]["ctrl"] - team_feats[A]["ctrl"],
        "touch": team_feats[H]["touch"] - team_feats[A]["touch"],
        "width": team_feats[H]["width"] - team_feats[A]["width"],
        "cx": team_feats[H]["cx_rel"] - team_feats[A]["cx_rel"],
        "bvx": bvx,
    }
    for key, x in diffs.items():
        for w in WINDOWS:
            add("{}_c{}".format(key, w), _nanmean_window(x, w))
        add(key + "_past1s", _shifted_mean(x, -FPS, 0))
        add(key + "_next1s", _shifted_mean(x, 1, FPS + 1))
    return np.stack(cols, 1), names


def swap_teams(Xf, names):
    """The same rows seen with the teams exchanged: home <-> away columns, home attack flipped,
    differences negated."""
    Y = Xf.copy()
    idx = {nm: i for i, nm in enumerate(names)}
    for nm, i in idx.items():
        if nm.endswith("_home"):
            j = idx[nm[:-5] + "_away"]
            Y[:, i], Y[:, j] = Xf[:, j], Xf[:, i]
    Y[:, idx["ball_x_home_attack"]] = 105.0 - Xf[:, idx["ball_x_home_attack"]]
    Y[:, idx["ball_vx_home_attack"]] = -Xf[:, idx["ball_vx_home_attack"]]
    for nm, i in idx.items():
        if any(nm.startswith(k + "_") for k in ("d1", "ctrl", "touch", "width", "cx", "bvx")) and \
                not nm.endswith(("_home", "_away")):
            Y[:, i] = -Xf[:, i]
    # cx_rel is relative to the ball in the home-attack frame: flip both teams' values
    for side in ("_home", "_away"):
        i = idx["cx_rel" + side]
        Y[:, i] = -Y[:, i]
    return Y


# ---- model ------------------------------------------------------------------------------------
def fit(Xs, ys, names, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xa = np.concatenate([np.concatenate(Xs), swap_teams(np.concatenate(Xs), names)])
    ya = np.concatenate([np.concatenate(ys), 1 - np.concatenate(ys)])
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=63,
                                         min_samples_leaf=200, l2_regularization=1.0, random_state=seed)
    clf.fit(Xa, ya)
    return clf


def prob_home(clf, Xf, names):
    """P(home has the ball), averaged over the two team orderings (exactly symmetric)."""
    a = clf.predict_proba(Xf)[:, 1]
    b = 1.0 - clf.predict_proba(swap_teams(Xf, names))[:, 1]
    return (a + b) / 2.0


# ---- decoder ----------------------------------------------------------------------------------
def decode(p_home, v2_codes, switch_cost):
    """Inside every non-D, non-X stretch of v2, the Viterbi H/A path for emissions log P with a
    switch_cost per change. D and X frames are kept from v2."""
    out = v2_codes.copy()
    alive = np.isin(v2_codes, (H, A))
    e = np.flatnonzero(np.diff(np.concatenate(([0], alive.astype(np.int8), [0]))))
    eps = 1e-4
    lp = np.stack([np.log(np.clip(p_home, eps, 1 - eps)), np.log(np.clip(1 - p_home, eps, 1 - eps))], 1)
    for a, b in zip(e[::2], e[1::2]):
        seg = lp[a:b]
        m = len(seg)
        score = seg[0].copy()
        back = np.zeros((m, 2), np.int8)
        for t in range(1, m):
            stay = score
            swap = score[::-1] - switch_cost
            choose = swap > stay
            back[t] = np.where(choose, 1 - np.arange(2), np.arange(2))
            score = np.where(choose, swap, stay) + seg[t]
        s = int(np.argmax(score))
        path = np.empty(m, np.int8)
        for t in range(m - 1, -1, -1):
            path[t] = s
            s = back[t, s]
        out[a:b] = path
    return out


# ---- external games ---------------------------------------------------------------------------
def external_runs():
    import calibrate
    import idsse
    import metrica
    return [((g, 1), "tracab") for g in idsse.GAMES] + [((g, 1), None) for g in metrica.GAMES]


def _game_job(args):
    """Features, labels and v2's own labels for one external run (realistic tracking)."""
    import calibrate
    import idsse
    import possession_metrics as pm
    run, labels = args
    teams, players, ball, bs, blocks, gt = calibrate.case(run, "realistic", labels)
    pitch = calibrate.pitch_of(run)
    Xf, names = features(teams, players, ball, bs, blocks, pitch)
    p = dict(v2.PARAMS)
    if run[0] in idsse.GAMES:
        p["KICKOFF_MAX_S"] = 0
    lab, _, _ = v2.predict(teams, players, ball, bs, blocks, pitch, p)
    return run, Xf, names, gt, pm.label_codes(lab)


def load_external(workers=5):
    from multiprocessing import Pool
    with Pool(workers) as pool:
        return pool.map(_game_job, external_runs())


def cv(costs=(2, 4, 8, 16, 32, 64)):
    import possession_metrics as pm
    data = load_external()
    names = data[0][2]
    oof = {}
    for i, (run, Xf, _, gt, _) in enumerate(data):
        tr = [j for j in range(len(data)) if j != i]
        Xs, ys = [], []
        for j in tr:
            g = data[j][3]
            m = np.isin(g, (H, A))
            m[np.arange(len(m)) % TRAIN_EVERY != 0] = False
            Xs.append(data[j][1][m])
            ys.append((g[m] == H).astype(np.int8))
        clf = fit(Xs, ys, names)
        oof[i] = prob_home(clf, Xf, names)
        m = np.isin(gt, (H, A))
        acc = np.mean((oof[i][m] > 0.5) == (gt[m] == H))
        print("fold {} {}: frame accuracy on alive frames {:.3f}".format(i, run[0], acc), flush=True)
    print("\nspellF1 (mean over games; IDSSE tracab labels / Metrica) by switch cost:")
    for name, sel in (("IDSSE", [i for i, d in enumerate(data) if not isinstance(d[0][0], int)]),
                      ("Metrica", [i for i, d in enumerate(data) if isinstance(d[0][0], int)])):
        base = np.mean([pm.evaluate(data[i][3], data[i][4], bootstrap_n=0)["spell_f1_mean"] for i in sel])
        line = "  {:8s} v2 {:.3f} |".format(name, base)
        for c in costs:
            s = [pm.evaluate(data[i][3], decode(oof[i], data[i][4], c), bootstrap_n=0)["spell_f1_mean"] for i in sel]
            line += "  cost {:>3}: {:.3f}".format(c, np.mean(s))
        print(line, flush=True)
    return data, oof


def train():
    """Final model on all the external games, saved to MODEL."""
    data = load_external()
    names = data[0][2]
    Xs, ys = [], []
    for run, Xf, _, gt, _ in data:
        m = np.isin(gt, (H, A))
        m[np.arange(len(m)) % TRAIN_EVERY != 0] = False
        Xs.append(Xf[m])
        ys.append((gt[m] == H).astype(np.int8))
    clf = fit(Xs, ys, names)
    with open(MODEL, "wb") as f:
        pickle.dump({"clf": clf, "names": names, "switch_cost": SWITCH_COST,
                     "games": [str(r[0]) for r, *_ in data]}, f)
    print("trained on {} games, {} rows (x2 swapped) -> {}".format(
        len(data), sum(len(x) for x in Xs), os.path.relpath(MODEL, ROOT)))


# ---- three states: the dead ball learnt too ------------------------------------------------------
# The model above trusts possession_v2 for the dead ball. This one decides home / away / dead
# together, with v2's own answer as one of its inputs, plus what tells a stoppage apart when the
# ball is lost: the players' pace and braking, the layouts of a set piece (layout.py), how long the
# ball gap is and where the ball was last seen and is seen again. Same training data (other
# matches only, tracking dirtied like BAR-ATH's), same symmetry in the two teams. IDSSE matches are
# also used with the ball removed while a keeper holds it (calibrate "keeper"), in training only.
MODEL3 = os.path.join(ROOT, "data", "derived", "possession_ml3.pkl")
COST_TEAM = 16.0           # log-likelihood per change of possession, chosen by cv3()
COST_DEAD = 32.0           # per change between alive and dead
# With SkillCorner and the PFF World Cup in the training set (cv3, 10/11 night), spellF1 out of fold:
# IDSSE 0.543, Metrica 0.692, SkillCorner 0.736, PFF (10 matches held out) 0.832 against v2's
# 0.341 / 0.570 / 0.548 / 0.621. A cost sweep (team 4 to 24) is flat over the three older sets
# (IDSSE prefers 8, Metrica and SkillCorner 16-24), so 16 / 32 stays.
# First model (IDSSE + Metrica only):
# cv3 (5 folds, 10/11), spellF1 out of fold: IDSSE v2 0.341, v2's dead ball + learnt team 0.455,
# three states 0.558; Metrica 0.570 / 0.661 / 0.689. IDSSE is best at team 8 / dead 16 (0.566),
# Metrica at 32 / 32 (0.714); 16 / 32 maximises the sum, as SWITCH_COST does for the team alone.
# Frames right: 88-92% against v2's 72-87%.
TRAIN_EVERY3 = 5


def features3(teams, players, ball, ball_speed, blocks, pitch, codes, f, r):
    """features() plus the dead-ball inputs. codes, f, r: possession_v2.predict on the same arrays
    (codes = pm.label_codes(labels), f = features, r = result)."""
    import layout
    X1, names = features(teams, players, ball, ball_speed, blocks, pitch)
    L, W = pitch
    n = len(ball)
    vis = ~np.isnan(ball[:, 0])
    cols, extra = [], []

    def add(name, x):
        cols.append(np.asarray(x, np.float32))
        extra.append(name)

    add("v2_dead", codes == D)
    add("v2_owner_home", codes == H)
    add("v2_owner_away", codes == A)
    add("v2_dead_score", r["dead_score"])
    add("v2_alive_score", r["alive_score"])
    ps = f["pspeed"].astype(np.float64)
    add("pace", ps)
    add("pace_c50", _nanmean_window(ps, 50))
    add("pace_past4s", _shifted_mean(ps, -100, 0))
    add("pace_next4s", _shifted_mean(ps, 1, 101))
    add("braking", f["f_rhythm"])
    add("players_far", f["far"])
    add("ball_out_m", f["out_dist"])
    gl = np.minimum(ball[:, 0], L - ball[:, 0])
    tl = np.minimum(ball[:, 1], W - ball[:, 1])
    add("ball_goal_line_m", gl)
    add("ball_touchline_m", tl)
    lay = layout.signals(teams, players, blocks, pitch)
    anyl = np.zeros(n, dtype=bool)
    for k in layout.NAMES:
        add("layout_" + k, lay[k])
        anyl |= lay[k]
    add("layout_any_c50", _nanmean_window(anyl.astype(np.float64), 50))
    # the ball gap the frame is in: length, pace inside it, where the ball was and is again
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(vis, idx, -1))
    nxt = np.minimum.accumulate(np.where(vis, idx, n)[::-1])[::-1]
    ingap = ~vis & (last >= 0) & (nxt < n)
    glen = np.where(ingap, (nxt - last - 1) / FPS, 0.0)
    cps = np.concatenate(([0.0], np.cumsum(np.nan_to_num(ps))))
    cpn = np.concatenate(([0], np.cumsum(~np.isnan(ps))))
    lo, hi = np.clip(last + 1, 0, n), np.clip(nxt, 0, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        gpace = np.where(ingap, (cps[hi] - cps[lo]) / np.maximum(cpn[hi] - cpn[lo], 1), np.nan)
    add("gap_len_s", np.clip(glen, 0, 120))
    add("gap_pace", gpace)
    li, ni = np.clip(last, 0, n - 1), np.clip(nxt, 0, n - 1)
    for tag, j in (("last", li), ("next", ni)):
        add("gap_{}_goal_line_m".format(tag), np.where(ingap, gl[j], np.nan))
        add("gap_{}_touchline_m".format(tag), np.where(ingap, tl[j], np.nan))
        add("gap_{}_centre_m".format(tag), np.where(ingap, np.hypot(ball[j, 0] - L / 2, ball[j, 1] - W / 2), np.nan))
        add("gap_{}_off_centre_y".format(tag), np.where(ingap, np.abs(ball[j, 1] - W / 2), np.nan))
    # --- second set (N_FEATURES_V1 columns come before): the stoppage seen over a longer time, and
    # whether the ball near a player is contested. A ball carried to the spot during a stoppage has
    # one player on it, nobody of the other team near, and everybody walking for seconds.
    d1h, d1a = X1[:, names.index("d1_home")], X1[:, names.index("d1_away")]
    with np.errstate(invalid="ignore"):
        dnear, dother = np.fmin(d1h, d1a), np.fmax(d1h, d1a)
    add("ball_nearest_m", dnear)
    add("ball_other_team_m", dother)
    add("ball_other_team_c50", _nanmean_window(dother.astype(np.float64), 50))
    add("ball_other_team_c125", _nanmean_window(dother.astype(np.float64), 125))
    add("pace_past10s", _shifted_mean(ps, -250, 0))
    add("pace_next10s", _shifted_mean(ps, 1, 251))
    add("pace_c250", _nanmean_window(ps, 250))
    sp = X1[:, names.index("ball_speed")].astype(np.float64)
    add("ball_speed_c50", _nanmean_window(sp, 50))
    add("ball_speed_past3s", _shifted_mean(sp, -75, 0))
    add("ball_speed_next3s", _shifted_mean(sp, 1, 76))
    add("ball_seen_c125", _nanmean_window(vis.astype(np.float64), 125))
    vd = (codes == D).astype(np.float64)
    add("v2_dead_c125", _nanmean_window(vd, 125))
    add("v2_dead_past5s", _shifted_mean(vd, -125, 0))
    add("v2_dead_next5s", _shifted_mean(vd, 1, 126))
    for name, mask in (("out", np.nan_to_num(f["out_dist"], nan=0.0) > 1.0), ("layout", anyl),
                       ("v2_dead", codes == D), ("fast_ball", np.nan_to_num(sp, nan=0.0) > 8.0)):
        last_e = np.maximum.accumulate(np.where(mask, idx, -10 ** 9))
        next_e = np.minimum.accumulate(np.where(mask, idx, 10 ** 9)[::-1])[::-1]
        add("since_{}_s".format(name), np.clip((idx - last_e) / FPS, 0, 30))
        add("until_{}_s".format(name), np.clip((next_e - idx) / FPS, 0, 30))
    # --- third set (12/11): the ball gap seen from each keeper (keeper_signals), so that a ball
    # lost in the keeper's hands is not taken for a stoppage. Out of fold (spell_lab "kpfeat"):
    # hold frames called dead 12% -> 4% on SkillCorner and PFF, spellF1 unchanged on the mean.
    import keeper_signals
    K, kn = keeper_signals.features(teams, players, ball, blocks, pitch)
    # --- fourth set (13/11): set pieces read from the players alone (setpiece_signals: corner,
    # penalty, throw-in, wall, crowded box) and which team will restart, for the stoppages whose
    # ball is not tracked. Out of fold together with the "view" training condition (spell_lab
    # "kpspv"): mean spellF1 0.704 -> 0.705 realistic, 0.693 -> 0.699 on camera-framed tracking.
    import setpiece_signals
    S, sn = setpiece_signals.features(teams, players, ball, blocks, pitch)
    return np.concatenate([X1, np.stack(cols, 1), K, S], axis=1), names + extra + kn + sn


N_FEATURES_V1 = 82      # the columns of the first three-state model (10/11): features3()[:, :82]


def dead_metrics(gt, pred):
    """What the eye sees wrong in the dead ball. (dead recall, dead precision, possessions lying
    wholly inside a true stoppage per true stoppage, seconds of them per true stoppage, share of
    true stoppages with no predicted one, start lags s, end lags s)."""
    g = gt != X
    td, pdead = (gt == D) & g, (pred == D) & g
    rec = np.mean(pdead[td]) if td.any() else np.nan
    prec = np.mean(td[pdead]) if pdead.any() else np.nan

    def runs(m):
        e = np.flatnonzero(np.diff(np.concatenate(([0], m.astype(np.int8), [0]))))
        return list(zip(e[::2], e[1::2]))
    ts, ps_ = runs(td), runs(pdead)
    cut = np.flatnonzero(np.diff(pred.astype(np.int16)) != 0) + 1
    st = np.concatenate(([0], cut))
    en = np.concatenate((cut, [len(pred)]))
    fake = [(a, b) for a, b in zip(st, en) if pred[a] in (H, A) and td[a:b].all()]
    lag_s, lag_e, missed = [], [], 0
    pa = np.array([a for a, _ in ps_]) if ps_ else np.zeros(0, dtype=np.int64)
    pb = np.array([b for _, b in ps_]) if ps_ else np.zeros(0, dtype=np.int64)
    for a, b in ts:
        ov = np.minimum(pb, b) - np.maximum(pa, a) if len(pa) else np.zeros(0)
        if not len(ov) or ov.max() <= 0:
            missed += 1
            continue
        hit = np.flatnonzero(ov > 0)
        lag_s.append((pa[hit[0]] - a) / FPS)       # first predicted stoppage touching it
        lag_e.append((pb[hit[-1]] - b) / FPS)      # last one
    n = max(len(ts), 1)
    return (rec, prec, len(fake) / n, sum(b - a for a, b in fake) / FPS / n, missed / n, lag_s, lag_e)


def swap_teams3(Xf, names):
    Y = swap_teams(Xf, names)
    i, j = names.index("v2_owner_home"), names.index("v2_owner_away")
    Y[:, i], Y[:, j] = Xf[:, j], Xf[:, i]
    return Y


def fit3(Xs, ys, names, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    X, y = np.concatenate(Xs), np.concatenate(ys)
    ys_ = np.where(y == D, D, 1 - y)
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=63,
                                         min_samples_leaf=200, l2_regularization=1.0, random_state=seed)
    clf.fit(np.concatenate([X, swap_teams3(X, names)]), np.concatenate([y, ys_]))
    return clf


def prob3(clf, Xf, names):
    """(n, 3) P(home), P(away), P(dead), averaged over the two team orderings."""
    a = clf.predict_proba(Xf)
    b = clf.predict_proba(swap_teams3(Xf, names))[:, [1, 0, 2]]
    return (a + b) / 2.0


def decode3(p3, blocks, cost_team=COST_TEAM, cost_dead=COST_DEAD):
    """Most likely H / A / D sequence in each graded block (X elsewhere), with a cost per change
    of possession and a cost per change between alive and dead."""
    n = len(p3)
    out = np.full(n, X, dtype=np.int8)
    lp = np.log(np.clip(p3, 1e-4, 1.0))
    T = ((0.0, cost_team, cost_dead), (cost_team, 0.0, cost_dead), (cost_dead, cost_dead, 0.0))
    for _, a, z in blocks:
        e = lp[a:z + 1].tolist()
        sc = list(e[0])
        back = []
        for t in range(1, len(e)):
            new, bk = [0.0, 0.0, 0.0], [0, 0, 0]
            for j in range(3):
                best, arg = sc[j], j
                for i in range(3):
                    v = sc[i] - T[i][j]
                    if v > best:
                        best, arg = v, i
                new[j], bk[j] = best + e[t][j], arg
            sc = new
            back.append(bk)
        k = max(range(3), key=lambda j: sc[j])
        path = [k]
        for bk in reversed(back):
            k = bk[k]
            path.append(k)
        out[a:z + 1] = path[::-1]
    return out


def _game_job3(args):
    import calibrate
    import idsse
    import possession_metrics as pm
    import pff
    import skillcorner
    run, labels, cond = args[:3]
    stride = args[3] if len(args) > 3 else 0
    teams, players, ball, bs, blocks, gt = calibrate.case(run, cond, labels)
    pitch = calibrate.pitch_of(run)
    p = dict(v2.PARAMS)
    if run[0] in idsse.GAMES or run[0] in skillcorner.GAMES or run[0] in pff.GAMES:   # their halves start at the kick-off itself
        p["KICKOFF_MAX_S"] = 0
    lab, f, r = v2.predict(teams, players, ball, bs, blocks, pitch, p)
    codes = pm.label_codes(lab)
    X3, names = features3(teams, players, ball, bs, blocks, pitch, codes, f, r)
    # rows a model may learn from: graded, and with the players there (SkillCorner has no tracking
    # at all during replays and close-ups; a fixed tactical camera always has the players)
    usable = (gt != X) & ((~np.isnan(players[..., 0])).sum(axis=1) >= MIN_PLAYERS3)
    if stride:   # a training-only run: just its rows, one every `stride` (keeps the memory small)
        m = usable.copy()
        m[np.arange(len(m)) % stride != 0] = False
        return run, X3[m], names, gt[m], None, None, None
    return run, X3, names, gt, codes, blocks, usable


def load_external3(workers=5):
    """The training runs: every match in each condition of TRAIN3."""
    from multiprocessing import Pool
    with Pool(workers) as pool:
        return pool.map(_game_job3, _jobs3(TRAIN3))


def _rows3(d):
    gt = d[3]
    m = d[6].copy()
    m[np.arange(len(m)) % TRAIN_EVERY3 != 0] = False
    return d[1][m], gt[m].astype(np.int8)


TRAIN3 = (("realistic", 1), ("keeper", 2), ("moved", 3), ("view", 2))   # (condition, seed) of the training runs
# "view" (13/11): only what BAR-ATH's tactical camera would show (calibrate.case, degrade.camera_view)
MIN_PLAYERS3 = 10      # tracked players a frame needs to be a training row


def _jobs3(conds):
    """Every match in each condition. SkillCorner (broadcast tracking, used as it is) only under
    "realistic", "keeper" (its own ball with the keeper holds removed) and "view". Before 12/11 the keeper
    condition was IDSSE's only."""
    import idsse
    import metrica
    import skillcorner
    jobs = []
    for cond, seed in conds:
        jobs += [((g, seed), "tracab", cond) for g in idsse.GAMES]
        jobs += [((g, seed), None, cond) for g in metrica.GAMES]
        if cond in ("realistic", "keeper", "view"):
            jobs += [((g, seed), None, cond) for g in skillcorner.GAMES]
    return jobs


PFF_STRIDE = 15        # training rows of a PFF match: one frame in 15 (59 matches weigh like about 20)
PFF_CONDS = ("realistic", "moved", "keeper", "view")


def _pff_jobs(skip=(), stride=PFF_STRIDE):
    """Training-only jobs for the PFF World Cup matches on disk (rows only)."""
    import pff
    seed_of = dict(TRAIN3)
    return [((g, seed_of[c]), None, c, stride) for c in PFF_CONDS for g in pff.GAMES if g not in skip]


def _set_of(game):
    import idsse
    import pff
    import skillcorner
    if game in pff.GAMES:
        return "PFF"
    return "SkillCorner" if game in skillcorner.GAMES else ("IDSSE" if game in idsse.GAMES else "Metrica")


def cv3(folds=5, costs=((16, 32),), every=10, workers=5, pff_test=10, pff_stride=25):
    """Out of fold on the external matches, with and without the PFF World Cup matches in the
    training set. IDSSE (7), Metrica (3) and SkillCorner (20) are tested fold by fold, on realistic
    and (IDSSE, Metrica) on moved; pff_test PFF matches are held out of every training set and
    tested too."""
    import pff
    import possession_metrics as pm
    from multiprocessing import Pool
    held_pff = list(pff.GAMES[:pff_test])
    jobs = _jobs3(TRAIN3) + [j for j in _jobs3((("moved", 1),))] + [((g, 1), None, "realistic") for g in held_pff]
    with Pool(workers) as pool:
        data = pool.map(_game_job3, jobs)
        rows_pff = pool.map(_game_job3, _pff_jobs(skip=held_pff, stride=pff_stride)) if len(pff.GAMES) > pff_test else []
    names = data[0][2]
    key = {(j[0], j[2]): d for j, d in zip(jobs, data)}
    games = list(dict.fromkeys(j[0][0] for j in jobs))
    fold_of = {}
    for sname in ("IDSSE", "Metrica", "SkillCorner"):
        for i, g in enumerate([g for g in games if _set_of(g) == sname]):
            fold_of[g] = i % folds
    for g in held_pff:
        fold_of[g] = -1                   # never in a training set; tested with the model of fold 0
    tests = [(run, cond) for (run, cond) in key if run[1] == 1 and cond in ("realistic", "moved")]
    configs = (("without PFF", False), ("with PFF", True)) if rows_pff else (("without PFF", False),)
    res = {}
    for cname, use_pff in configs:
        for fo in range(folds):
            Xs, ys = [], []
            for (run, cond), d in key.items():
                if fold_of[run[0]] in (fo, -1) or (cond, run[1]) not in TRAIN3:
                    continue
                m = d[6].copy()
                m[np.arange(len(m)) % every != 0] = False
                Xs.append(d[1][m])
                ys.append(d[3][m].astype(np.int8))
            if use_pff:
                Xs += [d[1] for d in rows_pff]
                ys += [d[3].astype(np.int8) for d in rows_pff]
            clf = fit3(Xs, ys, names)
            for (run, cond) in tests:
                if fold_of[run[0]] == fo or (fold_of[run[0]] == -1 and fo == 0):
                    p3 = prob3(clf, key[(run, cond)][1], names)
                    for ct, cd in costs:
                        res[(cname, cond, run[0], ct, cd)] = decode3(p3, key[(run, cond)][5], ct, cd)
            print("{}: fold {} done ({} training rows)".format(cname, fo, sum(len(x) for x in Xs)), flush=True)
    for cond in ("realistic", "moved"):
        for sname in ("IDSSE", "Metrica", "SkillCorner", "PFF"):
            sel = [g for g in games if _set_of(g) == sname and ((g, 1), cond) in key]
            if not sel:
                continue
            print("\n== tested on {} tracking, {} ({} matches)".format(cond, sname, len(sel)))
            rows = [("v2", lambda g: key[((g, 1), cond)][4])]
            for cname, _ in configs:
                for ct, cd in costs:
                    rows.append(("{} ({}/{})".format(cname, ct, cd), lambda g, c=cname, ct=ct, cd=cd: res[(c, cond, g, ct, cd)]))
            for name, lab_of in rows:
                sp, M, acc = [], [], []
                for g in sel:
                    gt = key[((g, 1), cond)][3]
                    out = lab_of(g)
                    sp.append(pm.evaluate(gt, out, bootstrap_n=0)["spell_f1_mean"])
                    M.append(dead_metrics(gt, out))
                    acc.append(np.mean(out[gt != X] == gt[gt != X]))
                ls = np.concatenate([np.array(m[5]) for m in M])
                le = np.concatenate([np.array(m[6]) for m in M])
                print("  {:24s} spellF1 {:.3f} | frames right {:4.1f}% | dead ball: found {:4.1f}%, right {:4.1f}% | "
                      "fake possessions per stoppage {:.2f} | stoppages missed {:4.1f}% | "
                      "starts {:+.1f} s (median), within 1 s {:2.0f}%, mean |lag| {:.1f} s | ends {:+.1f} s, within 1 s {:2.0f}%".format(
                          name, np.mean(sp), 100 * np.mean(acc), 100 * np.mean([m[0] for m in M]),
                          100 * np.mean([m[1] for m in M]), np.mean([m[2] for m in M]),
                          100 * np.mean([m[4] for m in M]), np.median(ls), 100 * np.mean(np.abs(ls) <= 1),
                          np.mean(np.abs(ls)), np.median(le), 100 * np.mean(np.abs(le) <= 1)), flush=True)
    return key, res


def train3(workers=5):
    """Final model: every match in each condition of TRAIN3, plus the PFF World Cup matches
    (one frame in PFF_STRIDE, in each condition of PFF_CONDS)."""
    from multiprocessing import Pool
    with Pool(workers) as pool:
        data = pool.map(_game_job3, _jobs3(TRAIN3))
        rows_pff = pool.map(_game_job3, _pff_jobs())
    names = data[0][2]
    Xs, ys = zip(*[_rows3(d) for d in data])
    Xs, ys = list(Xs) + [d[1] for d in rows_pff], list(ys) + [d[3].astype(np.int8) for d in rows_pff]
    clf = fit3(Xs, ys, names)
    with open(MODEL3, "wb") as fh:
        pickle.dump({"clf": clf, "names": names, "cost_team": COST_TEAM, "cost_dead": COST_DEAD}, fh)
    print("three states: trained on {} runs + {} PFF runs, {} rows (x2 swapped) -> {}".format(
        len(data), len(rows_pff), sum(len(x) for x in Xs), os.path.relpath(MODEL3, ROOT)))


def predict3(teams, players, ball, bs, blocks, pitch, events=(), level=None, cut=None, model=None, return_p3=False):
    """(codes H/A/D/X of the three-state model, v2's codes[, P(H), P(A), P(D)]) on any tracking."""
    import possession_metrics as pm
    if model is None:
        with open(MODEL3, "rb") as fh:
            model = pickle.load(fh)
    lab, f, r = v2.predict(teams, players, ball, bs, blocks, pitch, v2.PARAMS, events, level, cut)
    codes = pm.label_codes(lab)
    X3, names = features3(teams, players, ball, bs, blocks, pitch, codes, f, r)
    assert names == model["names"]
    p3 = prob3(model["clf"], X3, names)
    out = decode3(p3, blocks, model["cost_team"], model["cost_dead"])
    return (out, codes, p3) if return_p3 else (out, codes)


def predict3_bar_ath(source="tactical", use_whistle=True):
    """BAR-ATH with the three-state model. Writes <stem>ml3_possession.csv and the model's
    probabilities to data/derived/<stem>ml3_probs.npz (for the review video)."""
    import sync_broadcast
    import tracking
    import whistle
    from baseline import write_submission
    from games import GAME, GAMES, half_video_blocks
    teams, players, ball, bs, _ = tracking.load(GAME, source)
    blocks = half_video_blocks(GAME)
    events, level = whistle.load(GAME) if use_whistle else ((), None)
    cut = None
    if source == "fused":
        sync = np.load(sync_broadcast.OUT)
        cut = (sync["b_of_t"] >= 0) & (sync["b_live"] < 0)
    out, codes, p3 = predict3(teams, players, ball, bs, blocks, GAMES[GAME]["pitch"], events, level, cut, return_p3=True)
    np.savez_compressed(os.path.join(ROOT, "data", "derived", "{}ml3_probs.npz".format(v2.STEMS[source])), p3=p3.astype(np.float32))
    labels = np.array(["H", "A", "D", "X"])[out]
    path = os.path.join(ROOT, "evaluation", "results", "{}ml3_possession{}.csv".format(
        v2.STEMS[source], "" if use_whistle else "_nowhistle"))
    write_submission(path, labels)
    for half, a, z in blocks:
        seg = out[a:z + 1]
        starts = np.concatenate(([0], np.flatnonzero(seg[1:] != seg[:-1]) + 1))
        print("half {}: H {:.1f}%  A {:.1f}%  D {:.1f}%  {} possessions   (v2: D {:.1f}%)".format(
            half, 100 * np.mean(seg == H), 100 * np.mean(seg == A), 100 * np.mean(seg == D),
            int(np.sum(seg[starts] < 2)), 100 * np.mean(codes[a:z + 1] == D)))
    g = codes != X
    print("frames equal to v2: {:.1f}%  -> {}".format(100 * np.mean(out[g] == codes[g]), os.path.relpath(path, ROOT)))


def predict_bar_ath(source="tactical", use_whistle=True):
    """BAR-ATH: v2 for the dead ball, the trained model for the team. Writes v2ml_* (v2fml_* fused)."""
    import possession_metrics as pm
    import sync_broadcast
    import tracking
    import whistle
    from baseline import write_submission
    from games import GAME, GAMES, half_video_blocks
    with open(MODEL, "rb") as f:
        M = pickle.load(f)
    teams, players, ball, bs, names = tracking.load(GAME, source)
    blocks = half_video_blocks(GAME)
    pitch = GAMES[GAME]["pitch"]
    events, level = whistle.load(GAME) if use_whistle else ((), None)
    cut = None
    if source == "fused":
        sync = np.load(sync_broadcast.OUT)
        cut = (sync["b_of_t"] >= 0) & (sync["b_live"] < 0)
    lab, _, _ = v2.predict(teams, players, ball, bs, blocks, pitch, v2.PARAMS, events, level, cut)
    codes = pm.label_codes(lab)
    Xf, fn = features(teams, players, ball, bs, blocks, pitch)
    assert fn == M["names"]
    ph = prob_home(M["clf"], Xf, fn)
    out = decode(ph, codes, M["switch_cost"])
    labels = np.array(["H", "A", "D", "X"])[out]
    stem = v2.STEMS[source] + "ml"
    path = os.path.join(ROOT, "evaluation", "results", "{}_possession{}.csv".format(
        stem, "" if use_whistle else "_nowhistle"))
    write_submission(path, labels)
    for half, a, z in blocks:
        seg = labels[a:z + 1]
        cutp = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        starts = np.concatenate(([0], cutp))
        poss = sum(seg[k] in "HA" for k in starts)
        print("half {}: H {:.1f}%  A {:.1f}%  D {:.1f}%  {} possessions (v2: {})".format(
            half, 100 * np.mean(seg == "H"), 100 * np.mean(seg == "A"), 100 * np.mean(seg == "D"), poss,
            sum(lab[a:z + 1][k] in "HA" for k in np.concatenate(([0], np.flatnonzero(lab[a:z + 1][1:] != lab[a:z + 1][:-1]) + 1)))))
    agree = np.mean(out[codes != X] == codes[codes != X])
    print("frames equal to v2: {:.1f}%  -> {}".format(100 * agree, os.path.relpath(path, ROOT)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("cv", "train", "predict", "cv3", "train3", "predict3"))
    ap.add_argument("--tracking", choices=tuple(v2.STEMS), default="tactical")
    ap.add_argument("--no-whistle", action="store_true")
    a = ap.parse_args()
    if a.cmd == "cv3":
        cv3()
    elif a.cmd == "train3":
        train3()
    elif a.cmd == "predict3":
        predict3_bar_ath(a.tracking, not a.no_whistle)
    elif a.cmd == "cv":
        cv()
    elif a.cmd == "train":
        train()
    else:
        predict_bar_ath(a.tracking, not a.no_whistle)


if __name__ == "__main__":
    main()
