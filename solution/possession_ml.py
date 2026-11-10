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
    stem = "v2ml" if source == "tactical" else "v2fml"
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
    ap.add_argument("cmd", choices=("cv", "train", "predict"))
    ap.add_argument("--tracking", choices=("tactical", "fused"), default="tactical")
    ap.add_argument("--no-whistle", action="store_true")
    a = ap.parse_args()
    if a.cmd == "cv":
        cv()
    elif a.cmd == "train":
        train()
    else:
        predict_bar_ath(a.tracking, not a.no_whistle)


if __name__ == "__main__":
    main()
