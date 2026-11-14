"""Experiments on the possession spells, on top of analisi/spell_lab.py (out-of-fold probabilities of
the three-state model on 89 external matches). Never touches BAR-ATH.

Protocol
  - models (first stage, second stage, boundary detectors) are always trained out of fold: the 5
    match folds of spell_lab.fold_of, every variant of a match in the same fold;
  - decoder parameters (costs, weights) are CHOSEN on half of the matches and SCORED on the other
    half, then the halves swap: every reported number comes from matches that took no part in
    choosing it. Halves are drawn inside each dataset, so both have some of each provider;
  - the choice maximises the mean over the four datasets of the mean spellF1 (each dataset weighs
    the same, whatever its number of matches), on the realistic runs.

    .venv/bin/python analisi/spell_exp.py costs [tag]       # decoder costs, symmetric vs asymmetric
    .venv/bin/python analisi/spell_exp.py stage2 [tag]      # MS-TCN-like second stage on the probabilities
    .venv/bin/python analisi/spell_exp.py bounds [tag]      # ASRF-like boundary detectors -> per-frame costs
"""

import itertools
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "solution"), os.path.join(ROOT, "solution", "external"),
          os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "starter"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import possession_metrics as pm  # noqa: E402
import spell_lab as L  # noqa: E402
import viterbi  # noqa: E402

H, A, D, X = 0, 1, 2, 3
SETS = ("DFL", "Metrica", "SkillCorner", "PFF")
FPS = 25


# ---- scoring -------------------------------------------------------------------------------------
def quick_scores(gt, pred):
    """(spellF1, macroF1) as the official scorer computes them, without the extras."""
    runs_ = pm.valid_runs(gt)
    sp = pm.spell_scores(gt, pm.fill_abstentions(pred, runs_), runs_)["mean_f1"]
    cm = pm.confusion(gt, pred)
    return sp, pm.macro_f1(pm.scored_confusion(cm))


def half_of():
    """{game: 0 / 1}: alternate matches inside each dataset."""
    out = {}
    for s in SETS:
        gs = sorted({r[0] for r in L.runs() if L.set_of(r[0]) == s}, key=str)
        for i, g in enumerate(gs):
            out[g] = i % 2
    return out


def test_keys(oofp, cond="realistic1"):
    return [k for k in oofp if k.endswith("_" + cond)]


# ---- generic grid over decoders --------------------------------------------------------------------
_G = {}


def _init(values):
    """Pool initializer: macOS starts workers with spawn, so module globals must be passed in."""
    _G.update(values)


def _grid_job(args):
    k, p3, extra = args
    r = L._run_of_key(k)
    z = L.load(r)
    gt = np.asarray(z["gt"])
    blocks = L.blocks_of(z)
    out = []
    for cfg in _G["configs"]:
        pred = _G["fn"](p3, blocks, cfg, extra, z)
        out.append(quick_scores(gt, pred))
    return k, np.array(out)


def grid(oofp, fn, configs, keys, extras=None, workers=6):
    """{key: (n_configs, 2)} spellF1 / macroF1 of fn(p3, blocks, cfg, extra, z) for every config."""
    jobs = [(k, oofp[k], None if extras is None else extras.get(k)) for k in keys]
    with Pool(workers, initializer=_init, initargs=({"fn": fn, "configs": configs},)) as pool:
        return dict(pool.map(_grid_job, jobs))


def choose_and_score(res, configs, name=""):
    """Cross-halves: choose on one half, score on the other. Returns ({key: (spellF1, macroF1)} of the
    held-out choice, [chosen config on half 0, on half 1]) and prints per dataset."""
    ho = half_of()
    chosen, out = [], {}
    for h in (0, 1):
        tune = [k for k in res if ho[L._run_of_key(k)[0]] == h]
        per_set = []
        for s in SETS:
            ks = [k for k in tune if k.startswith(s + "_")]
            if ks:
                per_set.append(np.mean([res[k][:, 0] for k in ks], axis=0))
        best = int(np.argmax(np.mean(per_set, axis=0)))
        chosen.append(configs[best])
        for k in res:
            if ho[L._run_of_key(k)[0]] != h:
                out[k] = res[k][best]
    return out, chosen


def summary(scores_by_variant, title):
    """scores_by_variant: [(name, {key: (spellF1, macroF1)})] -> table per dataset."""
    print("\n== " + title)
    head = "{:42s}".format("variant") + "".join("{:>18s}".format(s) for s in SETS) + "{:>10s}".format("mean")
    print(head)
    for name, sc in scores_by_variant:
        line = "{:42s}".format(name)
        means = []
        for s in SETS:
            v = [sc[k] for k in sc if k.startswith(s + "_")]
            if v:
                m = np.mean(v, axis=0)
                means.append(m[0])
                line += "{:>11.3f} ({:.3f})".format(m[0], m[1])
            else:
                line += "{:>18s}".format("-")
        line += "{:>10.3f}".format(np.mean(means)) if means else ""
        print(line, flush=True)
    print("  spellF1 (macroF1) per dataset, mean of the matches; 'mean' = mean of the four datasets")


# ---- experiment 1: decoder costs -------------------------------------------------------------------
def _dec_costs(p3, blocks, cfg, extra, z):
    ct, ci, co = cfg
    return viterbi.decode(p3, blocks, ct, ci, co)


def exp_costs(tag="base"):
    o = L.load_oof(tag)
    keys = test_keys(o)
    sym = [(ct, c, c) for ct in (8, 12, 16, 24) for c in (16, 24, 32, 48, 64)]
    asym = [(ct, ci, co) for ct in (8, 12, 16, 24) for ci in (16, 24, 32, 48, 64) for co in (16, 24, 32, 48, 64)]
    res = grid(o, _dec_costs, asym, keys)
    idx_sym = [asym.index(c) for c in sym]
    fixed = {k: v[asym.index((16, 32, 32))] for k, v in res.items()}
    s_sym, ch_sym = choose_and_score({k: v[idx_sym] for k, v in res.items()}, sym)
    s_asym, ch_asym = choose_and_score(res, asym)
    summary([("decoder 16 / 32 (current)", fixed), ("symmetric costs, chosen", s_sym), ("asymmetric in / out, chosen", s_asym)],
            "decoder costs (model '{}'), chosen on the other half".format(tag))
    print("  chosen (team, in, out) on half 0 / half 1: symmetric {} | asymmetric {}".format(ch_sym, ch_asym))
    return res, asym


# ---- experiment 2: second stage on the probabilities (MS-TCN-like) ---------------------------------
LAGS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
WINS = (12, 25, 50, 125, 250)


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def prob_features(p3, blocks):
    """(n, k) features of a probability sequence, block by block (nothing crosses half-time).
    Team-antisymmetric columns (home minus away) are listed in the second return value."""
    n = len(p3)
    ld = _logit(p3[:, 2])
    lt = np.log(np.clip(p3[:, 0], 1e-4, 1)) - np.log(np.clip(p3[:, 1], 1e-4, 1))
    cols, anti, names = [], [], []

    def add(nm, v, a=False):
        cols.append(v.astype(np.float32))
        names.append(nm)
        anti.append(a)

    add("ld", ld)
    add("lt", lt, True)
    for base, v, a in (("ld", ld, False), ("lt", lt, True)):
        for lag in LAGS:
            for sgn in (-1, 1):
                sh = np.full(n, np.nan)
                for _, s0, z0 in blocks:
                    seg = v[s0:z0 + 1]
                    m = len(seg)
                    if sgn > 0 and lag < m:
                        sh[s0:z0 + 1 - lag] = seg[lag:]
                    elif sgn < 0 and lag < m:
                        sh[s0 + lag:z0 + 1] = seg[:-lag]
                add("{}_{}{}".format(base, "p" if sgn > 0 else "m", lag), sh, a)
        for w in WINS:
            past = np.full(n, np.nan)
            nxt = np.full(n, np.nan)
            for _, s0, z0 in blocks:
                seg = v[s0:z0 + 1].astype(np.float64)
                c = np.concatenate(([0.0], np.cumsum(seg)))
                i = np.arange(len(seg))
                lo, hi = np.maximum(i - w, 0), np.minimum(i + w + 1, len(seg))
                with np.errstate(invalid="ignore", divide="ignore"):
                    past[s0:z0 + 1] = (c[i] - c[lo]) / np.maximum(i - lo, 1)
                    nxt[s0:z0 + 1] = (c[hi] - c[i + 1]) / np.maximum(hi - i - 1, 1)
            add("{}_past{}".format(base, w), past, a)
            add("{}_next{}".format(base, w), nxt, a)
    return np.stack(cols, 1), np.array(anti), names


def _fit_stage(X, y, anti, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xs = X.copy()
    Xs[:, anti] = -Xs[:, anti]
    ys = np.where(y == D, D, 1 - y)
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=200,
                                         l2_regularization=1.0, random_state=seed)
    clf.fit(np.concatenate([X, Xs]), np.concatenate([y, ys]))
    return clf


def _predict_stage(clf, X, anti):
    Xs = X.copy()
    Xs[:, anti] = -Xs[:, anti]
    a = clf.predict_proba(X)
    b = clf.predict_proba(Xs)[:, [1, 0, 2]]
    return ((a + b) / 2).astype(np.float32)


def _feat_job(k):
    if "oof" not in _G:
        _G["oof"] = L.load_oof(_G["oof_tag"])
    p3 = _G["oof"][k]
    z = L.load(L._run_of_key(k))
    F, anti, names = prob_features(p3, L.blocks_of(z))
    extra = _G.get("raw")
    if extra:
        base_names = [str(x) for x in z["names"]]
        F = np.concatenate([F, np.asarray(z["X"])[:, [base_names.index(c) for c in extra]]], axis=1)
        anti = np.concatenate([anti, np.zeros(len(extra), bool)])
    return k, F, anti


def stage2(tag="base", out_tag=None, raw=None, every=10):
    """Second stage: a classifier on windows of the first stage's out-of-fold probabilities, itself
    trained out of fold (same folds). raw: optional list of first-stage feature names added to it."""
    o = L.load_oof(tag)
    keys = list(o)
    with Pool(5, initializer=_init, initargs=({"oof_tag": tag, "raw": raw},)) as pool:
        feats = {k: (F, anti) for k, F, anti in pool.map(_feat_job, keys)}
    fo = L.fold_of()
    out = {}
    for f in range(L.FOLDS):
        Xs, ys = [], []
        for k in keys:
            r = L._run_of_key(k)
            if fo[r[0]] == f:
                continue
            z = L.load(r)
            m = np.asarray(z["usable"]).copy()
            m[np.arange(len(m)) % every != 0] = False
            Xs.append(feats[k][0][m])
            ys.append(np.asarray(z["gt"])[m].astype(np.int8))
        anti = feats[keys[0]][1]
        clf = _fit_stage(np.concatenate(Xs), np.concatenate(ys), anti)
        for k in keys:
            if fo[L._run_of_key(k)[0]] == f:
                out[k] = _predict_stage(clf, feats[k][0], anti)
        print("stage 2 fold {} done".format(f), flush=True)
    out_tag = out_tag or (tag + "_s2" + ("raw" if raw else ""))
    np.savez(os.path.join(L.LAB, "oof_{}.npz".format(out_tag)), **out)
    return out_tag


# ---- experiment 3: boundary detectors (ASRF-like) -> per-frame transition costs ---------------------
BOUND_K = 12           # frames: a boundary label covers +-0.5 s around the true change


def boundary_targets(gt, k=BOUND_K):
    """(n, 3) 0/1: within k frames of a true alive -> dead change, a dead -> alive change, a team change."""
    n = len(gt)
    out = np.zeros((n, 3), dtype=np.int8)
    for t in np.flatnonzero(gt[1:] != gt[:-1]) + 1:
        a, b = gt[t - 1], gt[t]
        if a == X or b == X:
            continue
        j = 0 if (a < 2 and b == D) else (1 if (a == D and b < 2) else 2)
        out[max(0, t - k):min(n, t + k + 1), j] = 1
    return out


def bounds(tag="base", raw=None, every=5, out_tag=None):
    """Three binary detectors (dead ball starts, play restarts, possession changes) on the windowed
    probabilities (+ raw features), trained out of fold. Saves LAB/bounds_<tag>.npz: (n, 3) per run."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    o = L.load_oof(tag)
    keys = list(o)
    with Pool(5, initializer=_init, initargs=({"oof_tag": tag, "raw": raw},)) as pool:
        feats = {k: F for k, F, _ in pool.map(_feat_job, keys)}
    fo = L.fold_of()
    out = {k: np.zeros((len(o[k]), 3), np.float32) for k in keys}
    for f in range(L.FOLDS):
        Xs, Ys = [], []
        for k in keys:
            r = L._run_of_key(k)
            if fo[r[0]] == f:
                continue
            z = L.load(r)
            gt = np.asarray(z["gt"])
            m = np.asarray(z["usable"]).copy() & (gt != X)
            Y = boundary_targets(gt)
            sel = m & ((np.arange(len(m)) % every == 0) | (Y.max(1) == 1))   # keep every positive
            Xs.append(feats[k][sel])
            Ys.append(Y[sel])
        Xa, Ya = np.concatenate(Xs), np.concatenate(Ys)
        for j in range(3):
            pos = Ya[:, j].mean()
            w = np.where(Ya[:, j] == 1, min(20.0, (1 - pos) / max(pos, 1e-6)), 1.0)
            clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=200,
                                                 l2_regularization=1.0, random_state=j)
            clf.fit(Xa, Ya[:, j], sample_weight=w)
            for k in keys:
                if fo[L._run_of_key(k)[0]] == f:
                    out[k][:, j] = clf.predict_proba(feats[k])[:, 1]
        print("bounds fold {} done".format(f), flush=True)
    out_tag = out_tag or (tag + "_b" + ("raw" if raw else ""))
    np.savez(os.path.join(L.LAB, "bounds_{}.npz".format(out_tag)), **out)
    return out_tag


def _dec_bounds(p3, blocks, cfg, extra, z):
    ct, ci, co, w_in, w_out, w_team = cfg
    q = extra
    return viterbi.decode(p3, blocks, ct * np.maximum(0.05, 1 - w_team * q[:, 2]), ci * np.maximum(0.05, 1 - w_in * q[:, 0]),
                          co * np.maximum(0.05, 1 - w_out * q[:, 1]))


def exp_bounds(tag="base", btag=None):
    o = L.load_oof(tag)
    b = dict(np.load(os.path.join(L.LAB, "bounds_{}.npz".format(btag or tag + "_b"))))
    keys = test_keys(o)
    cfgs = [(ct, ci, co, wi, wo, wt) for ct in (12, 16, 24) for ci in (24, 32, 48) for co in (24, 32, 48)
            for wi in (0.0, 0.5, 0.9) for wo in (0.0, 0.5, 0.9) for wt in (0.0, 0.5, 0.9)]
    res = grid(o, _dec_bounds, cfgs, keys, extras={k: b[k] for k in keys})
    nob = [i for i, c in enumerate(cfgs) if c[3:] == (0.0, 0.0, 0.0)]
    s0, ch0 = choose_and_score({k: v[nob] for k, v in res.items()}, [cfgs[i] for i in nob])
    s1, ch1 = choose_and_score(res, cfgs)
    fixed = {k: v[cfgs.index((16, 32, 32, 0.0, 0.0, 0.0))] for k, v in res.items()}
    summary([("decoder 16 / 32 (current)", fixed), ("costs chosen, no boundary detector", s0), ("costs + boundary detector, chosen", s1)],
            "boundary detectors (model '{}')".format(tag))
    print("  chosen (team, in, out, w_in, w_out, w_team) half 0 / 1: without {} | with {}".format(ch0, ch1))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "costs"
    tag = sys.argv[2] if len(sys.argv) > 2 else "base"
    if cmd == "costs":
        exp_costs(tag)
    elif cmd == "stage2":
        t2 = stage2(tag)
        exp_costs(t2)
    elif cmd == "bounds":
        bt = bounds(tag)
        exp_bounds(tag, bt)
