"""Laboratory for the possession spells: out-of-fold H/A/D probabilities of the three-state model on
every external match, cached, so that decoders, correctors and new signals can be compared fast and
without ever touching BAR-ATH.

Data (all external, labelled by the providers): IDSSE / DFL 7, Metrica 3, SkillCorner 20, PFF World
Cup 2022 59. Groups are MATCHES: every variant of a match (realistic, keeper, moved) is in the same
fold, so no model ever sees another version of the match it predicts.

    .venv/bin/python analisi/spell_lab.py cache        # features of every run -> data/derived/lab/ (once)
    .venv/bin/python analisi/spell_lab.py oof [tag]    # 5-fold out-of-fold probabilities -> data/derived/lab/oof_<tag>.npz
    .venv/bin/python analisi/spell_lab.py errors [tag] # error taxonomy of the decoded labels, per dataset

Runs: for each match the "test" runs are realistic seed 1 (SkillCorner and PFF: their own tracking)
and, for IDSSE / Metrica / PFF, moved seed 1 (the ball carried to the restart spot during the
stoppages). Training uses the TRAIN3 conditions of possession_ml plus PFF realistic and moved.
"""

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

FPS = 25
H, A, D, X = 0, 1, 2, 3
LAB = os.path.join(ROOT, "data", "derived", "lab")
FOLDS = 5
EVERY = 10          # training rows: one frame in EVERY (PFF: PFF_EVERY)
PFF_EVERY = 25


def set_of(game):
    import idsse
    import metrica
    import pff
    import skillcorner
    if game in idsse.GAMES:
        return "DFL"
    if game in metrica.GAMES:
        return "Metrica"
    if game in skillcorner.GAMES:
        return "SkillCorner"
    if game in pff.GAMES:
        return "PFF"
    raise KeyError(game)


def runs():
    """[(game, labels, condition, seed, role)] role: 'test' runs are predicted and scored, 'train'
    runs are only learnt from."""
    import idsse
    import metrica
    import pff
    import skillcorner
    out = []
    for g in idsse.GAMES:
        out += [(g, "tracab", "realistic", 1, "test"), (g, "tracab", "moved", 1, "test"),
                (g, "tracab", "keeper", 2, "train"), (g, "tracab", "moved", 3, "train")]
    for g in metrica.GAMES:
        out += [(g, None, "realistic", 1, "test"), (g, None, "moved", 1, "test"), (g, None, "moved", 3, "train")]
    for g in skillcorner.GAMES:
        out += [(g, None, "realistic", 1, "test")]
    for g in pff.GAMES:
        out += [(g, None, "realistic", 1, "test"), (g, None, "moved", 1, "test")]
    if KEEPER_RUNS:
        # the ball removed while a goalkeeper holds it (11/11): a test version of every match, and a
        # training version of the ones that did not have it (DFL had keeper seed 2 already)
        for g in idsse.GAMES:
            out += [(g, "tracab", "keeper", 1, "test")]
        for g in list(metrica.GAMES) + list(skillcorner.GAMES) + list(pff.GAMES):
            out += [(g, None, "keeper", 1, "test"), (g, None, "keeper", 2, "train")]
    if VIEW_RUNS:
        # only what BAR-ATH's tactical camera would show (calibrate "view", 12/11): a test and a
        # training version of every match
        for g in idsse.GAMES:
            out += [(g, "tracab", "view", 1, "test"), (g, "tracab", "view", 2, "train")]
        for g in list(metrica.GAMES) + list(skillcorner.GAMES) + list(pff.GAMES):
            out += [(g, None, "view", 1, "test"), (g, None, "view", 2, "train")]
    return out


KEEPER_RUNS = True
VIEW_RUNS = os.environ.get("MTAH_VIEW") == "1"     # off by default: the runs before 12/11


def original_training(r):
    """The training runs of the models before 11/11 (keeper versions only for DFL, seed 2)."""
    return not (r[2] == "keeper" and (r[3] == 1 or set_of(r[0]) != "DFL"))


def key(r):
    return "{}_{}_{}{}".format(set_of(r[0]), r[0], r[2], r[3])


def path(r):
    return os.path.join(LAB, key(r) + ".npz")


def _cache_job(r):
    import possession_ml as ml
    out = path(r)
    if os.path.exists(out):
        return key(r)
    game, labels, cond, seed, _ = r
    run, X3, names, gt, codes, blocks, usable = ml._game_job3(((game, seed), labels, cond))
    if any(nm.startswith("kp_") for nm in names):   # features3 grew the keeper block (12/11): the lab
        cut = next(i for i, nm in enumerate(names) if nm.startswith("kp_"))   # keeps it in "kp"
        X3, names = X3[:, :cut], names[:cut]
    tmp = out + ".tmp.npz"
    np.savez(tmp, X=X3.astype(np.float32), names=np.array(names), gt=gt.astype(np.int8), codes=codes.astype(np.int8),
             usable=usable, blocks=np.array([b[1:] for b in blocks], dtype=np.int64))
    os.replace(tmp, out)
    return key(r)


def cache(workers=5):
    os.makedirs(LAB, exist_ok=True)
    todo = [r for r in runs() if not os.path.exists(path(r))]
    print("{} runs to build, {} already cached".format(len(todo), len(runs()) - len(todo)), flush=True)
    with Pool(workers, maxtasksperchild=4) as pool:
        for i, k in enumerate(pool.imap_unordered(_cache_job, todo), 1):
            if i % 10 == 0 or i == len(todo):
                print("  {} / {} ({})".format(i, len(todo), k), flush=True)


def col_path(r, block="col"):
    return os.path.join(LAB, key(r) + "_{}.npz".format(block))


def _kp_job(r):
    import calibrate
    import keeper_signals
    import warnings
    warnings.filterwarnings("ignore")
    out = col_path(r, "kp")
    if os.path.exists(out):
        return key(r)
    game, labels, cond, seed, _ = r
    teams, players, ball, bs, blocks, gt = calibrate.case((game, seed), cond, labels)
    F, names = keeper_signals.features(teams, players, ball, blocks, calibrate.pitch_of((game, seed)))
    tmp = out + ".tmp.npz"
    np.savez(tmp, X=F, names=np.array(names))
    os.replace(tmp, out)
    return key(r)


def cache_kp(workers=5):
    """keeper_signals.features of every run -> LAB/<key>_kp.npz."""
    todo = [r for r in runs() if not os.path.exists(col_path(r, "kp"))]
    print("{} runs to build (keeper signals)".format(len(todo)), flush=True)
    with Pool(workers, maxtasksperchild=8) as pool:
        for i, k in enumerate(pool.imap_unordered(_kp_job, todo), 1):
            if i % 25 == 0 or i == len(todo):
                print("  {} / {} ({})".format(i, len(todo), k), flush=True)


def _sp_job(r):
    import calibrate
    import setpiece_signals
    import warnings
    warnings.filterwarnings("ignore")
    out = col_path(r, "sp")
    if os.path.exists(out):
        return key(r)
    game, labels, cond, seed, _ = r
    teams, players, ball, bs, blocks, gt = calibrate.case((game, seed), cond, labels)
    F, names = setpiece_signals.features(teams, players, ball, blocks, calibrate.pitch_of((game, seed)))
    tmp = out + ".tmp.npz"
    np.savez(tmp, X=F, names=np.array(names))
    os.replace(tmp, out)
    return key(r)


def cache_sp(workers=5):
    """setpiece_signals.features of every run -> LAB/<key>_sp.npz (12/11)."""
    todo = [r for r in runs() if not os.path.exists(col_path(r, "sp"))]
    print("{} runs to build (set-piece signals)".format(len(todo)), flush=True)
    with Pool(workers, maxtasksperchild=8) as pool:
        for i, k in enumerate(pool.imap_unordered(_sp_job, todo), 1):
            if i % 25 == 0 or i == len(todo):
                print("  {} / {} ({})".format(i, len(todo), k), flush=True)


def _col_job(r):
    import calibrate
    import collective
    out = col_path(r)
    if os.path.exists(out):
        return key(r)
    game, labels, cond, seed, _ = r
    teams, players, ball, bs, blocks, gt = calibrate.case((game, seed), cond, labels)
    F, names = collective.features(teams, players, ball, blocks, calibrate.pitch_of((game, seed)))
    tmp = out + ".tmp.npz"
    np.savez(tmp, X=F, names=np.array(names))
    os.replace(tmp, out)
    return key(r)


def cache_col(workers=3):
    """collective.features of every run -> LAB/<key>_col.npz."""
    import warnings
    warnings.filterwarnings("ignore")
    os.makedirs(LAB, exist_ok=True)
    todo = [r for r in runs() if not os.path.exists(col_path(r))]
    print("{} runs to build (collective)".format(len(todo)), flush=True)
    with Pool(workers, maxtasksperchild=4) as pool:
        for i, k in enumerate(pool.imap_unordered(_col_job, todo), 1):
            if i % 10 == 0 or i == len(todo):
                print("  {} / {} ({})".format(i, len(todo), k), flush=True)


def load(r, mmap=True):
    return np.load(path(r), mmap_mode="r" if mmap else None)


def blocks_of(z):
    return [(k + 1, int(a), int(b)) for k, (a, b) in enumerate(z["blocks"])]


def fold_of():
    """{game: fold}: matches dealt round-robin inside each dataset, so every fold has some of each."""
    out = {}
    for s in ("DFL", "Metrica", "SkillCorner", "PFF"):
        gs = sorted({r[0] for r in runs() if set_of(r[0]) == s}, key=str)
        for i, g in enumerate(gs):
            out[g] = i % FOLDS
    return out


def full_X(r, cols=None, extra=None, extra_cols=None):
    """X of a run: the cached features (cols: subset), plus the cached extra block (e.g. "col")."""
    z = load(r)
    Xr = np.asarray(z["X"]) if cols is None else np.asarray(z["X"])[:, cols]
    for b in _blocks(extra):
        e = np.load(col_path(r, b), mmap_mode="r")["X"]
        e = np.asarray(e) if (extra_cols is None or b != "col") else np.asarray(e)[:, extra_cols]
        Xr = np.concatenate([Xr, e], axis=1)
    return Xr


def _blocks(extra):
    if not extra:
        return ()
    return (extra,) if isinstance(extra, str) else tuple(extra)


def training_rows(rs, cols=None, every=EVERY, pff_every=PFF_EVERY, extra=None, extra_cols=None):
    """Stacked (X, y) of the usable rows of the given runs, one frame in `every`."""
    Xs, ys = [], []
    for r in rs:
        z = load(r)
        step = pff_every if set_of(r[0]) == "PFF" else every
        m = np.asarray(z["usable"]).copy()
        m[np.arange(len(m)) % step != 0] = False
        idx = np.flatnonzero(m)
        Xr = np.asarray(z["X"][idx])
        Xr = Xr if cols is None else Xr[:, cols]
        for b in _blocks(extra):
            e = np.asarray(np.load(col_path(r, b), mmap_mode="r")["X"][idx])
            Xr = np.concatenate([Xr, e if (extra_cols is None or b != "col") else e[:, extra_cols]], axis=1)
        Xs.append(Xr)
        ys.append(np.asarray(z["gt"][idx]).astype(np.int8))
    return np.concatenate(Xs), np.concatenate(ys)


def oof(tag="base", cols=None, extra=None, extra_names=None, train_filter=None):
    """Out-of-fold P(H), P(A), P(D) for every test run -> LAB/oof_<tag>.npz. cols: subset of the
    base feature columns (by name); extra: "col" to append the collective block (extra_names: a
    subset of it)."""
    import possession_ml as ml
    rs = runs()
    names = [str(x) for x in load(rs[0])["names"]]
    idx = None
    if cols is not None:
        idx = [names.index(c) for c in cols]
        names = list(cols)
    eidx = None
    for b in _blocks(extra):
        en = [str(x) for x in np.load(col_path(rs[0], b))["names"]]
        if extra_names is not None and b == "col":
            eidx = [en.index(c) for c in extra_names]
            en = list(extra_names)
        names = names + en
    fo = fold_of()
    out = {}
    for f in range(FOLDS):
        tr = [r for r in rs if fo[r[0]] != f and (train_filter is None or train_filter(r))]
        X, y = training_rows(tr, idx, extra=extra, extra_cols=eidx)
        clf = ml.fit3([X], [y], names)
        del X, y
        for r in rs:
            if fo[r[0]] != f or r[4] != "test":
                continue
            out[key(r)] = ml.prob3(clf, full_X(r, idx, extra, eidx), names).astype(np.float32)
        print("fold {} done".format(f), flush=True)
    np.savez(os.path.join(LAB, "oof_{}.npz".format(tag)), **out)
    if extra:
        imp = getattr(clf, "feature_importances_", None)
    return out


def oof_provider(tag="base", extra=None, every=20, pff_every=50):
    """Across providers: for each dataset, a model trained on the OTHER three only predicts its
    test runs -> LAB/oofprov_<tag>.npz. Lighter sampling than oof() (one frame in `every`), the same
    for every model compared, so the comparison is fair and it runs in a few hours."""
    import possession_ml as ml
    rs = runs()
    names = [str(x) for x in load(rs[0])["names"]]
    if extra:
        names = names + [str(x) for x in np.load(col_path(rs[0]))["names"]]
    out = {}
    for s in ("DFL", "Metrica", "SkillCorner", "PFF"):
        tr = [r for r in rs if set_of(r[0]) != s]
        X, y = training_rows(tr, every=every, pff_every=pff_every, extra=extra)
        clf = ml.fit3([X], [y], names)
        del X, y
        for r in rs:
            if set_of(r[0]) == s and r[4] == "test":
                out[key(r)] = ml.prob3(clf, full_X(r, None, extra), names).astype(np.float32)
        print("provider {} held out: done".format(s), flush=True)
    np.savez(os.path.join(LAB, "oof_prov_{}.npz".format(tag)), **out)
    return out


def load_oof(tag="base"):
    z = np.load(os.path.join(LAB, "oof_{}.npz".format(tag)))
    return {k: z[k] for k in z.files}


# ---- error taxonomy -------------------------------------------------------------------------------
def _segs(codes, runs_):
    return pm.segments(codes, runs_)


def taxonomy(gt, pred):
    """What goes wrong, in events per match and in their cost. Dead-ball errors are judged on the D
    runs, possession errors on the H / A spells (maximal runs, as the official metric builds them).

    false_stop     a predicted D run that touches no true D (play called dead): it splits a possession
    missed_stop    a true D run with no predicted D on it: two possessions merged (or a team spell
                   carried through the stoppage)
    early_start / late_start / early_end / late_end   seconds a matched D run starts / ends off the
                   true one (median over matched runs, and share off by more than 1 s)
    split          a true possession covered by two or more predicted spells of its team
    merged         a predicted spell covering two or more true possessions of its team
    wrong_team     seconds of alive play given to the wrong team
    """
    runs_ = pm.valid_runs(gt)
    gs, ps = _segs(gt, runs_), _segs(pred, runs_)
    gd = [(a, b) for a, b, k in gs if k == D]
    pdd = [(a, b) for a, b, k in ps if k == D]
    pa = np.array([a for a, _ in pdd]) if pdd else np.zeros(0, int)
    pb = np.array([b for _, b in pdd]) if pdd else np.zeros(0, int)
    ga = np.array([a for a, _ in gd]) if gd else np.zeros(0, int)
    gb = np.array([b for _, b in gd]) if gd else np.zeros(0, int)
    missed, ds, de = 0, [], []
    for a, b in gd:
        ov = np.minimum(pb, b) - np.maximum(pa, a)
        hit = np.flatnonzero(ov > 0)
        if not len(hit):
            missed += 1
            continue
        ds.append((pa[hit[0]] - a) / FPS)
        de.append((pb[hit[-1]] - b) / FPS)
    false_stop = sum(1 for a, b in pdd if not np.any((np.minimum(gb, b) - np.maximum(ga, a)) > 0))
    gsp = pm.possession_spells(gt, runs_)
    psp = pm.possession_spells(pred, runs_)
    split = merged = 0
    for a, b, k in gsp:
        n = sum(1 for c, e, kk in psp if kk == k and min(b, e) - max(a, c) > 0.2 * (b - a))
        split += n >= 2
    for c, e, kk in psp:
        n = sum(1 for a, b, k in gsp if k == kk and min(b, e) - max(a, c) > 0.2 * (b - a))
        merged += n >= 2
    alive = (gt < 2) & (pred < 2)
    ds, de = np.array(ds), np.array(de)
    return {"n_stops": len(gd), "n_poss": len(gsp), "n_pred_poss": len(psp), "false_stop": false_stop,
            "missed_stop": missed, "start_med": float(np.median(ds)) if len(ds) else np.nan,
            "start_late1": float(np.mean(ds > 1)) if len(ds) else np.nan,
            "start_early1": float(np.mean(ds < -1)) if len(ds) else np.nan,
            "end_med": float(np.median(de)) if len(de) else np.nan,
            "end_late1": float(np.mean(de > 1)) if len(de) else np.nan,
            "end_early1": float(np.mean(de < -1)) if len(de) else np.nan,
            "split": split, "merged": merged, "wrong_team_s": float(np.sum(alive & (gt != pred)) / FPS),
            "minutes": float(np.sum(gt != X) / FPS / 60)}


def scores(gt, pred):
    e = pm.evaluate(gt, pred, bootstrap_n=0)
    return e["spell_f1_mean"], e["macro_f1"]


def evaluate_decoder(oofp, decoder, which=None):
    """{run key: (spellF1, macroF1, taxonomy)} for every test run in oofp (or in `which`)."""
    out = {}
    for k, p3 in oofp.items():
        if which is not None and k not in which:
            continue
        r = _run_of_key(k)
        z = load(r)
        gt = np.asarray(z["gt"])
        pred = decoder(p3, z, r)
        out[k] = scores(gt, pred) + (taxonomy(gt, pred),)
    return out


_KEYS = None


def _run_of_key(k):
    global _KEYS
    if _KEYS is None:
        _KEYS = {key(r): r for r in runs()}
    return _KEYS[k]


def report(res, title, keys=("realistic1",)):
    """Per dataset: spellF1, macroF1 and the taxonomy (per match)."""
    print("\n== " + title)
    print("{:12s} {:>4s} {:>7s} {:>7s} | {:>9s} {:>9s} | {:>11s} {:>11s} | {:>11s} {:>11s} | {:>6s} {:>6s} {:>8s}".format(
        "set / cond", "n", "spellF1", "macroF1", "false D/m", "missed/m", "start med", ">1s lt/er", "end med", ">1s lt/er",
        "split", "merged", "team s/m"))
    for s in ("DFL", "Metrica", "SkillCorner", "PFF"):
        for cond in keys:
            rr = [v for k, v in res.items() if k.startswith(s + "_") and k.endswith("_" + cond)]
            if not rr:
                continue
            T = [v[2] for v in rr]
            mins = sum(t["minutes"] for t in T) / len(T)
            f = lambda kk: np.nanmean([t[kk] for t in T])  # noqa: E731
            per_poss = lambda kk: 100 * sum(t[kk] for t in T) / max(sum(t["n_poss"] for t in T), 1)  # noqa: E731
            print("{:12s} {:4d} {:7.3f} {:7.3f} | {:9.2f} {:9.2f} | {:+11.2f} {:5.0f}/{:<5.0f} | {:+11.2f} {:5.0f}/{:<5.0f} | {:5.1f}% {:5.1f}% {:8.1f}".format(
                s + " " + cond[:4], len(rr), np.mean([v[0] for v in rr]), np.mean([v[1] for v in rr]),
                100 * f("false_stop") / mins, 100 * f("missed_stop") / mins,
                f("start_med"), 100 * f("start_late1"), 100 * f("start_early1"),
                f("end_med"), 100 * f("end_late1"), 100 * f("end_early1"),
                per_poss("split"), per_poss("merged"), f("wrong_team_s") / mins))
    print("  false D / missed D: per 100 minutes of match; start / end: seconds predicted minus true (median), and the share of "
          "matched stoppages more than 1 s late / early; split / merged: share of true possessions; team: s per minute")


def decode_default(p3, z, r):
    import possession_ml as ml
    return ml.decode3(p3, blocks_of(z), ml.COST_TEAM, ml.COST_DEAD)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "errors"
    tag = sys.argv[2] if len(sys.argv) > 2 else "base"
    if cmd == "cache":
        cache()
    elif cmd == "cache_col":
        cache_col()
    elif cmd == "oof_col":
        oof(tag, extra="col")
    elif cmd == "oof_orig":      # the training set of before 11/11, predicting every test run (keeper ones too)
        oof(tag, train_filter=original_training)
    elif cmd == "oof_all":       # every run in training, keeper versions of all the datasets included
        oof(tag)
    elif cmd == "oof_kp":        # + the keeper signals
        oof(tag, extra="kp")
    elif cmd == "cache_kp":
        cache_kp()
    elif cmd == "cache_sp":
        cache_sp()
    elif cmd == "oof_kpsp":      # + the keeper signals + the set-piece signals (12/11)
        oof(tag, extra=("kp", "sp"))
    elif cmd == "oof_kpsp_nv":   # the same, trained without the camera-framing runs (predicts them too)
        oof(tag, extra=("kp", "sp"), train_filter=lambda r: r[2] != "view")
    elif cmd == "prov":
        oof_provider(tag, extra="col" if tag.startswith("col") else None)
    elif cmd == "oof":
        oof(tag)
    elif cmd == "errors":
        o = load_oof(tag)
        res = evaluate_decoder(o, decode_default)
        report(res, "out of fold, model '{}', decoder 16 / 32".format(tag), keys=("realistic1", "moved1"))
        v2 = evaluate_decoder(o, lambda p3, z, r: np.asarray(z["codes"]))
        report(v2, "v2 (rules) on the same runs", keys=("realistic1", "moved1"))
