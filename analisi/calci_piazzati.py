"""Set pieces and who takes them (solution/setpiece_signals.py), on the external matches.

    .venv/bin/python analisi/calci_piazzati.py check [condition]     (view runs: MTAH_VIEW=1)
        per layout: episodes per match, share of them on a dead ball, share whose team takes the
        next restart (true labels), and how many true restarts have a layout of the right team in
        the 10 s before them
    .venv/bin/python analisi/calci_piazzati.py eval <oof tag> [<oof tag> ...]
        restarts in the decoded out-of-fold labels: first team after each true restart right,
        and short possessions right after a restart (under 3 s) found
"""

import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for d in ("", "starter", "solution", "solution/external", "evaluation", "analisi"):
    sys.path.insert(0, os.path.join(ROOT, d))
import spell_lab as L  # noqa: E402

FPS = 25
SETS = ("DFL", "Metrica", "SkillCorner", "PFF")
D = 2


def restarts(gt):
    """frames t where a true stoppage ends: gt[t-1] == D and gt[t] is H or A."""
    return np.flatnonzero((gt[:-1] == D) & np.isin(gt[1:], (0, 1))) + 1


def _check(r):
    import calibrate
    import setpiece_signals as sp
    import warnings
    warnings.filterwarnings("ignore")
    game, labels, cond, seed, _ = r
    teams, players, ball, bs, blocks, gt = calibrate.case((game, seed), cond, labels)
    lay = sp.layouts(teams, players, blocks, calibrate.pitch_of((game, seed)))
    rs = restarts(gt)
    res = {}
    for nm in sp.NAMES:
        ep, dead, right, lag = 0, 0, 0, []
        for k in (0, 1):
            e = np.flatnonzero(np.diff(np.concatenate(([0], lay[nm][:, k].astype(np.int8), [0]))))
            for a, b in zip(e[::2], e[1::2]):
                if np.all(gt[a:b] == 3):
                    continue
                ep += 1
                dead += np.mean(gt[a:b] == D) >= 0.5
                nxt = rs[(rs >= a) & (rs <= b + 60 * FPS)]
                if len(nxt):
                    right += gt[nxt[0]] == k
                    lag.append((nxt[0] - b) / FPS)
        cover = 0
        for t in rs:
            w = slice(max(0, t - 10 * FPS), t + 1)
            cover += lay[nm][w, gt[t]].any()
        res[nm] = (ep, dead, right, len(lag), lag, cover)
    res["_restarts"] = len(rs)
    return L.set_of(game), res


def check(cond="realistic"):
    rs = [r for r in L.runs() if r[2] == cond and r[3] == 1]
    with Pool(5) as pool:
        out = pool.map(_check, rs)
    import setpiece_signals as sp
    for s in SETS:
        sel = [res for ss, res in out if ss == s]
        nr = sum(x["_restarts"] for x in sel)
        print("\n== {} ({} matches, {} true restarts)".format(s, len(sel), nr))
        print("  {:8s} {:>9s} {:>9s} {:>16s} {:>14s} {:>18s}".format(
            "layout", "per match", "on dead", "team restarts", "restart after", "restarts covered"))
        for nm in sp.NAMES:
            ep = sum(x[nm][0] for x in sel)
            dead = sum(x[nm][1] for x in sel)
            right = sum(x[nm][2] for x in sel)
            nn = sum(x[nm][3] for x in sel)
            lag = np.concatenate([np.array(x[nm][4], dtype=float) for x in sel]) if nn else np.zeros(0)
            cov = sum(x[nm][5] for x in sel)
            print("  {:8s} {:9.1f} {:8.0f}% {:15.0f}% {:12.1f} s {:17.1f}%".format(
                nm, ep / len(sel), 100 * dead / max(ep, 1), 100 * right / max(nn, 1),
                float(np.median(lag)) if len(lag) else np.nan, 100 * cov / max(nr, 1)))


def restart_metrics(gt, pred):
    rs = restarts(gt)
    first_ok, n_first, short_ok, n_short = 0, 0, 0, 0
    alive = np.isin(gt, (0, 1))
    for t in rs:
        w = pred[t:t + 2 * FPS]
        tm = w[np.isin(w, (0, 1))]
        if len(tm):
            n_first += 1
            first_ok += tm[0] == gt[t]
        end = t
        while end < len(gt) and gt[end] == gt[t]:
            end += 1
        if (end - t) < 3 * FPS and alive[t]:
            n_short += 1
            short_ok += np.mean(pred[t:end] == gt[t]) >= 0.5
    return first_ok, n_first, short_ok, n_short


def evaluate(tags):
    import possession_ml as ml
    import possession_metrics as pm
    import viterbi
    oo = {t: L.load_oof(t) for t in tags}
    keys = sorted(set.intersection(*[set(o) for o in oo.values()]))
    keys = [k for k in keys if k.endswith(("_realistic1", "_keeper1", "_view1"))]
    acc = {}
    for k in keys:
        r = L._run_of_key(k)
        z = L.load(r)
        gt, bl = np.asarray(z["gt"]), L.blocks_of(z)
        cond = k.rsplit("_", 1)[1][:-1]
        for t in tags:
            pred = viterbi.decode(oo[t][k], bl, ml.COST_TEAM, ml.COST_DEAD)
            e = pm.evaluate(gt, pred, bootstrap_n=0)
            acc.setdefault((cond, t, L.set_of(r[0])), []).append(
                (e["spell_f1_mean"], e["macro_f1"]) + restart_metrics(gt, pred))
    for cond in ("realistic", "keeper", "view"):
        if not any(c == cond for c, _, _ in acc):
            continue
        print("\n== {} tracking".format(cond))
        print("  {:14s} {:12s} {:>8s} {:>8s} {:>22s} {:>26s}".format(
            "dataset", "model", "spellF1", "macroF1", "first team after restart", "short restart possessions"))
        for s in SETS:
            for t in tags:
                v = acc.get((cond, t, s))
                if not v:
                    continue
                a = np.array(v, dtype=float)
                print("  {:14s} {:12s} {:8.3f} {:8.3f} {:21.1f}% {:25.1f}%".format(
                    s, t, a[:, 0].mean(), a[:, 1].mean(), 100 * a[:, 2].sum() / max(a[:, 3].sum(), 1),
                    100 * a[:, 4].sum() / max(a[:, 5].sum(), 1)))
        for t in tags:
            m = [np.array(acc[(cond, t, s)], dtype=float)[:, 0].mean() for s in SETS if (cond, t, s) in acc]
            print("  {:14s} {:12s} {:8.3f}  (mean of the datasets)".format("mean", t, np.mean(m)))


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check(sys.argv[2] if len(sys.argv) > 2 else "realistic")
    else:
        evaluate(sys.argv[2:])
