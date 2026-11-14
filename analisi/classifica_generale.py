"""One ranking of the versions on the external matches, same runs and same protocol for all of them
(the test runs of spell_lab, realistic tracking and the one with the keeper holds hidden):

  baseline_smooth50   starter/baseline.py (closest player's team, D where the ball is missing, +-2 s)
  v2                  the rules (possession_v2), the "codes" cached in the lab
  model, base         three states, training set of before 12/11 (out of fold, spell_lab "orig")
  model, keeper       three states + keeper holds and keeper signals (out of fold, "kpfeat")
  model, final        + set-piece signals, camera-framed runs in training, final penalty rule (out of fold, "kpspv2", 13/11)

    .venv/bin/python analisi/classifica_generale.py
"""

import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for d in ("", "starter", "solution", "solution/external", "evaluation", "analisi"):
    sys.path.insert(0, os.path.join(ROOT, d))
import possession_metrics as pm  # noqa: E402
import spell_lab as L  # noqa: E402
import viterbi  # noqa: E402

SETS = ("DFL", "Metrica", "SkillCorner", "PFF")


def _baseline(k):
    import baseline
    import calibrate
    import warnings
    warnings.filterwarnings("ignore")
    game, labels, cond, seed, _ = L._run_of_key(k)
    teams, players, ball, bs, blocks, gt = calibrate.case((game, seed), cond, labels)
    d = np.linalg.norm(players - ball[:, None], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    j = np.argmin(d, axis=1)
    codes = np.where(np.isfinite(d[np.arange(len(d)), j]), np.where(teams[j] == 0, baseline.HOME, baseline.AWAY),
                     baseline.NOBODY).astype(np.int8)
    c = pm.label_codes(baseline.smooth(codes, 50))
    return k, np.where(gt == 3, 3, c).astype(np.int8)


def main():
    import possession_ml as ml
    oo, ok, of = L.load_oof("orig"), L.load_oof("kpfeat"), L.load_oof("kpspv2")
    keys = sorted(k for k in oo if k in ok and k in of and (k.endswith("_realistic1") or k.endswith("_keeper1")))
    with Pool(5) as pool:
        base = dict(pool.map(_baseline, keys))
    res = {}
    for k in keys:
        z = L.load(L._run_of_key(k))
        gt, bl = np.asarray(z["gt"]), L.blocks_of(z)
        outs = {"baseline_smooth50": base[k], "v2": np.where(gt == 3, 3, np.asarray(z["codes"])),
                "model, base": viterbi.decode(oo[k], bl, ml.COST_TEAM, ml.COST_DEAD),
                "model, keeper": viterbi.decode(ok[k], bl, ml.COST_TEAM, ml.COST_DEAD),
                "model, final": viterbi.decode(of[k], bl, ml.COST_TEAM, ml.COST_DEAD)}
        s = L.set_of(L._run_of_key(k)[0])
        cond = "keeper" if k.endswith("_keeper1") else "realistic"
        for name, c in outs.items():
            e = pm.evaluate(gt, c, bootstrap_n=0)
            res.setdefault((cond, name, s), []).append((e["spell_f1_mean"], e["macro_f1"]))
    for cond in ("realistic", "keeper"):
        print("\n== {} tracking{}".format(cond, " (ball hidden while a keeper holds it)" if cond == "keeper" else ""))
        print("  {:20s}".format("") + "".join("{:>20s}".format(s) for s in SETS) + "{:>20s}".format("mean of the 4"))
        for name in ("baseline_smooth50", "v2", "model, base", "model, keeper", "model, final"):
            v = [np.mean(res[(cond, name, s)], axis=0) for s in SETS]
            m = np.mean(v, axis=0)
            print("  {:20s}".format(name) + "".join("     {:.3f} / {:.3f}".format(a, b) for a, b in v) +
                  "     {:.3f} / {:.3f}".format(*m))
    print("\n(spellF1 / macroF1; matches per set: " + ", ".join(
        "{} {}".format(s, len(res[("realistic", "v2", s)])) for s in SETS) + ")")


if __name__ == "__main__":
    main()
