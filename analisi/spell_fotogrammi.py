"""Frame-level view of out-of-fold models (same decoder): macroF1, frame accuracy, alive / dead F1, and
team accuracy on the frames that are truly alive, per dataset (realistic runs).

    .venv/bin/python analisi/spell_fotogrammi.py base col col_s2dead
"""

import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE] + [os.path.join(os.path.dirname(HERE), d) for d in ("solution", "evaluation", "starter", "solution/external")]
import possession_metrics as pm  # noqa: E402
import spell_exp as E  # noqa: E402
import spell_lab as L  # noqa: E402
import viterbi  # noqa: E402

_O = {}


def _init(tags):
    _O["o"] = [L.load_oof(t) for t in tags]


def job(k):
    z = L.load(L._run_of_key(k))
    gt = np.asarray(z["gt"])
    g = gt != 3
    out = []
    for o in _O["o"]:
        pred = viterbi.decode(o[k], L.blocks_of(z), 16, 32)
        cm = pm.scored_confusion(pm.confusion(gt, pred))
        alive = g & (gt < 2)
        out.append((pm.macro_f1(cm), float(np.mean(pred[g] == gt[g])), pm.alive_dead_f1(cm),
                    float(np.mean(pred[alive] == gt[alive]))))
    return k, out


def main(tags):
    keys = [k for k in L.load_oof(tags[0]) if k.endswith("_realistic1")]
    with Pool(6, initializer=_init, initargs=(tags,)) as pool:
        R = dict(pool.map(job, keys))
    print("{:12s} {:28s} {:>8s} {:>9s} {:>11s} {:>12s}".format("dataset", "model", "macroF1", "frames %", "alive/dead", "team (alive)"))
    for s in E.SETS:
        ks = [k for k in R if k.startswith(s + "_")]
        for i, t in enumerate(tags):
            v = np.mean([R[k][i] for k in ks], axis=0)
            print("{:12s} {:28s} {:8.3f} {:8.1f}% {:11.3f} {:11.1f}%".format(s if i == 0 else "", t, v[0], 100 * v[1], v[2], 100 * v[3]))


if __name__ == "__main__":
    main(sys.argv[1:])
