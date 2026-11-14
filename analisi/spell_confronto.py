"""Two out-of-fold models side by side with the same decoder, per dataset and condition, with the
error taxonomy: where one gains and the other loses.

    .venv/bin/python analisi/spell_confronto.py base base_s2 [team cost] [dead cost]
"""

import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE] + [os.path.join(os.path.dirname(HERE), d) for d in ("solution", "evaluation", "starter", "solution/external")]
import spell_exp as E  # noqa: E402
import spell_lab as L  # noqa: E402
import viterbi  # noqa: E402

_O = {}


def _init(tags, ct, cd):
    _O["o"] = [L.load_oof(t) for t in tags]
    _O["c"] = (ct, cd)


def job(k):
    z = L.load(L._run_of_key(k))
    gt = np.asarray(z["gt"])
    bl = L.blocks_of(z)
    out = []
    for o in _O["o"]:
        pred = viterbi.decode(o[k], bl, *_O["c"])
        out.append(E.quick_scores(gt, pred) + (L.taxonomy(gt, pred),))
    return k, out


def main(a, b, ct=16.0, cd=32.0):
    keys = list(L.load_oof(a))
    with Pool(6, initializer=_init, initargs=((a, b), ct, cd)) as pool:
        R = dict(pool.map(job, keys))
    print("{:22s} {:>4s} | {:>15s} | {:>15s} | {:>13s} | {:>13s} | {:>17s} | {:>17s} | {:>13s}".format(
        "set / condition", "n", "spellF1 " + a[:6] + "/" + b[-6:], "macroF1", "false D /100m", "missed /100m",
        "start >1s off %", "end >1s off %", "merged %"))
    for s in E.SETS:
        for cond in ("realistic1", "moved1"):
            ks = [k for k in R if k.startswith(s + "_") and k.endswith("_" + cond)]
            if not ks:
                continue
            row = []
            for i in (0, 1):
                T = [R[k][i][2] for k in ks]
                mins = np.mean([t["minutes"] for t in T])
                row.append((np.mean([R[k][i][0] for k in ks]), np.mean([R[k][i][1] for k in ks]),
                            100 * np.mean([t["false_stop"] for t in T]) / mins, 100 * np.mean([t["missed_stop"] for t in T]) / mins,
                            100 * np.nanmean([t["start_late1"] + t["start_early1"] for t in T]),
                            100 * np.nanmean([t["end_late1"] + t["end_early1"] for t in T]),
                            100 * sum(t["merged"] for t in T) / sum(t["n_poss"] for t in T)))
            print("{:22s} {:4d} | {:7.3f}/{:<7.3f} | {:7.3f}/{:<7.3f} | {:6.2f}/{:<6.2f} | {:6.2f}/{:<6.2f} | {:8.1f}/{:<8.1f} | {:8.1f}/{:<8.1f} | {:6.1f}/{:<6.1f}".format(
                s + " " + cond[:-1], len(ks), row[0][0], row[1][0], row[0][1], row[1][1], row[0][2], row[1][2], row[0][3], row[1][3],
                row[0][4], row[1][4], row[0][5], row[1][5], row[0][6], row[1][6]), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(float(x) for x in sys.argv[3:5]))
