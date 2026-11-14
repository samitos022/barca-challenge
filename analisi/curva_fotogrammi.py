"""Does training on more frames help? The same model trained on fold 0's training matches with one
frame in `every` (PFF: one in `pff_every`, so it keeps the same weight against the other datasets),
scored on fold 0's held-out matches with the decoder 16 / 32.

    .venv/bin/python analisi/curva_fotogrammi.py
"""

import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE] + [os.path.join(os.path.dirname(HERE), d) for d in ("solution", "evaluation", "starter", "solution/external")]
import spell_exp as E  # noqa: E402
import spell_lab as L  # noqa: E402
import viterbi  # noqa: E402

STEPS = ((20, 50), (10, 25), (5, 12), (3, 8), (2, 5))
FOLD = 0


def _score(args):
    k, p3 = args
    z = L.load(L._run_of_key(k))
    gt = np.asarray(z["gt"])
    return k, E.quick_scores(gt, viterbi.decode(p3, L.blocks_of(z), 16, 32))


def main():
    import possession_ml as ml
    rs = L.runs()
    fo = L.fold_of()
    names = [str(x) for x in L.load(rs[0])["names"]]
    tr = [r for r in rs if fo[r[0]] != FOLD]
    te = [r for r in rs if fo[r[0]] == FOLD and r[4] == "test" and r[2] == "realistic"]
    print("fold {}: {} training runs, {} test runs".format(FOLD, len(tr), len(te)), flush=True)
    for every, pff_every in STEPS:
        t0 = time.time()
        X, y = L.training_rows(tr, every=every, pff_every=pff_every)
        n = len(y)
        clf = ml.fit3([X], [y], names)
        del X, y
        P = {L.key(r): ml.prob3(clf, L.full_X(r), names) for r in te}
        with Pool(5) as pool:
            S = dict(pool.map(_score, list(P.items())))
        line = "one frame in {:2d} (PFF {:2d}): {:9d} rows, {:5.0f} min |".format(every, pff_every, n, (time.time() - t0) / 60)
        means = []
        for s in E.SETS:
            v = [S[k] for k in S if k.startswith(s + "_")]
            m = np.mean(v, axis=0)
            means.append(m[0])
            line += " {} {:.3f} ({:.3f})".format(s, m[0], m[1])
        print(line + " | mean {:.3f}".format(np.mean(means)), flush=True)


if __name__ == "__main__":
    main()
