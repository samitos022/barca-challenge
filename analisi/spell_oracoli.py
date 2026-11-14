"""Where the spellF1 is lost, out of fold on the external matches: the model against oracles that
fix one thing with the truth.

    .venv/bin/python analisi/spell_oracoli.py [tag]
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

H, A, D, X = 0, 1, 2, 3
_O = {}


def job(k):
    o = _O["o"]
    z = L.load(L._run_of_key(k))
    gt = np.asarray(z["gt"])
    bl = L.blocks_of(z)
    p3 = o[k]
    pred = viterbi.decode(p3, bl, 16, 32)
    g = gt != X
    team = np.where(p3[:, 0] >= p3[:, 1], H, A)
    od = np.where(gt == D, D, np.where(pred == D, team, pred))
    od[~g] = X
    lp = np.log(np.clip(p3, 1e-4, 1))
    em = lp.copy()
    em[gt == D] = [-50.0, -50.0, 0.0]
    em[gt < 2, 2] = -50.0
    od2 = viterbi.decode(p3, bl, 16, 32, emit=em)
    ot = pred.copy()
    m = (gt < 2) & (pred < 2)
    ot[m] = gt[m]
    # timing oracle: predicted stoppages that overlap a true one take its exact extent (missed / false ones stay)
    otm = pred.copy()
    import possession_metrics as pm
    runs_ = pm.valid_runs(gt)
    for a, b, kk in pm.segments(gt, runs_):
        if kk == D and np.any(pred[a:b] == D):
            otm[a:b] = D
            # frames the model had as dead around it but truly alive: back to the model's team
    for a, b, kk in pm.segments(pred, runs_):
        if kk == D and np.any(gt[a:b] == D):
            sel = np.arange(a, b)
            sel = sel[gt[sel] != D]
            otm[sel] = team[sel]
    # team oracles split by kind: near a true possession change (timing), inside short true possessions
    # (missed / invented short spells), elsewhere (whole spells given to the wrong team)
    ch = np.flatnonzero((gt[1:] != gt[:-1]) & (gt[1:] < 2) & (gt[:-1] < 2)) + 1
    near = {}
    for w in (12, 25, 75):
        mk = np.zeros(len(gt), bool)
        for t in ch:
            mk[max(0, t - w):t + w] = True
        near[w] = mk
    short = np.zeros(len(gt), bool)
    for a, b, kk in pm.possession_spells(gt, runs_):
        if b - a < 4 * 25:
            short[a:b] = True

    def fix(mask):
        q = pred.copy()
        mm = m & mask
        q[mm] = gt[mm]
        return q
    extra = [E.quick_scores(gt, fix(near[12])), E.quick_scores(gt, fix(near[25])), E.quick_scores(gt, fix(near[75])),
             E.quick_scores(gt, fix(short)), E.quick_scores(gt, fix(~near[75] & ~short))]
    return k, [E.quick_scores(gt, pred), E.quick_scores(gt, otm), E.quick_scores(gt, od), E.quick_scores(gt, od2), E.quick_scores(gt, ot)] + extra


def main(tag="base"):
    _O["o"] = L.load_oof(tag)
    keys = E.test_keys(_O["o"])
    with Pool(6, initializer=_init, initargs=(tag,)) as pool:
        R = dict(pool.map(job, keys))
    names = ["model, decoder 16 / 32", "oracle: timing of the stoppages it finds", "oracle: dead ball (team = frame argmax)",
             "oracle: dead ball, team decoded", "oracle: team (dead ball from the model)",
             "  team only within 0.5 s of a true change", "  team only within 1 s of a true change", "  team only within 3 s of a true change",
             "  team only inside true possessions < 4 s", "  team only elsewhere (not near, not short)"]
    E.summary([(n, {k: v[i] for k, v in R.items()}) for i, n in enumerate(names)],
              "where the spellF1 is lost (model '{}', out of fold, realistic runs)".format(tag))


def _init(tag):
    _O["o"] = L.load_oof(tag)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "base")
