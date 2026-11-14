"""How many goalkeeper holds each external dataset has (degrade.keeper_holds on the provider's own
truth and ball), and how the out-of-fold models treat them once the ball is removed (keeper runs).

    .venv/bin/python analisi/prese_portiere.py count
    .venv/bin/python analisi/prese_portiere.py eval [tag ...]
"""

import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE] + [os.path.join(os.path.dirname(HERE), d) for d in ("solution", "evaluation", "starter", "solution/external")]
import spell_lab as L  # noqa: E402

FPS = 25


def _count(args):
    import calibrate
    import degrade
    import idsse
    game, labels = args
    if game in idsse.GAMES:
        teams, players, ball, bs, blocks, gt = calibrate.case((game, 1), "clean", labels)
    else:
        teams, players, ball, bs, blocks, gt = calibrate.case((game, 1), "realistic", labels)
    holds = degrade.keeper_holds(teams, players, ball, gt, blocks, calibrate.pitch_of((game, 1)))
    mins = np.sum(gt != 3) / FPS / 60
    return L.set_of(game), len(holds), sum(b - a for a, b, _ in holds) / FPS, mins


def count():
    games = {}
    for r in L.runs():
        games[r[0]] = r[1]
    with Pool(5) as pool:
        R = pool.map(_count, list(games.items()))
    for s in ("DFL", "Metrica", "SkillCorner", "PFF"):
        rr = [x for x in R if x[0] == s]
        n = sum(x[1] for x in rr)
        print("{:12s} {:2d} matches: {:4d} holds ({:.1f} per match, {:.1f} per 90 min), {:.1f} s each".format(
            s, len(rr), n, n / len(rr), 90 * n / sum(x[3] for x in rr), sum(x[2] for x in rr) / max(n, 1)))


_E = {}


def _init(tags):
    _E["o"] = [L.load_oof(t) for t in tags]


def _eval(k):
    import calibrate
    import viterbi
    import spell_exp as E
    r = L._run_of_key(k)
    game, labels, cond, seed, _ = r
    z = L.load(r)
    gt = np.asarray(z["gt"])
    bl = L.blocks_of(z)
    hidden = np.zeros(len(gt), bool)
    if cond == "keeper":
        b_kp = calibrate.case((game, seed), "keeper", labels)[2]
        b_re = calibrate.case((game, seed), "realistic", labels)[2]
        hidden = np.isnan(b_kp[:, 0]) & ~np.isnan(b_re[:, 0]) & (gt < 2)
    out = []
    for o in _E["o"]:
        pred = viterbi.decode(o[k], bl, 16, 32)
        sp, mf = E.quick_scores(gt, pred)
        out.append((sp, mf, int(hidden.sum()), int((hidden & (pred == 2)).sum()), int((hidden & (pred < 2) & (pred == gt)).sum())))
    return k, out


def evaluate(tags):
    keys = [k for k in L.load_oof(tags[0]) if k.endswith("_realistic1") or k.endswith("_keeper1")]
    with Pool(5, initializer=_init, initargs=(tags,)) as pool:
        R = dict(pool.map(_eval, keys))
    print("{:12s} {:10s} {:26s} {:>8s} {:>8s} {:>22s} {:>22s}".format("dataset", "condition", "model", "spellF1", "macroF1",
                                                                     "hold frames called dead", "hold frames right team"))
    for s in ("DFL", "Metrica", "SkillCorner", "PFF"):
        for cond in ("realistic1", "keeper1"):
            ks = [k for k in R if k.startswith(s + "_") and k.endswith("_" + cond)]
            if not ks:
                continue
            for i, t in enumerate(tags):
                v = np.array([R[k][i] for k in ks], dtype=float)
                hold = v[:, 2].sum()
                print("{:12s} {:10s} {:26s} {:8.3f} {:8.3f} {:>22s} {:>22s}".format(
                    s if i == 0 else "", cond[:-1] if i == 0 else "", t, v[:, 0].mean(), v[:, 1].mean(),
                    "{:.1f}%".format(100 * v[:, 3].sum() / hold) if hold else "-", "{:.1f}%".format(100 * v[:, 4].sum() / hold) if hold else "-"))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        evaluate(sys.argv[2:])
    else:
        count()
