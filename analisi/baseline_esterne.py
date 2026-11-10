"""The official baseline (starter/baseline.py: the team of the player closest to the ball, D where
the ball is not tracked, majority vote over +-2 s) on the external matches, same tracking
conditions as the other versions (analisi/classifica_versioni.py, possession_ml.cv3).

    .venv/bin/python analisi/baseline_esterne.py
"""

import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "starter", "solution", "solution/external", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, d))
import baseline  # noqa: E402
import calibrate  # noqa: E402
import idsse  # noqa: E402
import metrica  # noqa: E402
import pff  # noqa: E402
import possession_metrics as pm  # noqa: E402
import skillcorner  # noqa: E402

WINDOWS = (("raw", 0), ("smooth25", 25), ("smooth50", 50), ("smooth125", 125))


def job(args):
    game, labels = args
    teams, players, ball, bs, blocks, gt = calibrate.case((game, 1), "realistic", labels)
    d = np.linalg.norm(players - ball[:, None], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    j = np.argmin(d, axis=1)
    codes = np.where(np.isfinite(d[np.arange(len(d)), j]), np.where(teams[j] == 0, baseline.HOME, baseline.AWAY),
                     baseline.NOBODY).astype(np.int8)
    out = {}
    for name, w in WINDOWS:
        c = pm.label_codes(baseline.smooth(codes, w))
        c = np.where(gt == 3, 3, c)
        e = pm.evaluate(gt, c, bootstrap_n=0)
        out[name] = (e["spell_f1_mean"], e["macro_f1"], float(np.mean(c[gt != 3] == gt[gt != 3])))
    return game, out


def main():
    sets = (("DFL (7)", [(g, "tracab") for g in idsse.GAMES]), ("Metrica (3)", [(g, None) for g in metrica.GAMES]),
            ("SkillCorner (20)", [(g, None) for g in skillcorner.GAMES]), ("Mondiale, 10 escluse", [(g, None) for g in pff.GAMES[:10]]))
    with Pool(5) as pool:
        R = dict(pool.map(job, [j for _, js in sets for j in js]))
    for name, _ in WINDOWS:
        line = "baseline_{:10s}".format(name)
        for sname, js in sets:
            v = np.array([R[g][name] for g, _ in js])
            line += " | {} spellF1 {:.3f} macroF1 {:.3f} right {:.1f}%".format(sname, v[:, 0].mean(), v[:, 1].mean(), 100 * v[:, 2].mean())
        print(line, flush=True)


if __name__ == "__main__":
    main()
