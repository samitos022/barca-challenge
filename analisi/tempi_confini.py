"""Timing of v2's possession boundaries on the external full matches (never BAR-ATH).

For every true spell whose best same-team predicted spell has IoU >= 0.3: predicted start - true
start and predicted end - true end, in seconds, split by what starts the spell (a turnover or a
restart after a dead ball) and what ends it. Systematic offsets are what a principled correction
could remove; spread is not.

    .venv/bin/python analisi/tempi_confini.py [--cond degraded]
"""

import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "solution"), os.path.join(ROOT, "solution", "external"),
          os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "starter")):
    sys.path.insert(0, p)
import calibrate  # noqa: E402
import idsse  # noqa: E402
import possession_metrics as pm  # noqa: E402
import possession_v2 as v2  # noqa: E402

FPS = 25


def job(args):
    run, cond, labels = args
    teams, players, ball, bs, blocks, gt = calibrate.case(run, cond, labels)
    p = dict(v2.PARAMS)
    if run[0] in idsse.GAMES:
        p["KICKOFF_MAX_S"] = 0
    lab, _, _ = v2.predict(teams, players, ball, bs, blocks, calibrate.pitch_of(run), p)
    pc = pm.label_codes(lab)
    runs = pm.valid_runs(gt)
    seg = pm.segments(gt, runs)
    before = {a: (seg[k - 1][2] if k and seg[k - 1][1] == a else 3) for k, (a, b, c) in enumerate(seg)}
    after = {b: (seg[k + 1][2] if k + 1 < len(seg) and seg[k + 1][0] == b else 3) for k, (a, b, c) in enumerate(seg)}
    gs, ps = pm.possession_spells(gt, runs), pm.possession_spells(pc, runs)
    best = {}
    for iou, i, j in pm.spell_overlaps(gs, ps):
        if iou >= 0.3 and i not in best:
            best[i] = j
    out = []
    for i, j in best.items():
        a, b, c = gs[i]
        pa, pb, _ = ps[j]
        out.append(((pa - a) / FPS, (pb - b) / FPS, before[a], after[b], (b - a) / FPS))
    return out


def summary(name, x):
    if not len(x):
        return "{:34s} n 0".format(name)
    return "{:34s} n {:5d}  median {:+5.2f} s  mean {:+5.2f}  IQR {:+5.2f}..{:+5.2f}  |err| < 1 s {:3.0f}%".format(
        name, len(x), np.median(x), np.mean(x), np.percentile(x, 25), np.percentile(x, 75), 100 * np.mean(np.abs(x) < 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="degraded")
    a = ap.parse_args()
    for games, labels, seeds in (("idsse", "flag", (1,)), ("idsse", "tracab", (1,)), ("metrica", None, (1, 2))):
        runs = [r for r in calibrate.GAME_SETS[games] if r[1] in seeds]
        with Pool(min(6, len(runs))) as pool:
            R = np.array([x for res in pool.map(job, [(r, a.cond, labels) for r in runs]) for x in res])
        print("\n== {} {} ({} tracking): spells with a match at IoU >= 0.3: {}".format(games, labels or "", a.cond, len(R)))
        st, en, bf, af = R[:, 0], R[:, 1], R[:, 2], R[:, 3]
        print(summary("start, after a turnover", st[np.isin(bf, (0, 1))]))
        print(summary("start, after a dead ball (restart)", st[bf == 2]))
        print(summary("end, by a turnover", en[np.isin(af, (0, 1))]))
        print(summary("end, by a dead ball (stoppage)", en[af == 2]))


if __name__ == "__main__":
    main()
