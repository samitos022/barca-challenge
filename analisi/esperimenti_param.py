"""PARAMS variants of possession_v2 on the external full matches (never BAR-ATH), side by side.

    .venv/bin/python analisi/esperimenti_param.py "base=" "move_last=MOVE_LAST=1" "x=MOVE_LAST=1,SEQ_RULE=1"
    [--sets idsse-flag,idsse-tracab,metrica] [--cond degraded]

Each variant is name=KEY=value,KEY=value (empty = current PARAMS). The first variant is the
reference for the runs up / down column. IDSSE runs use KICKOFF_MAX_S = 0 (TRACAB halves start
after the kick-off, see analisi/bench_esterno.py).
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

SETS = {
    "idsse-flag": ("idsse", "flag", (1,)),
    "idsse-tracab": ("idsse", "tracab", (1,)),
    "metrica": ("metrica", None, (1, 2)),
}


def parse(spec):
    name, _, rest = spec.partition("=")
    out = {}
    for kv in filter(None, rest.split(",")):
        k, v = kv.split("=")
        out[k] = float(v)
    return name, out


def job(args):
    run, cond, labels, extra = args
    teams, players, ball, bs, blocks, gt = calibrate.case(run, cond, labels)
    p = {**v2.PARAMS, **extra}
    if run[0] in idsse.GAMES:
        p["KICKOFF_MAX_S"] = 0
    lab, _, _ = v2.predict(teams, players, ball, bs, blocks, calibrate.pitch_of(run), p)
    pc = pm.label_codes(lab)
    e = pm.evaluate(gt, pc, bootstrap_n=0)
    g = gt != 3
    return e["spell_f1_mean"], e["macro_f1"], e["n_pred_spells"], e["n_gt_spells"], float(np.mean(pc[g] == 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="+")
    ap.add_argument("--sets", default="idsse-flag,idsse-tracab,metrica")
    ap.add_argument("--cond", default="degraded")
    a = ap.parse_args()
    variants = [parse(s) for s in a.variants]
    for sname in a.sets.split(","):
        games, labels, seeds = SETS[sname]
        runs = [r for r in calibrate.GAME_SETS[games] if r[1] in seeds]
        print("\n== {} ({} runs, {} tracking)".format(sname, len(runs), a.cond))
        print("{:28s} {:>8s} {:>8s} {:>9s} {:>6s} {:>10s}".format("variant", "spellF1", "macroF1", "poss", "D%", "up/down"))
        base = None
        for name, extra in variants:
            with Pool(min(7, len(runs))) as pool:
                R = np.array(pool.map(job, [(r, a.cond, labels, extra) for r in runs]))
            if base is None:
                base = R
            up = int(np.sum(R[:, 0] > base[:, 0] + 1e-9))
            dn = int(np.sum(R[:, 0] < base[:, 0] - 1e-9))
            print("{:28s} {:8.3f} {:8.3f} {:4.0f}/{:<4.0f} {:6.1f} {:>5d}/{:<4d}".format(
                name, R[:, 0].mean(), R[:, 1].mean(), R[:, 2].mean(), R[:, 3].mean(), 100 * R[:, 4].mean(), up, dn),
                flush=True)


if __name__ == "__main__":
    main()
