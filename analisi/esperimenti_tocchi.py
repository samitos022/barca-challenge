"""Touch / control variants of v1 measured on the external full matches (never BAR-ATH).

The four corrections archived in esperimenti/possession_v2_con_4_correzioni.py (a attribution up
to R_TOUCH_FAR with a lead, b control reach with a lead, c launch kicked by the player behind,
d confirmation by the next touch or control) are swapped into possession_v1 in memory, one at a
time and all together; possession_v2.py itself is not modified.

    .venv/bin/python analisi/esperimenti_tocchi.py [--sets idsse-flag,idsse-tracab,metrica]
"""

import argparse
import importlib.util
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
import possession_v1 as v1  # noqa: E402
import possession_v2 as v2  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "v2_4corr", os.path.join(ROOT, "esperimenti", "possession_v2_con_4_correzioni.py"))
c4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c4)

ORIG = {"control": v1.control, "touches": v1.touches, "look_ahead": v1.look_ahead}
EXTRA = {k: c4.PARAMS[k] for k in ("R_TOUCH_FAR", "R_FAR_FREE", "FREE_LEAD", "LAUNCH_BACK", "NEXT_CTRL_FRAMES")}
VARIANTS = {
    "base": "",
    "a attribution 3.5 m": "a",
    "b control reach": "b",
    "c launch": "c",
    "d next touch": "d",
    "a+c": "ac",
    "a+b+c+d": "abcd",
}
SETS = {
    "idsse-flag": ("idsse", "flag", (1,)),
    "idsse-tracab": ("idsse", "tracab", (1,)),
    "metrica": ("metrica", None, (1, 2)),
}


def patch(which, ctx):
    """Install the chosen corrections into possession_v1 (in this worker process only)."""
    v1.control = c4.control_v2 if "b" in which else ORIG["control"]
    v1.look_ahead = c4.confirm_next if "d" in which else ORIG["look_ahead"]
    if "a" in which or "c" in which:
        def touches(dv, dist, p):
            vel, speed, _, _ = v1.ball_kinematics(ctx["ball"], ctx["bs"], p)
            pp = dict(p)
            if "a" not in which:
                pp["R_TOUCH_FAR"] = p["R_TOUCH"]   # attribution as in v1
            team, is_touch, kind = c4.touches_v2(dv, dist, ctx["ball"], vel, speed, ctx["players"],
                                                 ctx["teams"], pp)
            if "c" not in which:
                team = np.where(kind == 2, -1, team).astype(np.int8)
                is_touch = is_touch & (kind != 2)
            return team, is_touch
        v1.touches = touches
    else:
        v1.touches = ORIG["touches"]


def job(args):
    run, labels, which = args
    teams, players, ball, bs, blocks, gt = calibrate.case(run, "degraded", labels)
    patch(which, {"ball": ball, "bs": bs, "players": players, "teams": teams})
    p = {**v2.PARAMS, **EXTRA}
    if run[0] in idsse.GAMES:
        p["KICKOFF_MAX_S"] = 0  # TRACAB halves start after the kick-off (harness fix, as bench_esterno)
    lab, _, _ = v2.predict(teams, players, ball, bs, blocks, calibrate.pitch_of(run), p)
    pc = pm.label_codes(lab)
    e = pm.evaluate(gt, pc, bootstrap_n=0)
    g = gt != 3
    return e["spell_f1_mean"], e["macro_f1"], e["n_pred_spells"], e["n_gt_spells"], float(np.mean(pc[g] == 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="idsse-flag,idsse-tracab,metrica")
    a = ap.parse_args()
    for sname in a.sets.split(","):
        games, labels, seeds = SETS[sname]
        runs = [r for r in calibrate.GAME_SETS[games] if r[1] in seeds]
        base = None
        print("\n== {} ({} runs, degraded tracking)".format(sname, len(runs)))
        print("{:22s} {:>8s} {:>8s} {:>9s} {:>6s} {:>12s}".format("variant", "spellF1", "macroF1", "poss", "D%", "runs up/down"))
        for vname, which in VARIANTS.items():
            with Pool(min(6, len(runs))) as pool:
                res = pool.map(job, [(r, labels, which) for r in runs])
            R = np.array(res)
            if base is None:
                base = R
            up = int(np.sum(R[:, 0] > base[:, 0] + 1e-9))
            dn = int(np.sum(R[:, 0] < base[:, 0] - 1e-9))
            print("{:22s} {:8.3f} {:8.3f} {:4.0f}/{:<4.0f} {:6.1f} {:>7d}/{:<4d}".format(
                vname, R[:, 0].mean(), R[:, 1].mean(), R[:, 2].mean(), R[:, 3].mean(), 100 * R[:, 4].mean(), up, dn),
                flush=True)


if __name__ == "__main__":
    main()
