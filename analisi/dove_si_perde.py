"""Where v2 loses its spellF1 on the external full matches (never BAR-ATH).

1. Tracking: the same v2 on clean -> players degraded -> + ball gaps -> + BAR-ATH ball shape.
2. Oracles on the realistic tracking, each fixing ONE thing with the true labels:
   - dead ball: D exactly where the truth has D; frames v2 called D but are alive take v2's
     nearest team decision;
   - team: v2's own dead / alive split, but the true team on the frames both call alive;
   - both: the truth (sanity, 1.0).
   The gap between v2 and an oracle is the most that fixing that part could gain.

    .venv/bin/python analisi/dove_si_perde.py
"""

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

H, A, D, X = 0, 1, 2, 3
SETS = {"DFL flag": ("idsse", "flag", (1,)), "DFL tracab": ("idsse", "tracab", (1,)), "Metrica": ("metrica", None, (1, 2))}
CONDS = ("clean", "players", "degraded", "realistic")


def ffill_team(codes):
    """Every frame -> the last H/A decision at or before it (else the first after it)."""
    idx = np.where(np.isin(codes, (H, A)), np.arange(len(codes)), -1)
    last = np.maximum.accumulate(idx)
    first = np.flatnonzero(idx >= 0)
    last[last < 0] = first[0] if len(first) else 0
    return codes[last]


def job(args):
    run, cond, labels = args
    teams, players, ball, bs, blocks, gt = calibrate.case(run, cond, labels)
    p = dict(v2.PARAMS)
    if run[0] in idsse.GAMES:
        p["KICKOFF_MAX_S"] = 0
    lab, _, _ = v2.predict(teams, players, ball, bs, blocks, calibrate.pitch_of(run), p)
    pc = pm.label_codes(lab)
    out = {"v2": pm.evaluate(gt, pc, bootstrap_n=0)["spell_f1_mean"]}
    if cond == "realistic":
        g = gt != X
        team = ffill_team(pc)
        o_dead = np.where(gt == D, D, np.where(pc == D, team, pc))
        o_dead[~g] = X
        o_team = pc.copy()
        both_alive = np.isin(gt, (H, A)) & np.isin(pc, (H, A))
        o_team[both_alive] = gt[both_alive]
        out["oracle dead"] = pm.evaluate(gt, o_dead, bootstrap_n=0)["spell_f1_mean"]
        out["oracle team"] = pm.evaluate(gt, o_team, bootstrap_n=0)["spell_f1_mean"]
    return out


def main():
    for sname, (games, labels, seeds) in SETS.items():
        runs = [r for r in calibrate.GAME_SETS[games] if r[1] in seeds]
        row = {}
        for cond in CONDS:
            if games == "metrica" and cond == "players":
                continue  # Metrica's ball already has BAR-ATH-like gaps: players == degraded
            with Pool(min(7, len(runs))) as pool:
                R = pool.map(job, [(r, cond, labels) for r in runs])
            for k in R[0]:
                row[(cond, k)] = np.mean([x[k] for x in R])
        print("\n== {}".format(sname))
        print("  tracking:  " + "  ->  ".join("{} {:.3f}".format(c, row[(c, "v2")]) for c in CONDS if (c, "v2") in row))
        print("  oracles on realistic: v2 {:.3f}   perfect dead ball {:.3f} (+{:.3f})   perfect team {:.3f} (+{:.3f})".format(
            row[("realistic", "v2")], row[("realistic", "oracle dead")],
            row[("realistic", "oracle dead")] - row[("realistic", "v2")], row[("realistic", "oracle team")],
            row[("realistic", "oracle team")] - row[("realistic", "v2")]), flush=True)


if __name__ == "__main__":
    main()
