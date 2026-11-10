"""What v1's touch detector sees at the true changes of possession (external games, clean tracking).

For every true turnover (H -> A or A -> H with no dead ball between) of the IDSSE matches: the
v1 touches within +-WIN frames, their attribution, the largest velocity change there, and the
distances of both teams to the ball. Aggregate numbers only.

    .venv/bin/python analisi/tocchi_ai_cambi.py [--labels tracab] [--cond clean]
"""

import argparse
import os
import sys
from collections import Counter
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "solution"), os.path.join(ROOT, "solution", "external"),
          os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "starter")):
    sys.path.insert(0, p)
import calibrate  # noqa: E402
import possession_metrics as pm  # noqa: E402
import possession_v1 as v1  # noqa: E402
import possession_v2 as v2  # noqa: E402

WIN = 12  # frames each side of the labelled change (0.5 s)


def job(args):
    run, cond, labels = args
    teams, players, ball, bs, blocks, gt = calibrate.case(run, cond, labels)
    p = v2.PARAMS
    vel, speed, dv, angle = v1.ball_kinematics(ball, bs, p)
    dist, _ = v1.team_distances(players, ball, teams)
    tt, is_t = v1.touches(dv, dist, p)
    out = []
    for t, a, b in pm.transitions(gt, pm.valid_runs(gt)):
        if a not in (0, 1) or b not in (0, 1):
            continue
        lo, hi = max(0, t - WIN), min(len(ball), t + WIN + 1)
        k_t = np.flatnonzero(is_t[lo:hi]) + lo
        new_t = int(np.sum(tt[lo:hi] == b))
        old_t = int(np.sum(tt[lo:hi] == a))
        none_t = int(np.sum(is_t[lo:hi] & (tt[lo:hi] < 0)))
        dmax = float(np.nanmax(np.concatenate(([0.0], dv[lo:hi]))))
        dn = np.nan_to_num(dist[b, lo:hi], nan=np.inf).min()
        do = np.nan_to_num(dist[a, lo:hi], nan=np.inf).min()
        smax = float(np.nanmax(np.concatenate(([0.0], speed[lo:hi]))))
        out.append((len(k_t), new_t, old_t, none_t, dmax, dn, do, smax))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="tracab")
    ap.add_argument("--cond", default="clean")
    a = ap.parse_args()
    runs = [r for r in calibrate.GAME_SETS["idsse"] if r[1] == 1]
    with Pool(len(runs)) as pool:
        R = np.array([x for res in pool.map(job, [(r, a.cond, a.labels) for r in runs]) for x in res])
    n = len(R)
    print("true turnovers: {} ({} labels, {} tracking); window +-{} frames".format(n, a.labels, a.cond, WIN))
    print("  some v1 touch in the window {:.0f}%: by the NEW team {:.0f}%, only by the old team {:.0f}%, "
          "only unattributed (nobody within R_TOUCH) {:.0f}%".format(
              100 * np.mean(R[:, 0] > 0), 100 * np.mean(R[:, 1] > 0),
              100 * np.mean((R[:, 1] == 0) & (R[:, 2] > 0)),
              100 * np.mean((R[:, 0] > 0) & (R[:, 1] == 0) & (R[:, 2] == 0))))
    nt = R[R[:, 0] == 0]
    print("  no touch at all ({:.0f}%): largest velocity change median {:.1f} m/s (threshold {}), "
          "p75 {:.1f}; ball max speed median {:.1f} m/s".format(
              100 * len(nt) / n, np.median(nt[:, 4]), v2.PARAMS["DV_TOUCH"], np.percentile(nt[:, 4], 75),
              np.median(nt[:, 7])))
    for lo, hi in ((0, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 99)):
        m = (R[:, 4] >= lo) & (R[:, 4] < hi)
        print("    max dv {:>4}-{:<4}: {:3.0f}% of turnovers".format(lo, hi, 100 * m.mean()))
    print("  new team's closest approach in the window: < 1 m {:.0f}%, < 2 m {:.0f}%, < 3 m {:.0f}%; old team "
          "< 2 m {:.0f}%".format(100 * np.mean(R[:, 5] < 1), 100 * np.mean(R[:, 5] < 2),
                                100 * np.mean(R[:, 5] < 3), 100 * np.mean(R[:, 6] < 2)))
    print("  touch detector cadence: all frames with a touch (any team) per minute would be printed by "
          "bench; here touches per turnover window mean {:.2f}".format(R[:, 0].mean()))


if __name__ == "__main__":
    main()
