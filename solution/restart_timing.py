"""How late (or early) does v2 resume play after each kind of stoppage? Measured on the external
games, where the events say when every restart happened and what it was.

For every restart event in the external labels (the first frame after a dead-ball run), the
predicted restart is the first non-D frame within +-RANGE_S of it; latency = predicted - true.

    .venv/bin/python solution/restart_timing.py
"""

import os
import sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "external"))
import calibrate  # noqa: E402
import metrica  # noqa: E402
import possession_v2 as v2  # noqa: E402

FPS = 25
RANGE_S = 15.0


def kind_of(ev, frame):
    """Type of the restart event that starts at frame, from the event list."""
    for e in ev:
        if e[3] == frame:
            if e[1] == "SET PIECE":
                return sorted(e[2])[0] if e[2] else "SET PIECE"
            if "GOAL KICK" in e[2]:
                return "GOAL KICK"
            return "other (" + e[1] + ")"
    return "unknown"


def restarts(run, params=None):
    teams, players, ball, ball_speed, blocks, gt = calibrate.case(run)
    labels, _, _ = v2.predict(teams, players, ball, ball_speed, blocks, metrica.PITCH,
                              params or v2.PARAMS)
    ev = metrica.events(run[0])
    D = 2  # label code of D
    out = []
    dead = gt == D
    starts = np.flatnonzero(dead[:-1] & ~dead[1:]) + 1  # first alive frame after a GT dead run
    rng = int(RANGE_S * FPS)
    for f in starts:
        if gt[f] == 3:  # end of the graded block, not a restart
            continue
        lo, hi = max(0, f - rng), min(len(gt), f + rng)
        pd = labels[lo:hi] == "D"
        # predicted restart: the end of the predicted dead run that overlaps the true one
        if not pd.any():
            out.append((kind_of(ev, f), None))
            continue
        k = f - lo
        if pd[min(k, len(pd) - 1)] or (k > 0 and pd[k - 1]):
            j = k
            while j < len(pd) and pd[j]:
                j += 1
            out.append((kind_of(ev, f), (lo + j - f) / FPS))
        else:
            # predicted dead ended earlier: last predicted D before f
            before = np.flatnonzero(pd[:k])
            out.append((kind_of(ev, f), (lo + before[-1] + 1 - f) / FPS if len(before) else None))
    return out


def main():
    with Pool(len(calibrate.RUNS)) as pool:
        res = pool.map(restarts, calibrate.RUNS)
    by = defaultdict(list)
    for r in res:
        for kind, lat in r:
            by[kind].append(lat)
    print("{:<22} {:>4}  {:>7}  {:>7}  {:>7}  {:>6}".format(
        "restart", "n", "median", "p25", "p75", "missed"))
    for kind, lats in sorted(by.items(), key=lambda x: -len(x[1])):
        x = np.array([v for v in lats if v is not None])
        miss = sum(v is None for v in lats) / len(lats)
        if len(x):
            print("{:<22} {:>4}  {:>+6.1f}s  {:>+6.1f}s  {:>+6.1f}s  {:>5.0f}%".format(
                kind, len(lats), np.median(x), *np.percentile(x, [25, 75]), 100 * miss))


if __name__ == "__main__":
    main()
