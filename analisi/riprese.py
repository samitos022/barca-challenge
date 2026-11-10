"""Restart timing on the external full matches (never BAR-ATH): which v2 mechanism dates which
restart, and how wrong. Also what the clean tracking shows around the true restart instant.

A true restart is a GT transition D -> H/A. Its predicted counterpart is the predicted D -> H/A
transition closest in time within MATCH_S. The reason is the one v2 recorded for that change.

    .venv/bin/python analisi/riprese.py [--cond degraded] [--labels tracab]
"""

import argparse
import os
import sys
from collections import defaultdict
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

FPS = 25
MATCH_S = 8.0


def job(args):
    run, cond, labels, extra = args
    teams, players, ball, bs, blocks, gt = calibrate.case(run, cond, labels)
    p = {**v2.PARAMS, **extra}
    if run[0] in idsse.GAMES:
        p["KICKOFF_MAX_S"] = 0
    lab, f, r = v2.predict(teams, players, ball, bs, blocks, calibrate.pitch_of(run), p)
    pc = pm.label_codes(lab)
    runs = pm.valid_runs(gt)
    gtr = [(t, b) for t, a, b in pm.transitions(gt, runs) if a == 2 and b in (0, 1)]
    prr = [(t, b) for t, a, b in pm.transitions(pc, runs) if a == 2 and b in (0, 1)]
    reason_at = {}
    for t_dec, start, new, reason in r["changes"]:
        if new in (0, 1):
            reason_at[start] = reason
    pt = np.array([t for t, _ in prr]) if prr else np.zeros(0, int)
    out = []
    seen = ~np.isnan(ball[:, 0])
    _, speed, _, _ = v1.ball_kinematics(ball, bs, p)
    for t, team in gtr:
        if len(pt):
            k = int(np.argmin(np.abs(pt - t)))
            if abs(pt[k] - t) <= MATCH_S * FPS:
                ptk, pteam = prr[k]
                out.append(((ptk - t) / FPS, reason_at.get(ptk, "?"), pteam == team, bool(seen[t]),
                            float(np.nan_to_num(speed[t], nan=-1))))
                continue
        out.append((np.nan, "missed", False, bool(seen[t]), float(np.nan_to_num(speed[t], nan=-1))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="degraded")
    ap.add_argument("--labels", default="tracab")
    ap.add_argument("--games", default="idsse")
    ap.add_argument("--set", default="", help="PARAMS overrides, e.g. MOVE_LAST=1,LOOK_S=1.5")
    a = ap.parse_args()
    extra = {k: float(v) for k, v in (kv.split("=") for kv in a.set.split(",") if kv)}
    seeds = (1,) if a.games == "idsse" else (1, 2)
    runs = [r for r in calibrate.GAME_SETS[a.games] if r[1] in seeds]
    with Pool(min(7, len(runs))) as pool:
        R = [x for res in pool.map(job, [(r, a.cond, a.labels, extra) for r in runs]) for x in res]
    print("true restarts {} ({} {} tracking)".format(len(R), a.games, a.cond))
    by = defaultdict(list)
    for off, reason, ok, seen, sp in R:
        by[reason].append((off, ok, seen))
    for reason, rows in sorted(by.items(), key=lambda kv: -len(kv[1])):
        o = np.array([x[0] for x in rows], float)
        if reason == "missed":
            print("  {:24s} n {:4d} ({:3.0f}%)  no predicted restart within {:.0f} s".format(
                reason, len(rows), 100 * len(rows) / len(R), MATCH_S))
            continue
        print("  {:24s} n {:4d} ({:3.0f}%)  offset median {:+5.2f} s  IQR {:+5.2f}..{:+5.2f}  within 1 s {:3.0f}%  "
              "right team {:3.0f}%  ball seen at the true restart {:3.0f}%".format(
                  reason, len(rows), 100 * len(rows) / len(R), np.median(o), np.percentile(o, 25),
                  np.percentile(o, 75), 100 * np.mean(np.abs(o) < 1), 100 * np.mean([x[1] for x in rows]),
                  100 * np.mean([x[2] for x in rows])))


if __name__ == "__main__":
    main()
