"""Ball gaps that start near a penalty area, on the external matches: what the truth is inside
them (dead ball, the defending team in possession = typically the keeper holding the ball, the
attacking team), by where the ball is seen again, by length and by what the players do.

Used to design the "keeper holds the ball" rule from data of other matches, never from BAR-ATH.

    .venv/bin/python analisi/buchi_area.py [--cond realistic|keeper]

--cond keeper: IDSSE only, with the ball also removed while a keeper holds it (as BAR-ATH's tracker
does, degrade.hide_keeper_holds): the gaps a "keeper holds the ball" rule has to recognise.
"""

import argparse
import itertools
import os
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "solution"), os.path.join(ROOT, "solution", "external"),
          os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "starter")):
    sys.path.insert(0, p)
import calibrate  # noqa: E402
import idsse  # noqa: E402
import metrica  # noqa: E402
from track_fixes import own_goals  # noqa: E402

FPS = 25
H, A, D, X = 0, 1, 2, 3
MIN_GAP_S = 2.0
BOX_X, BOX_Y, GA_X, GA_Y = 16.5, 20.16, 5.5, 9.16
MARGIN = 5.0


def where(bx, by, gxs, pitch):
    """Class of a ball position with respect to the goal at x = gxs."""
    L, W = pitch
    dx, dy = abs(bx - gxs), abs(by - W / 2)
    if dx <= 4.0 and min(by, W - by) <= 4.0:
        return "corner"
    if dx <= GA_X + 1.0 and dy <= GA_Y + 1.0:
        return "goal area"
    if dx <= BOX_X and dy <= BOX_Y:
        return "box"
    if np.hypot(bx - L / 2, by - W / 2) <= 3.0:
        return "centre"
    if min(by, W - by) <= 2.0 or min(bx, L - bx) <= 2.0:
        return "line"
    return "field"


def job(args):
    run, labels, cond = args
    teams, players, ball, bs, blocks, gt = calibrate.case(run, cond, labels)
    pitch = calibrate.pitch_of(run)
    L, W = pitch
    vis = ~np.isnan(ball[:, 0])
    vel = np.linalg.norm(np.gradient(players.astype(np.float64), axis=0), axis=2) * FPS
    with np.errstate(invalid="ignore"):
        pace = np.nanmean(vel, axis=1)
    rows = []
    for _, a0, z0 in blocks:
        gx = own_goals(teams, players, a0, z0, pitch)
        e = np.flatnonzero(np.diff(np.concatenate(([0], (~vis[a0:z0 + 1]).astype(np.int8), [0]))))
        for a, b in zip(a0 + e[::2], a0 + e[1::2]):
            if b - a < MIN_GAP_S * FPS or a == a0 or b > z0:
                continue
            lx, ly = ball[a - 1]
            side = 0 if abs(lx - gx[0]) < abs(lx - gx[1]) else 1      # team defending the near goal
            g = gx[side]
            if not (abs(lx - g) <= BOX_X + MARGIN and abs(ly - W / 2) <= BOX_Y + MARGIN):
                continue
            t = gt[a:b]
            t = t[t != X]
            if len(t) == 0:
                continue
            sh = [np.mean(t == D), np.mean(t == side), np.mean(t == 1 - side)]
            truth = ("dead", "defending team", "attacking team")[int(np.argmax(sh))]
            # players inside the box: at the start and at the end of the gap, per team
            def inbox(f, k):
                P = players[f][teams == k]
                return int(np.sum((np.abs(P[:, 0] - g) <= BOX_X) & (np.abs(P[:, 1] - W / 2) <= BOX_Y)))
            # the deepest defender (keeper) at the end of the gap, and the ball's distance from him
            Pd = players[b][teams == side]
            dep = np.abs(Pd[:, 0] - g)
            kd = np.nan
            if np.isfinite(dep).any():
                k = int(np.nanargmin(dep))
                kd = float(np.hypot(*(Pd[k] - ball[b])))
            rows.append({"dur": (b - a) / FPS, "back": where(ball[b, 0], ball[b, 1], g, pitch), "truth": truth,
                         "last": where(lx, ly, g, pitch), "pace": float(np.nanmean(pace[a:b])),
                         "att_in0": inbox(a, 1 - side), "att_in1": inbox(b, 1 - side),
                         "last_gl": float(abs(lx - g)), "last_tl": float(min(ly, W - ly)),
                         "inbox_last": bool(abs(lx - g) <= BOX_X and abs(ly - W / 2) <= BOX_Y),
                         "keeper_d": kd, "idsse": run[0] in idsse.GAMES})
    return rows


def table(rows, key, title):
    print("\n" + title)
    cnt = defaultdict(Counter)
    for r in rows:
        cnt[key(r)][r["truth"]] += 1
    for k in sorted(cnt, key=lambda k: -sum(cnt[k].values())):
        c = cnt[k]
        n = sum(c.values())
        print("  {:22s} n {:4d}   dead {:3.0f}%   defending team {:3.0f}%   attacking team {:3.0f}%".format(
            str(k), n, 100 * c["dead"] / n, 100 * c["defending team"] / n, 100 * c["attacking team"] / n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="realistic", choices=("realistic", "keeper"))
    cond = ap.parse_args().cond
    runs = [((g, s), "tracab", cond) for g in idsse.GAMES for s in (1, 2)]
    if cond == "realistic":
        runs += [((g, s), None, cond) for g in metrica.GAMES for s in (1, 2)]
    with Pool(4) as pool:
        rows = [r for rr in pool.map(job, runs) for r in rr]
    print("condition: {}".format(cond))
    print("{} gaps of {} s or more that start within {} m of a penalty area ({} runs)".format(
        len(rows), MIN_GAP_S, MARGIN, len(runs)))
    table(rows, lambda r: "all", "all")
    table(rows, lambda r: r["back"], "by where the ball is seen again")
    table(rows, lambda r: "{:>3s} s".format(str(next(e for e in (5, 8, 12, 20, 40, 999) if r["dur"] < e))),
          "by length (upper bound)")
    table(rows, lambda r: (r["back"] in ("corner", "goal area", "centre", "line"),
                           next(e for e in (8, 15, 999) if r["dur"] < e)),
          "set-piece spot at the return (True/False) x length (upper bound)")
    table(rows, lambda r: "pace < {:.1f}".format(next(e for e in (0.8, 1.2, 1.6, 2.0, 9.9) if r["pace"] < e)),
          "by mean pace of the players in the gap (m/s)")
    table(rows, lambda r: "attackers in box {:+d}".format(int(np.clip(r["att_in1"] - r["att_in0"], -3, 3))),
          "by change of attackers inside the box, start to end of the gap")
    table(rows, lambda r: "ball-keeper < {:g} m".format(next(e for e in (3, 8, 20, 999) if not r["keeper_d"] >= e)),
          "by distance of the returning ball from the deepest defender")
    table(rows, lambda r: "last seen < {:g} m from the goal line".format(next(e for e in (1, 2, 3, 6, 99) if r["last_gl"] < e)),
          "by distance of the last seen ball from the goal line")
    spot = lambda r: r["back"] in ("corner", "goal area", "centre", "line")  # noqa: E731
    cnt = defaultdict(Counter)
    for P, T in itertools.product((1.0, 1.2, 1.4, 1.55), (12, 20, 30)):
        for r in rows:
            if not spot(r) and r["pace"] >= P and r["dur"] <= T:
                cnt[(P, T)][r["truth"]] += 1
    print("\ncandidate rule: ball back away from a set-piece spot, pace >= P, length <= T")
    for (P, T), c in cnt.items():
        n = sum(c.values())
        print("  pace >= {:.2f}, length <= {:2d} s   n {:4d}   dead {:3.0f}%   defending team {:3.0f}%   attacking team {:3.0f}%".format(
            P, T, n, 100 * c["dead"] / n, 100 * c["defending team"] / n, 100 * c["attacking team"] / n))
    tot = Counter(r["truth"] for r in rows)
    print("  (all gaps: {})".format(dict(tot)))
    for t in ("dead", "defending team", "attacking team"):
        d = np.array([r["dur"] for r in rows if r["truth"] == t])
        pc = np.array([r["pace"] for r in rows if r["truth"] == t])
        if len(d):
            print("{:15s} length s p25/50/75 {:.1f} {:.1f} {:.1f}   pace p25/50/75 {:.2f} {:.2f} {:.2f}".format(
                t, *np.percentile(d, [25, 50, 75]), *np.nanpercentile(pc, [25, 50, 75])))


if __name__ == "__main__":
    main()
