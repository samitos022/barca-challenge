"""Why v2 misses a third of the possessions on the external full matches (IDSSE, Metrica).

External open data only, never BAR-ATH. For every true possession spell (maximal H or A run, as
the scorer defines it) of every run, the best same-team predicted spell (IoU) and what v2 said
inside it, then the misses broken down by duration, by what came before (the other team, or a
dead ball) and by what the tracking showed.

    .venv/bin/python analisi/possessi_persi.py [--cond clean|degraded] [--games idsse|metrica|all]
"""

import argparse
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
import possession_metrics as pm  # noqa: E402
import possession_v2 as v2  # noqa: E402

FPS = 25
H, A, D, X = 0, 1, 2, 3
BINS = ((0, 2), (2, 4), (4, 8), (8, 15), (15, 30), (30, 1e9))


def job(args):
    run, cond, extra, labels = args
    teams, players, ball, bs, blocks, gt = calibrate.case(run, cond, labels)
    p = {**v2.PARAMS, **extra}
    if run[0] in idsse.GAMES:
        p["KICKOFF_MAX_S"] = 0  # TRACAB halves start after the kick-off (bench_esterno --kickoff-fix)
    lab, f, r = v2.predict(teams, players, ball, bs, blocks, calibrate.pitch_of(run), p)
    pc = pm.label_codes(lab)
    runs = pm.valid_runs(gt)
    gs, ps = pm.possession_spells(gt, runs), pm.possession_spells(pc, runs)
    ov = pm.spell_overlaps(gs, ps)
    matched = pm.matched_gt_flags(ov, len(gs), 0.5)
    best = np.zeros(len(gs))
    for iou, i, j in ov:
        best[i] = max(best[i], iou)
    seen = ~np.isnan(ball[:, 0])
    tt = f["touch_team"]
    out = []
    seg = pm.segments(gt, runs)
    prev_of = {}
    for k, (a, b, c) in enumerate(seg):
        prev_of[a] = seg[k - 1][2] if k and seg[k - 1][1] == a else X
    for i, (a, b, c) in enumerate(gs):
        inside = pc[a:b]
        own = np.mean(inside == c)
        other = np.mean(inside == 1 - c)
        dead = np.mean(inside == D)
        # touches by the spell's team seen by v1, and its control
        n_touch = int(np.sum(tt[a:b] == c))
        ctrl = float(np.max(f["ctrl"][c][a:b])) if b > a else 0.0
        dn = np.nan_to_num(f["dist"][c][a:b], nan=np.inf)      # new team's closest player
        do = np.nan_to_num(f["dist"][1 - c][a:b], nan=np.inf)
        k = int(np.argmin(dn)) if b > a else 0
        mind = float(dn[k]) if b > a else np.inf
        spd = float(np.nan_to_num(f["speed"][a + k], nan=-1)) if b > a else -1
        margin = float(do[k] - dn[k]) if b > a else 0.0
        out.append(dict(dur=(b - a) / FPS, matched=bool(matched[i]), iou=best[i], own=own, other=other,
                        dead=dead, prev=prev_of.get(a, X), seen=float(seen[a:b].mean()),
                        touch=n_touch, ctrl=ctrl, start_seen=bool(seen[a]),
                        mind=mind, spd=spd, margin=margin))
    return run, out, len(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="degraded")
    ap.add_argument("--games", default="idsse")
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--labels", default="flag", help="IDSSE label variant: flag | events | tracab")
    a = ap.parse_args()
    seeds = tuple(int(s) for s in a.seeds.split(","))
    runs = [r for r in calibrate.GAME_SETS[a.games] if r[1] in seeds]
    with Pool(min(4, len(runs))) as pool:
        res = pool.map(job, [(r, a.cond, {}, a.labels) for r in runs])
    S = [s for _, out, _ in res for s in out]
    n_pred = sum(n for *_, n in res)
    print("cond {}, games {}, runs {}: true spells {}, predicted spells {}, matched at IoU 0.5 {} ({:.0f}%)".format(
        a.cond, a.games, len(runs), len(S), n_pred, sum(s["matched"] for s in S),
        100 * np.mean([s["matched"] for s in S])))

    def cls(s):
        if s["matched"]:
            return "matched"
        if s["iou"] >= 0.3:
            return "timing (IoU 0.3-0.5)"
        if s["own"] >= 0.5:
            return "own but split / long"
        if s["other"] >= s["dead"]:
            return "kept by the other team"
        return "called dead"

    print("\nby duration of the true spell: count, matched %, and how the missed ones were missed")
    for lo, hi in BINS:
        ss = [s for s in S if lo <= s["dur"] < hi]
        if not ss:
            continue
        c = Counter(cls(s) for s in ss)
        miss = len(ss) - c["matched"]
        print("  {:>4.0f}-{:<4.0f}s n {:4d} ({:4.1f}% of spells, {:4.1f}% of time)  matched {:3.0f}%  | ".format(
            lo, min(hi, 999), len(ss), 100 * len(ss) / len(S),
            100 * sum(s["dur"] for s in ss) / sum(s["dur"] for s in S), 100 * c["matched"] / len(ss)) +
            ", ".join("{} {:.0f}%".format(k, 100 * v / max(miss, 1)) for k, v in c.most_common() if k != "matched"))

    print("\nby what preceded the true spell")
    for prev, name in ((1 - 0, "other team (a turnover)"), (D, "dead ball (a restart)")):
        pass
    by_prev = defaultdict(list)
    for s in S:
        by_prev["restart (after D)" if s["prev"] == D else "turnover (after the other team)"
                if s["prev"] in (H, A) else "start of a run"].append(s)
    for k, ss in by_prev.items():
        c = Counter(cls(s) for s in ss)
        print("  {:32s} n {:5d} matched {:3.0f}%  {}".format(k, len(ss), 100 * c["matched"] / len(ss),
              ", ".join("{} {}".format(x, y) for x, y in c.most_common() if x != "matched")))

    print("\nmissed turnovers kept by the other team: what v1 saw of the new team")
    ss = [s for s in S if cls(s) == "kept by the other team"]
    ok = [s for s in S if s["matched"] and s["prev"] in (H, A)]
    for name, grp in (("missed", ss), ("matched turnovers", ok)):
        if grp:
            print("  {:18s} n {:5d}: duration median {:.1f} s, ball seen {:.0f}%, touches by the new team "
                  "median {:.0f} (none {:.0f}%), max control median {:.2f} (none {:.0f}%)".format(
                      name, len(grp), np.median([s["dur"] for s in grp]),
                      100 * np.mean([s["seen"] for s in grp]), np.median([s["touch"] for s in grp]),
                      100 * np.mean([s["touch"] == 0 for s in grp]), np.median([s["ctrl"] for s in grp]),
                      100 * np.mean([s["ctrl"] < 0.05 for s in grp])))
    geometry(S, cls)


def geometry(S, cls):
    """For the missed turnovers with no touch and no control: why v1 could not see them."""
    ss = [s for s in S if cls(s) == "kept by the other team" and s["touch"] == 0 and s["ctrl"] < 0.05]
    if not ss:
        return
    md = np.array([s["mind"] for s in ss])
    sp = np.array([s["spd"] for s in ss])
    mg = np.array([s["margin"] for s in ss])
    print("\nmissed turnovers with no touch and no control (n {}): closest approach of the new team".format(len(ss)))
    for lo, hi in ((0, 1), (1, 2), (2, 2.5), (2.5, 4), (4, 8), (8, 1e9)):
        m = (md >= lo) & (md < hi)
        print("   {:>4} - {:<4} m: {:3.0f}%   ball there median {:4.1f} m/s, fast (> 8 m/s) {:3.0f}%, opponent "
              "within 1 m of him too {:3.0f}%".format(lo, hi if hi < 1e8 else "inf", 100 * m.mean(),
                                                     np.median(sp[m]) if m.any() else np.nan,
                                                     100 * np.mean(sp[m] > 8) if m.any() else 0,
                                                     100 * np.mean(mg[m] < 1) if m.any() else 0))


if __name__ == "__main__":
    main()
