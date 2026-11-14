"""How the Smart Tagging of BAR-ATH relates to what our model already says, with aggregates only (no
ground truth is used: the public five minutes are left out of everything here).

For each code: how much of its intervals our model calls dead, and whether its Team tag is the team
our model gives the ball to (or the other one). For the set pieces: where in the interval our model
puts the restart, and whether the restart goes to the tagged team.

    .venv/bin/python analisi/smart_tagging_coerenza.py [submission csv]
"""

import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "starter"), os.path.join(ROOT, "evaluation")]
import load_smart_tagging as st  # noqa: E402
import possession_metrics as pm  # noqa: E402
from baseline import team_names  # noqa: E402
from games import GAME, require  # noqa: E402

PUBLIC = (11185, 18684)
FPS = 25


def main(path=os.path.join(ROOT, "evaluation", "results", "v2ml3_possession.csv")):
    pred = pm.label_codes(pm.read_label_csv(path))
    home, away = team_names(require(["atd_meta"], GAME)[0])
    tags = st.load(GAME)
    n = len(pred)
    keep = []
    for t in tags:
        a, b = max(0, t["start_frame"]), min(n, t["end_frame"])
        if b <= a or (a < PUBLIC[1] and b > PUBLIC[0]):
            continue                                   # skip anything touching the public minutes
        keep.append((t, a, b))
    print("{} of {} intervals used ({} touch the public five minutes and are left out)".format(len(keep), len(tags), len(tags) - len(keep)))
    print("teams in the tags: {} / home {} / away {}".format(sorted({str(t["tags"].get("Team")) for t, *_ in keep}), home, away))
    agg = defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0])   # n, seconds, dead share, same team share, other team share
    for t, a, b in keep:
        seg = pred[a:b]
        g = seg != 3
        if not g.any():
            continue
        team = str(t["tags"].get("Team"))
        k = 0 if team == home else (1 if team == away else -1)
        alive = g & (seg < 2)
        row = agg[t["code"]]
        row[0] += 1
        row[1] += (b - a) / FPS
        row[2] += np.mean(seg[g] == 2)
        if k >= 0 and alive.any():
            row[3] += np.mean(seg[alive] == k)
            row[4] += np.mean(seg[alive] == 1 - k)
    print("\n{:30s} {:>4s} {:>7s} {:>9s} {:>16s}".format("code", "n", "mean s", "dead %", "alive: tag team %"))
    for code, (k, sec, dsh, same, other) in sorted(agg.items(), key=lambda x: -x[1][0]):
        print("{:30s} {:4d} {:7.1f} {:8.0f}% {:15.0f}%".format(code, k, sec / k, 100 * dsh / k, 100 * same / max(same + other, 1e-9) * 1))
    # set pieces: our restart inside the interval, and its team
    print("\nset pieces (Type): our model's dead -> alive change inside the tagged interval, where, and to which team")
    rows = defaultdict(list)
    for t, a, b in keep:
        if t["code"].lower() != "set pieces":
            continue
        seg = pred[a:b]
        ch = np.flatnonzero((seg[1:] < 2) & (seg[:-1] == 2)) + 1
        typ = str(t["tags"].get("Type"))
        team = str(t["tags"].get("Team"))
        k = 0 if team == home else (1 if team == away else -1)
        if len(ch):
            c = ch[0]
            rows[typ].append((1, c / max(b - a, 1), (b - a) / FPS, int(seg[c] == k) if k >= 0 else -1))
        else:
            dead_before = np.any(pred[max(0, a - 10 * FPS):a] == 2)
            rows[typ].append((0, np.nan, (b - a) / FPS, -1 if not dead_before else -2))
    for typ, v in rows.items():
        v = np.array(v, dtype=float)
        has = v[:, 0] == 1
        same = v[has, 3]
        print("  {:22s} n {:3d} | interval {:4.1f} s | a restart inside {:3.0f}% (at {:3.0f}% of the interval, median) | restart to the tagged team {:3.0f}% of those with a team".format(
            typ, len(v), np.mean(v[:, 2]), 100 * has.mean(), 100 * np.nanmedian(v[has, 1]) if has.any() else np.nan,
            100 * np.mean(same[same >= 0] == 1) if np.any(same >= 0) else np.nan))
    cover = np.zeros(n, bool)
    for t, a, b in keep:
        if t["code"].lower() == "set pieces":
            cover[a:b] = True
    dd = np.flatnonzero((pred[1:] < 2) & (pred[:-1] == 2)) + 1
    dd = dd[(dd < PUBLIC[0]) | (dd > PUBLIC[1])]
    print("\nour model's restarts: {}; inside a tagged set piece: {:.0f}%".format(len(dd), 100 * np.mean(cover[dd])))


if __name__ == "__main__":
    main(*sys.argv[1:2])
