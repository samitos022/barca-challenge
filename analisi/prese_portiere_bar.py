"""BAR-ATH, no labels: the ball gaps that look like a keeper holding the ball (last seen within
KEEPER_D m of a keeper, inside his box, gap of 1-15 s), and how two versions of the output label them.

    .venv/bin/python analisi/prese_portiere_bar.py <before.csv> <after.csv> [tracking source]
"""

import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "starter", "solution", "solution/external", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, d))
import keeper_signals  # noqa: E402
import tracking  # noqa: E402
from games import GAME, GAMES, half_video_blocks  # noqa: E402

KEEPER_D = 5.0
CODE = {"H": 0, "A": 1, "D": 2, "X": 3}


def labels(path):
    with open(path) as h:
        r = csv.reader(h)
        next(r)
        return np.array([CODE[row[1]] for row in r], dtype=np.int8)


def main(before, after, source="tactical_clean"):
    teams, players, ball, _, _ = tracking.load(GAME, source)
    blocks = half_video_blocks(GAME)
    F, names = keeper_signals.features(teams, players, ball, blocks, GAMES[GAME]["pitch"])
    c = {nm: F[:, i] for i, nm in enumerate(names)}
    gap = np.nan_to_num(c["kp_in_gap"]) > 0
    glen = np.nan_to_num(c["kp_gap_elapsed_s"]) + np.nan_to_num(c["kp_gap_left_s"])
    hold = np.zeros(len(ball), dtype=bool)
    for side in ("home", "away"):
        hold |= (np.nan_to_num(c["kp_vanish_d_" + side], nan=99) < KEEPER_D) & \
                (np.nan_to_num(c["kp_vanish_box_" + side]) > 0)
    hold &= gap & (glen >= 1) & (glen <= 15)
    graded = np.zeros(len(ball), dtype=bool)
    for _, a, z in blocks:
        graded[a:z + 1] = True
    hold &= graded
    e = np.flatnonzero(np.diff(np.concatenate(([0], hold.astype(np.int8), [0]))))
    print("probable keeper holds: {} gaps, {:.0f} s".format(len(e) // 2, hold.sum() / 25))
    for name, path in (("before", before), ("after", after)):
        lab = labels(path)
        dead = lab[hold] == 2
        whole = sum(np.all(lab[a:b] == 2) for a, b in zip(e[::2], e[1::2]))
        d_all = np.mean(lab[graded & (lab != 3)] == 2)
        print("  {:6s} hold frames dead {:4.1f}% | holds wholly dead {:2d} | dead ball on the whole match {:4.1f}%".format(
            name, 100 * dead.mean(), whole, 100 * d_all))


if __name__ == "__main__":
    main(*sys.argv[1:])
