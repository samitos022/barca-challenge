"""BAR-ATH with the teams re-assigned (team_reassign.py): possession_v2 and the learnt model
(possession_ml) on each tracking source, side by side. Aggregates only; the public five minutes
are a sanity check (17 possessions), never used to choose anything. No file is written.

    .venv/bin/python analisi/riassegna_bar_ath.py [--sources tactical,tactical_clean,fused,fused_reassigned]
"""

import argparse
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "starter", "solution", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, d))
from games import GAME, GAMES, half_video_blocks  # noqa: E402
import possession_metrics as pm  # noqa: E402
import possession_ml as ml  # noqa: E402
import possession_v2 as v2  # noqa: E402
import sync_broadcast  # noqa: E402
import team_reassign as tr  # noqa: E402
import tracking  # noqa: E402
import whistle  # noqa: E402

SOURCES = "tactical,tactical_nofrozen,tactical_reassigned,tactical_clean,fused,fused_reassigned"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=SOURCES)
    a = ap.parse_args()
    blocks = half_video_blocks(GAME)
    pitch = GAMES[GAME]["pitch"]
    events, level = whistle.load(GAME)
    gt = pm.label_codes(pm.read_label_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")))
    with open(ml.MODEL, "rb") as fh:
        M = pickle.load(fh)
    print("{:20s} {:>8s} {:>8s} {:>9s} {:>11s}   {:>8s} {:>8s} {:>9s}".format(
        "source", "v2 spell", "macroF1", "poss", "D% h1/h2", "ml spell", "macroF1", "poss"))
    for src in a.sources.split(","):
        fused = src.startswith("fused")
        teams, pl, ball, bs, _ = tracking.load(GAME, "fused" if fused else src)
        if src == "fused_reassigned":
            teams, pl = tr.reassign(teams, pl, ball, blocks, pitch, tr.load_model())
        cut = None
        if fused:
            sync = np.load(sync_broadcast.OUT)
            cut = (sync["b_of_t"] >= 0) & (sync["b_live"] < 0)
        labels, f, r = v2.predict(teams, pl, ball, bs, blocks, pitch, v2.PARAMS, events, level, cut)
        codes = pm.label_codes(labels)
        Xf, fn = ml.features(teams, pl, ball, bs, blocks, pitch)
        out = ml.decode(ml.prob_home(M["clf"], Xf, fn), codes, M["switch_cost"])
        row = []
        for c in (codes, out):
            res = pm.evaluate(gt, c, bootstrap_n=0)
            poss = []
            for _, s0, z0 in blocks:
                seg = c[s0:z0 + 1]
                starts = np.concatenate(([0], np.flatnonzero(seg[1:] != seg[:-1]) + 1))
                poss.append(int(np.sum(seg[starts] < 2)))
            row.append((res["spell_f1_mean"], res["macro_f1"], poss))
        dsh = [100 * np.mean(codes[s0:z0 + 1] == 2) for _, s0, z0 in blocks]
        print("{:20s} {:8.4f} {:8.4f} {:>4d}/{:<4d} {:5.1f}/{:<5.1f}   {:8.4f} {:8.4f} {:>4d}/{:<4d}".format(
            src, row[0][0], row[0][1], *row[0][2], *dsh, row[1][0], row[1][1], *row[1][2]), flush=True)


if __name__ == "__main__":
    main()
