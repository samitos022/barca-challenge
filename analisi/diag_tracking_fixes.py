"""possession_v2.predict on each tracking source of tracking.py (fixes alone and together).

Aggregates only: possessions and dead-ball share per half, rule-8 gaps, the public 5-minute
spellF1 / macroF1 (sanity check only: 17 possessions, noise about +-0.33, never used to choose),
and where the recovered ball falls (live play vs players walking). possession_v2.py is not
modified: predict() is called from here, like analisi/diag_fusione.py does.

    .venv/bin/python analisi/diag_tracking_fixes.py [--sources tactical,tactical_ball,...]
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "starter", "solution", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, d))
from games import GAME, GAMES, half_video_blocks  # noqa: E402
import possession_metrics as pm  # noqa: E402
import possession_v2 as v2  # noqa: E402
import tracking  # noqa: E402
import whistle  # noqa: E402

FPS = 25
SOURCES = ("tactical", "tactical_keepers", "tactical_teams", "tactical_nofrozen", "tactical_ball", "tactical_fixed")


def runs(mask):
    e = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return e[::2], e[1::2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=",".join(SOURCES))
    a = ap.parse_args()
    blocks = half_video_blocks(GAME)
    pitch = GAMES[GAME]["pitch"]
    events, level = whistle.load(GAME)
    gt = pm.label_codes(pm.read_label_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")))
    ref, ref_gaps, ref_pspeed, ref_ball = None, None, None, None
    print("{:18s} {:>7s} {:>7s} {:>9s} {:>9s} {:>7s} {:>7s} {:>6s} {:>6s}".format(
        "source", "spellF1", "macroF1", "poss h1", "poss h2", "D% h1", "D% h2", "rule8", "=tact%"))
    for src in a.sources.split(","):
        teams, pl, ball, bs, _ = tracking.load(GAME, src)
        labels, f, r = v2.predict(teams, pl, ball, bs, blocks, pitch, v2.PARAMS, events, level)
        res = pm.evaluate(gt, pm.label_codes(labels), bootstrap_n=0)
        poss, dsh = [], []
        g = np.zeros(len(ball), bool)
        for _, s0, z0 in blocks:
            seg = labels[s0:z0 + 1]
            g[s0:z0 + 1] = True
            cut = np.flatnonzero(seg[1:] != seg[:-1]) + 1
            starts = np.concatenate(([0], cut))
            poss.append(int(sum(seg[s] in "HA" for s in starts)))
            dsh.append(100 * np.mean(seg == "D"))
        if ref is None:
            ref, ref_gaps, ref_pspeed, ref_ball = labels, f["gap_why"], f["pspeed"], ball
        same = 100 * np.mean(labels[g] == ref[g])
        print("{:18s} {:7.4f} {:7.4f} {:9d} {:9d} {:7.1f} {:7.1f} {:6d} {:6.1f}".format(
            src, res["spell_f1_mean"], res["macro_f1"], poss[0], poss[1], dsh[0], dsh[1], len(f["gap_why"]), same), flush=True)
        if src in ("tactical_ball", "tactical_fixed") and ref_ball is not None:
            ball_report(ref_ball, ball, ref_pspeed, ref_gaps, g)


def ball_report(atd, ball, pspeed, gaps8, g):
    """Where the recovered ball falls: ATD gaps by players' mean pace; rule-8 gaps it splits."""
    rec = np.isnan(atd[:, 0]) & ~np.isnan(ball[:, 0]) & g
    st, en = runs(np.isnan(atd[:, 0]) & g)
    live = walk = mid = 0
    gl = {"live": [0, 0], "walk": [0, 0], "mid": [0, 0]}   # gap frames, recovered frames
    slow = (v2.PARAMS["SPD_PLAY"] + v2.PARAMS["SPD_DEAD"]) / 2
    for a, b in zip(st, en):
        sp = np.nanmean(pspeed[a:b]) if np.any(~np.isnan(pspeed[a:b])) else np.nan
        k = "live" if sp > 1.8 else ("walk" if sp < slow else "mid")
        gl[k][0] += b - a
        gl[k][1] += int(rec[a:b].sum())
    tot = max(sum(v[1] for v in gl.values()), 1)
    print("   recovered ball: {} graded frames ({:.1f}% of the ATD gap frames); by gap pace: live (> 1.8 m/s) {:.0f}% of "
          "the recovered ({:.1f}% of those gaps' frames), walking (< {:.2f}) {:.0f}% ({:.1f}%), between {:.0f}% ({:.1f}%)".format(
              int(rec.sum()), 100 * rec.sum() / max(sum(b - a for a, b in zip(st, en)), 1),
              100 * gl["live"][1] / tot, 100 * gl["live"][1] / max(gl["live"][0], 1), slow,
              100 * gl["walk"][1] / tot, 100 * gl["walk"][1] / max(gl["walk"][0], 1),
              100 * gl["mid"][1] / tot, 100 * gl["mid"][1] / max(gl["mid"][0], 1)))
    hit = [(a, b) for a, b, _ in gaps8 if rec[a:b].any()]
    longest = []
    for a, b in hit:
        sub_s, sub_e = runs(~rec[a:b])
        longest.append(max(sub_e - sub_s) / FPS if len(sub_s) else 0)
    longest = np.array(longest)
    print("   rule-8 gaps of the tactical run: {}; with a recovered ball inside {} (split); of those, the longest "
          "remaining piece still >= {:.0f} s in {}".format(len(gaps8), len(hit), v2.PARAMS["GAP_DEAD_S"],
                                                         int(np.sum(longest >= v2.PARAMS["GAP_DEAD_S"]))))


if __name__ == "__main__":
    main()
