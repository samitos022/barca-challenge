"""Which part of the broadcast fusion costs: ablation of the fused tracking, aggregate numbers only.

Runs the team's possession_v2.predict() on tracking variants built in memory from the tactical and
the fused caches. Prints, per variant: spellF1 / macroF1 on the public 5 minutes (control only),
possessions and dead-ball share on the whole match, touches, ball jumps, agreement with the
tactical-only labels. Nothing frame-level is printed.

    .venv/bin/python diag_fusione.py [--no-whistle]
"""

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
NT = 46


def main(use_whistle=True):
    t_teams, t_pl, t_ball, t_bs, _ = tracking.load(GAME)
    fz = np.load(os.path.join(ROOT, "data", "derived", "{}_fused_tracking.npz".format(GAME)))
    f_teams, f_pl, f_ball, f_bs, src = fz["teams"], fz["players"], fz["ball"], fz["ball_speed"], fz["ball_src"]
    blocks = half_video_blocks(GAME)
    pitch = GAMES[GAME]["pitch"]
    events, level = whistle.load(GAME) if use_whistle else ((), None)
    n = len(t_ball)
    graded = np.zeros(n, bool)
    for _, a, z in blocks:
        graded[a:z + 1] = True

    gt = pm.label_codes(pm.read_label_csv(os.path.join(ROOT, "evaluation", "gt",
                                                       "BAR-ATH_possession_gt_public.csv")))

    # ball variants
    BOTH, BRO = 3, 2
    fill_ball, fill_bs = t_ball.copy(), t_bs.copy()
    m = src == BRO
    fill_ball[m], fill_bs[m] = f_ball[m], f_bs[m]
    # players: tactical slots blended, extras unblended
    blend_only = f_pl[:, :NT]
    extras = np.concatenate([t_pl, f_pl[:, NT:]], axis=1)

    variants = [
        ("A tattica", t_teams, t_pl, t_ball, t_bs),
        ("B fusa completa", f_teams, f_pl, f_ball, f_bs),
        ("C giocatori fusi + palla tattica", f_teams, f_pl, t_ball, t_bs),
        ("D giocatori tattici + palla fusa", t_teams, t_pl, f_ball, f_bs),
        ("E solo media 75/25 giocatori", t_teams, blend_only, t_ball, t_bs),
        ("F solo giocatori extra", f_teams, extras, t_ball, t_bs),
        ("G palla: broadcast solo nei buchi", t_teams, t_pl, fill_ball, fill_bs),
        ("H G + giocatori fusi", f_teams, f_pl, fill_ball, fill_bs),
    ]
    ref = None
    print("{:36s} {:>7s} {:>7s} {:>5s} {:>6s} {:>6s} {:>7s} {:>7s} {:>7s}".format(
        "variante", "spellF1", "macroF1", "poss5", "possT", "D%", "tocchi", "salti", "=A %"))
    for name, teams, pl, ball, bs in variants:
        labels, f, r = v2.predict(teams, pl, ball, bs, blocks, pitch, v2.PARAMS, events, level)
        codes = pm.label_codes(labels)
        res = pm.evaluate(gt, codes)
        g = graded
        seg = labels[g]
        cut = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        starts = np.concatenate(([0], cut))
        poss_total = int(sum(seg[s] in "HA" for s in starts))
        vis = ~np.isnan(ball[:, 0])
        step = np.linalg.norm(np.diff(ball, axis=0), axis=1)
        jumps = int(np.sum((step > 1.5) & g[1:]))  # > 37 m/s between two frames: impossible
        touches = int(np.sum((f["touch_team"] >= 0) & g))
        if ref is None:
            ref = labels
        same = 100 * np.mean(labels[g] == ref[g])
        print("{:36s} {:7.4f} {:7.4f} {:5d} {:6d} {:6.1f} {:7d} {:7d} {:7.1f}".format(
            name, res["spell_f1_mean"], res["macro_f1"], int(res["poss"]) if "poss" in res else -1,
            poss_total, 100 * np.mean(seg == "D"), touches, jumps, same))

    # ball provenance switches: how big is the jump when the source changes
    g = graded
    sw = (src[1:] != src[:-1]) & (src[1:] > 0) & (src[:-1] > 0) & g[1:]
    step = np.linalg.norm(np.diff(f_ball, axis=0), axis=1)
    st = np.linalg.norm(np.diff(t_ball, axis=0), axis=1)
    print()
    print("ball, graded frames: switches of source (both seen) {}, jump at a switch median {:.2f} m, "
          "p90 {:.2f} m, > 1.5 m {}".format(int(sw.sum()), np.nanmedian(step[sw]),
                                            np.nanpercentile(step[sw], 90), int(np.sum(step[sw] > 1.5))))
    print("frame-to-frame step, tactical median {:.3f} m p99 {:.2f}; fused median {:.3f} m p99 {:.2f}".format(
        np.nanmedian(st[g[1:]]), np.nanpercentile(st[g[1:]], 99),
        np.nanmedian(step[g[1:]]), np.nanpercentile(step[g[1:]], 99)))
    gap = fz["ball_gap"]
    both = g & np.isfinite(gap)
    print("both see the ball on {:.1f}% of graded frames; apart > 2 m {:.1f}%, > 5 m {:.1f}% of those".format(
        100 * both.mean(), 100 * np.mean(gap[both] > 2), 100 * np.mean(gap[both] > 5)))
    # players: slot jumps (pairing flips)
    def slot_jumps(p):
        d = np.linalg.norm(np.diff(p, axis=0), axis=2)
        return int(np.nansum((d > 0.8) & g[1:, None]))  # > 20 m/s between two frames
    print("player slot jumps > 0.8 m/frame: tactical {}, fused tactical slots {}, extras {}".format(
        slot_jumps(t_pl), slot_jumps(f_pl[:, :NT]), slot_jumps(f_pl[:, NT:])))


if __name__ == "__main__":
    main("--no-whistle" not in sys.argv)
