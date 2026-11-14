"""Windows around the errors, for looking at them: on the external matches (out-of-fold
probabilities of spell_lab), and on BAR-ATH's public five minutes (only to look, never to choose).

Each picture covers +-SPAN seconds around an error: true state, predicted state, P(H) / P(A) / P(D),
and the collective signals that should tell a stoppage (players' pace and its ratio to the last 30 s,
share standing, pressure on the ball, players in their own half, ball seen and moving). Written to
data/derived/lab/finestre/<error type>/, with an index.csv (match, frame, seconds, type). For BAR-ATH
the index also gives the video time, to open the review video at that moment.

    .venv/bin/python analisi/finestre_errori.py [tag] [n per type and dataset]
"""

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "solution"), os.path.join(ROOT, "solution", "external"),
          os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "starter"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import possession_metrics as pm  # noqa: E402
import spell_lab as L  # noqa: E402
import viterbi  # noqa: E402

FPS = 25
SPAN = 15
H, A, D, X = 0, 1, 2, 3
COL = {H: "#2f6fd6", A: "#e0523a", D: "#9a9a9a", X: "#202020"}
SIGNALS = (("col_speed_med_home", "pace home m/s"), ("col_speed_med_away", "pace away m/s"),
           ("col_pace_vs_30s", "pace / last 30 s"), ("col_still_home", "still share home"),
           ("col_still_away", "still share away"), ("col_press_d_home", "pressure d home m"),
           ("col_press_d_away", "pressure d away m"), ("col_own_half_home", "own half home"),
           ("col_ball_move_past1s", "ball moved last 1 s m"))


def error_events(gt, pred):
    """[(frame, type)]: false_stop, missed_stop, late_start, early_start, late_end, early_end
    (more than 1 s off), split_possession."""
    runs_ = pm.valid_runs(gt)
    gs, ps = pm.segments(gt, runs_), pm.segments(pred, runs_)
    gd = [(a, b) for a, b, k in gs if k == D]
    pdd = [(a, b) for a, b, k in ps if k == D]
    out = []
    for a, b in pdd:
        if not any(min(b, d) - max(a, c) > 0 for c, d in gd):
            out.append((a, "false_stop"))
    for a, b in gd:
        hit = [(c, d) for c, d in pdd if min(b, d) - max(a, c) > 0]
        if not hit:
            out.append((a, "missed_stop"))
            continue
        ds, de = (hit[0][0] - a) / FPS, (hit[-1][1] - b) / FPS
        if ds > 1:
            out.append((a, "late_start"))
        elif ds < -1:
            out.append((hit[0][0], "early_start"))
        if de > 1:
            out.append((b, "late_end"))
        elif de < -1:
            out.append((hit[-1][1], "early_end"))
    gsp, psp = pm.possession_spells(gt, runs_), pm.possession_spells(pred, runs_)
    for a, b, k in gsp:
        pieces = [(c, e) for c, e, kk in psp if kk == k and min(b, e) - max(a, c) > 0.2 * (b - a)]
        if len(pieces) >= 2:
            out.append((pieces[0][1], "split_possession"))
    return out


def strip(ax, codes, t0, y, h, label):
    for i, c in enumerate(codes):
        ax.add_patch(plt.Rectangle((t0 + i / FPS, y), 1 / FPS, h, color=COL[int(c)], lw=0))
    ax.text(t0 - 0.2, y + h / 2, label, ha="right", va="center", fontsize=8)


def plot(path, title, gt, pred, p3, col, cnames, ball, f, extra_line=None):
    a, b = max(0, f - SPAN * FPS), min(len(gt), f + SPAN * FPS)
    t = (np.arange(a, b) - f) / FPS
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [1, 2, 3]})
    ax = axes[0]
    strip(ax, gt[a:b], t[0], 1.1, 0.8, "truth")
    strip(ax, pred[a:b], t[0], 0.1, 0.8, "predicted")
    ax.set_ylim(0, 2)
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    ax = axes[1]
    for j, nm in ((H, "P(home)"), (A, "P(away)"), (D, "P(dead)")):
        ax.plot(t, p3[a:b, j], color=COL[j], lw=1.2, label=nm)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", fontsize=7, ncol=3)
    ax.axvline(0, color="k", lw=0.6, ls=":")
    ax = axes[2]
    for nm, lab in SIGNALS:
        if nm in cnames:
            v = col[a:b, cnames.index(nm)]
            if np.isfinite(v).any():
                ax.plot(t, v, lw=0.9, label=lab)
    seen = ~np.isnan(ball[a:b, 0])
    ax.fill_between(t, 0, 0.3, where=~seen, color="#ffcc66", alpha=0.6, step="mid", label="ball not tracked")
    ax.axvline(0, color="k", lw=0.6, ls=":")
    ax.legend(loc="upper left", fontsize=7, ncol=3)
    ax.set_xlabel("seconds from the error" + ("   " + extra_line if extra_line else ""))
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main(tag="base", per=4, cond="realistic1"):
    import calibrate
    import possession_ml as ml
    o = L.load_oof(tag)
    outdir = os.path.join(L.LAB, "finestre")
    os.makedirs(outdir, exist_ok=True)
    rows = []
    rng = np.random.default_rng(0)
    for s in ("DFL", "Metrica", "SkillCorner", "PFF"):
        keys = [k for k in o if k.startswith(s + "_") and k.endswith("_" + cond)]
        events = []
        for k in keys:
            z = L.load(L._run_of_key(k))
            gt = np.asarray(z["gt"])
            pred = viterbi.decode(o[k], L.blocks_of(z), ml.COST_TEAM, ml.COST_DEAD)
            events += [(k, f, ty) for f, ty in error_events(gt, pred)]
        for ty in sorted({e[2] for e in events}):
            ev = [e for e in events if e[2] == ty]
            for i in rng.choice(len(ev), size=min(per, len(ev)), replace=False):
                k, f, _ = ev[i]
                r = L._run_of_key(k)
                z = L.load(r)
                gt = np.asarray(z["gt"])
                pred = viterbi.decode(o[k], L.blocks_of(z), ml.COST_TEAM, ml.COST_DEAD)
                cz = np.load(L.col_path(r)) if os.path.exists(L.col_path(r)) else None
                col = cz["X"] if cz is not None else np.full((len(gt), 1), np.nan)
                cn = [str(x) for x in cz["names"]] if cz is not None else []
                ball = calibrate.case((r[0], r[3]), r[2], r[1])[2]
                os.makedirs(os.path.join(outdir, ty), exist_ok=True)
                p = os.path.join(outdir, ty, "{}_{}.png".format(k, f))
                plot(p, "{} | {} | frame {}".format(ty, k, f), gt, pred, o[k], col, cn, ball, f)
                rows.append((s, k, f, round(f / FPS, 1), ty, os.path.relpath(p, ROOT)))
        print("{}: {} error events, pictures written".format(s, len(events)), flush=True)
    with open(os.path.join(outdir, "index.csv"), "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(("dataset", "run", "frame", "seconds", "type", "picture"))
        w.writerows(rows)
    print("{} pictures, index in {}".format(len(rows), os.path.relpath(os.path.join(outdir, "index.csv"), ROOT)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "base", int(sys.argv[2]) if len(sys.argv) > 2 else 4)
