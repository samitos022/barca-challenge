"""The same sanity aggregates for every external calibration set, so they can be compared.

    .venv/bin/python solution/external/sanity.py                    # every set that is on disk
    .venv/bin/python solution/external/sanity.py idsse skillcorner  # only these

Per match: graded minutes, share of D, possessions per minute and their mean length, stoppages per
minute and their mean length, share of graded frames with the ball tracked, tracked players per
frame, and the two numbers that tell how the ball behaves in a stoppage:
  dead_ball_pct         share of D frames with the ball tracked;
  dead_ball_moving_pct  of the D frames where a speed can be measured, share faster than MOVING
                        (2 m/s). Speed from the positions: central difference over 5 frames
                        (|p[i+2] - p[i-2]| / 0.16 s), so the 2 frames at each end of a tracked
                        stretch have none.
"""

import sys

import numpy as np

FPS = 25
MOVING = 2.0     # m/s
HALF = 2         # central difference over 2 * HALF + 1 frames


def speed(ball):
    """(N,) m/s from the positions, NaN where p[i - HALF] or p[i + HALF] is missing."""
    v = np.full(len(ball), np.nan, dtype=np.float32)
    d = ball[2 * HALF:] - ball[:-2 * HALF]
    v[HALF:-HALF] = np.hypot(d[:, 0], d[:, 1]) * FPS / (2 * HALF)
    return v


def aggregates(lab, players, ball):
    """Dict of the aggregates over the graded (non-X) frames. lab: 'H' / 'A' / 'D' / 'X' per frame."""
    g = lab != "X"
    seg = lab[g]
    cut = np.flatnonzero(seg[1:] != seg[:-1]) + 1
    starts = np.concatenate(([0], cut))
    lens = np.diff(np.concatenate((starts, [len(seg)])))
    kinds = seg[starts]
    minutes = len(seg) / FPS / 60
    dead = g & (lab == "D")
    alive = g & ~dead
    seen = ~np.isnan(ball[:, 0])
    v = speed(ball)
    with np.errstate(invalid="ignore"):
        fast = v > MOVING
    has_v = ~np.isnan(v)
    pct = lambda m: 100 * float(np.mean(m)) if len(m) else float("nan")
    return dict(
        frames=int(len(seg)), minutes=minutes, d_pct=pct(seg == "D"),
        poss_per_min=float(np.sum(kinds != "D") / minutes),
        poss_len_s=float(lens[kinds != "D"].mean() / FPS) if np.any(kinds != "D") else float("nan"),
        dead_per_min=float(np.sum(kinds == "D") / minutes),
        dead_len_s=float(lens[kinds == "D"].mean() / FPS) if np.any(kinds == "D") else float("nan"),
        ball_pct=pct(seen[g]), players=float(np.mean(np.sum(~np.isnan(players[g, :, 0]), axis=1))),
        dead_ball_pct=pct(seen[dead]), dead_ball_moving_pct=pct(fast[dead & has_v]),
        alive_ball_pct=pct(seen[alive]), alive_ball_moving_pct=pct(fast[alive & has_v]))


def mean(rows):
    return {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}


def line(name, s):
    return ("{:<14} {:6.1f} min  D {:4.1f}%  poss {:4.2f}/min x {:4.1f} s  stoppages {:4.2f}/min x {:4.1f} s  "
            "ball {:5.1f}%  players {:4.1f}  | dead: ball {:5.1f}%, moving {:4.1f}%  | alive: ball {:5.1f}%, "
            "moving {:4.1f}%").format(
                str(name), s["minutes"], s["d_pct"], s["poss_per_min"], s["poss_len_s"], s["dead_per_min"],
                s["dead_len_s"], s["ball_pct"], s["players"], s["dead_ball_pct"], s["dead_ball_moving_pct"],
                s["alive_ball_pct"], s["alive_ball_moving_pct"])


def _cases(name):
    """[(game, labels, players, ball)] of one set, loaded lazily."""
    if name == "idsse":
        import idsse
        for g in idsse.GAMES:
            _, players, ball, _, _ = idsse.load(g)
            yield g, idsse.labels(g), players, ball
    elif name == "metrica":
        import metrica
        for g in metrica.GAMES:
            _, players, ball, _, periods = metrica.load(g)
            yield g, metrica.labels(g, len(ball), periods), players, ball
    else:
        mod = __import__(name)
        for g in mod.GAMES:
            try:
                _, players, ball, _, _ = mod.load(g)
            except FileNotFoundError:
                continue
            yield g, mod.labels(g), players, ball


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    known = [f[:-3] for f in sorted(os.listdir(here))
             if f.endswith(".py") and f[:-3] not in ("sanity", "degrade")]
    for name in sys.argv[1:] or known:
        rows = []
        try:
            for g, lab, players, ball in _cases(name):
                rows.append(aggregates(lab, players, ball))
                print(line("{} {}".format(name[:6], g), rows[-1]), flush=True)
        except FileNotFoundError as e:
            print("{}: not on disk ({})".format(name, e))
            continue
        if rows:
            print(line("{} mean".format(name[:8]), mean(rows)), flush=True)
