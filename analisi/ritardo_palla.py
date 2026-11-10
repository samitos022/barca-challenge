"""Is the ATD ball late with respect to the players? Unlabelled test.

At every velocity change of the ball (a candidate touch: dv local maximum over DV_MIN), the
distance from the ball at t + tau to the closest player at t, for tau in -MAXLAG..+MAXLAG frames.
If the ball track is delayed by L frames, the ball reaches the touching player's position L frames
after the player is there: the median distance is smallest at tau = +L. Reference: the clean IDSSE
ball (TRACAB, no lag expected) and the same ball through a causal filter (known lag).

    .venv/bin/python analisi/ritardo_palla.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "solution"), os.path.join(ROOT, "solution", "external"),
          os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "starter"), HERE):
    sys.path.insert(0, p)
import calibrate  # noqa: E402
import idsse  # noqa: E402
import possession_v1 as v1  # noqa: E402
import possession_v2 as v2  # noqa: E402
import tracking  # noqa: E402
from fit_liscio_palla import ema, runs_apply  # noqa: E402
from games import GAME, half_video_blocks  # noqa: E402

MAXLAG = 25
DV_MIN = 2.0


def profile(players, ball, bs, graded):
    _, _, dv, _ = v1.ball_kinematics(ball, bs, v2.PARAMS)
    x = np.nan_to_num(dv, nan=0.0)
    w = 3
    lm = np.lib.stride_tricks.sliding_window_view(np.pad(x, w), 2 * w + 1).max(axis=1)
    cand = np.flatnonzero((x >= DV_MIN) & (x == lm) & graded)
    cand = cand[(cand > MAXLAG) & (cand < len(ball) - MAXLAG)]
    taus = np.arange(-MAXLAG, MAXLAG + 1)
    med = []
    for tau in taus:
        b = ball[cand + tau]
        d = np.linalg.norm(players[cand] - b[:, None], axis=2)
        d = np.where(np.isnan(d), np.inf, d).min(axis=1)
        d = d[np.isfinite(d)]
        med.append(np.median(d))
    med = np.array(med)
    k = int(np.argmin(med))
    return taus[k], med[k], med[MAXLAG], len(cand), med


def show(name, r):
    tau, best, at0, n, med = r
    print("{:34s} touches {:6d}: closest at tau {:+3d} frames ({:+.2f} s), median distance {:.2f} m "
          "(at tau 0: {:.2f} m)".format(name, n, tau, tau / 25, best, at0))
    print("{:34s} median distance by tau -20..+20 step 5: {}".format(
        "", " ".join("{:.2f}".format(med[MAXLAG + t]) for t in range(-20, 21, 5))))


def main():
    teams, pl, ball, bs, _ = tracking.load(GAME)
    g = np.zeros(len(ball), bool)
    for _, a, z in half_video_blocks(GAME):
        g[a:z + 1] = True
    show("BAR-ATH tactical ATD", profile(pl, ball, bs, g))
    _, bpl, bball, bbs, _ = tracking.load(GAME, "broadcast")
    show("BAR-ATH broadcast ATD", profile(bpl, bball, bbs, np.ones(len(bball), bool)))
    for game in idsse.GAMES[:2]:
        t, P_, b, s_, blocks, gt = calibrate.case((game, 1), "clean", "tracab")
        show("IDSSE {} clean".format(game), profile(P_, b, s_, gt != 3))
        lagged = runs_apply(b, ema(0.08))
        show("IDSSE {} causal filter 0.08".format(game), profile(P_, lagged, np.full(len(b), np.nan), gt != 3))
    t, P_, b, s_, blocks, gt = calibrate.case((1, 1), "clean")
    show("Metrica game 1", profile(P_, b, s_, gt != 3))


if __name__ == "__main__":
    main()
