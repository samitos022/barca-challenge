"""Fit a ball degradation (blur + slow position error + jitter) to BAR-ATH's unlabelled ball statistics.

Targets, all measured on BAR-ATH without labels: v1 touches per visible minute, share of them with
a player within R_TOUCH, velocity-change percentiles, and the median ball-to-nearest-player
distance at velocity changes (1.46 m on BAR-ATH, 0.57 m on clean TRACAB: a slowly varying position
error of the ATD ball, not a lag; see analisi/ritardo_palla.py).

    .venv/bin/python analisi/fit_palla_realistica.py
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
from fit_liscio_palla import gauss, runs_apply  # noqa: E402
from games import GAME, half_video_blocks  # noqa: E402

P = v2.PARAMS
NAMES = ("touches/min", "attributed", "dv p50", "dv p90", "dv p99", "touch dist")


def stats(teams, players, ball, bs, graded):
    _, _, dv, _ = v1.ball_kinematics(ball, bs, P)
    dist, _ = v1.team_distances(players, ball, teams)
    tt, ist = v1.touches(dv, dist, P)
    vis = ~np.isnan(ball[:, 0]) & graded
    vm = vis.sum() / 25 / 60
    d = dv[vis & np.isfinite(dv)]
    x = np.nan_to_num(dv)
    lm = np.lib.stride_tricks.sliding_window_view(np.pad(x, 3), 7).max(axis=1)
    c = np.flatnonzero((x >= 2) & (x == lm) & graded)
    dd = np.linalg.norm(players[c] - ball[c][:, None], axis=2)
    dd = np.where(np.isnan(dd), np.inf, dd).min(axis=1)
    dd = dd[dd < 4]
    return np.array([ist[vis].sum() / vm, (tt[vis] >= 0).sum() / max(ist[vis].sum(), 1),
                     *np.percentile(d, [50, 90, 99]), np.median(dd)])


def slow_noise(shape, amp, tau, rng, like):
    """Gaussian noise of standard deviation amp per axis, correlated over ~tau frames."""
    z = rng.normal(0, 1, size=shape)
    z = runs_apply(np.where(np.isnan(like), np.nan, z), gauss(tau))
    z /= np.nanstd(z)
    return amp * z


def degrade_ball_shape(ball, blur, amp, tau, jitter, rng):
    out = runs_apply(ball, gauss(blur)) if blur else ball.copy()
    if amp:
        out = out + slow_noise(out.shape, amp, tau, rng, out)
    if jitter:
        out = out + slow_noise(out.shape, jitter, 2.0, rng, out)
    return out


def main():
    teams, pl, ball, bs, _ = tracking.load(GAME)
    g = np.zeros(len(ball), bool)
    for _, a, z in half_video_blocks(GAME):
        g[a:z + 1] = True
    target = stats(teams, pl, ball, bs, g)
    print("{:30s} ".format("BAR-ATH (target)") + "  ".join("{} {:5.2f}".format(n, v) for n, v in zip(NAMES, target)))
    cases = [calibrate.case((game, 1), "clean", "tracab") for game in idsse.GAMES[:3]]
    scale = np.array([target[0], 0.1, target[2], target[3], target[4], target[5]])
    rng = np.random.default_rng(0)
    best = None
    for blur in (4, 6):
        for amp in (0.4, 0.7, 1.0):
            for tau in (6, 12, 25):
                for jitter in (0.04, 0.08):
                    S = []
                    for t, P_, b, s_, blocks, gt in cases:
                        sb = degrade_ball_shape(b, blur, amp, tau, jitter, rng)
                        S.append(stats(t, P_, sb, np.full(len(b), np.nan, np.float32), gt != 3))
                    S = np.mean(S, 0)
                    err = np.sqrt(np.mean(((S - target) / scale) ** 2))
                    if best is None or err < best[0]:
                        best = (err, blur, amp, tau, jitter, S)
                    print("{:30s} ".format("blur {} amp {} tau {} jit {}".format(blur, amp, tau, jitter)) + "  ".join(
                        "{} {:5.2f}".format(n, x) for n, x in zip(NAMES, S)) + "   dist {:.2f}".format(err), flush=True)
    print("best: blur {} amp {} tau {} jitter {} (distance {:.2f})".format(*best[1:5], best[0]))


if __name__ == "__main__":
    main()
