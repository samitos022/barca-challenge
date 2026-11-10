"""Which smoothing makes an external ball look like BAR-ATH's ATD ball (unlabelled statistics only).

BAR-ATH's ball is much smoother than TRACAB's or Metrica's: v1 sees 16.7 touches per visible
minute against 30-43, the velocity change p99 is 8.7 m/s against 20-22, and only 56% of its
touches have a player within R_TOUCH against 88-93%. Two filter families are fitted on IDSSE
clean balls: centred Gaussian (blur only) and causal exponential (blur + lag). The target is
BAR-ATH's aggregate numbers, printed first.

    .venv/bin/python analisi/fit_liscio_palla.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "solution"), os.path.join(ROOT, "solution", "external"),
          os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "starter")):
    sys.path.insert(0, p)
import calibrate  # noqa: E402
import idsse  # noqa: E402
import possession_v1 as v1  # noqa: E402
import possession_v2 as v2  # noqa: E402
import tracking  # noqa: E402
from games import GAME, half_video_blocks  # noqa: E402

P = v2.PARAMS


def stats(teams, players, ball, bs, graded):
    _, _, dv, _ = v1.ball_kinematics(ball, bs, P)
    dist, _ = v1.team_distances(players, ball, teams)
    tt, ist = v1.touches(dv, dist, P)
    vis = ~np.isnan(ball[:, 0]) & graded
    vm = vis.sum() / 25 / 60
    d = dv[vis & np.isfinite(dv)]
    return np.array([ist[vis].sum() / vm, (tt[vis] >= 0).sum() / max(ist[vis].sum(), 1),
                     *np.percentile(d, [50, 90, 99])])


def runs_apply(ball, fn):
    """Apply fn to every run of visible frames separately (gaps stay gaps)."""
    out = ball.copy()
    vis = ~np.isnan(ball[:, 0])
    e = np.flatnonzero(np.diff(np.concatenate(([0], vis.astype(np.int8), [0]))))
    for a, b in zip(e[::2], e[1::2]):
        out[a:b] = fn(ball[a:b])
    return out


def gauss(sigma):
    r = int(3 * sigma) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()

    def fn(x):
        if len(x) < 2:
            return x
        pad = np.pad(x, ((r, r), (0, 0)), mode="edge")
        return np.stack([np.convolve(pad[:, j], k, mode="valid") for j in range(2)], 1)
    return fn


def ema(alpha):
    def fn(x):
        y = np.empty_like(x)
        y[0] = x[0]
        for i in range(1, len(x)):
            y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
        return y
    return fn


def main():
    names = ("touches/min", "attributed", "dv p50", "dv p90", "dv p99")
    teams, pl, ball, bs, _ = tracking.load(GAME)
    g = np.zeros(len(ball), bool)
    for _, a, z in half_video_blocks(GAME):
        g[a:z + 1] = True
    target = stats(teams, pl, ball, bs, g)
    print("{:24s} ".format("BAR-ATH (target)") + "  ".join("{} {:5.2f}".format(n, v) for n, v in zip(names, target)))
    nanbs = None
    cases = []
    for game in idsse.GAMES[:3]:
        t, P_, b, s_, blocks, gt = calibrate.case((game, 1), "clean", "tracab")
        cases.append((t, P_, b, gt != 3))
    scale = np.array([target[0], 0.1, target[2], target[3], target[4]])
    rng = np.random.default_rng(0)
    grid = [(fam, v, nz) for fam, values in (("gauss sigma", (4, 6, 8)), ("ema alpha", (0.12, 0.08, 0.05)))
            for v in values for nz in (0.0, 0.03, 0.06, 0.1)]
    best = None
    for fam, v, nz in grid:
        mk = gauss if fam.startswith("gauss") else ema
        if True:
            S = []
            for t, P_, b, graded in cases:
                sb = runs_apply(b, mk(v))
                if nz:
                    # position noise, itself smoothed over ~0.1 s (the ATD wobble, not white noise)
                    noise = rng.normal(0, nz, size=sb.shape)
                    noise = runs_apply(np.where(np.isnan(sb), np.nan, noise), gauss(2.0)) * np.sqrt(2 * np.sqrt(np.pi) * 2.0)
                    sb = sb + noise
                nan_speed = np.full(len(sb), np.nan, np.float32)  # speed from positions, as for a smoothed ATD
                S.append(stats(t, P_, sb, nan_speed, graded))
            S = np.mean(S, 0)
            err = np.sqrt(np.mean(((S - target) / scale) ** 2))
            if best is None or err < best[0]:
                best = (err, fam, v, nz)
            print("{:24s} ".format("{} {} noise {}".format(fam, v, nz)) + "  ".join(
                "{} {:5.2f}".format(n, x) for n, x in zip(names, S)) + "   distance {:.2f}".format(err), flush=True)
    print("best: {} {} noise {} m (distance {:.2f})".format(best[1], best[2], best[3], best[0]))


if __name__ == "__main__":
    main()
