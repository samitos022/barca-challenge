"""What the tactical camera can see: missing players / keepers / ball versus the field of view.

The homography maps pitch (0-1) to pixels; a pitch point is in view when it projects inside the
1920x1080 frame (with the point in front of the camera). Aggregate numbers only.

    .venv/bin/python analisi/audit_inquadratura.py
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "starter", "solution"):
    sys.path.insert(0, os.path.join(ROOT, d))
from games import GAME, GAMES, asset_paths, half_video_blocks  # noqa: E402
import tracking  # noqa: E402

FPS = 25
L, W = GAMES[GAME]["pitch"]
IMG_W, IMG_H, MARGIN = 1920, 1080, 10


def in_view(H, x, y):
    """H (n, 3, 3), x, y in metres (broadcastable to (n, k)) -> bool (n, k)."""
    u, v = x / L, y / W
    X = H[:, 0, 0, None] * u + H[:, 0, 1, None] * v + H[:, 0, 2, None]
    Y = H[:, 1, 0, None] * u + H[:, 1, 1, None] * v + H[:, 1, 2, None]
    Z = H[:, 2, 0, None] * u + H[:, 2, 1, None] * v + H[:, 2, 2, None]
    px, py = X / Z, Y / Z
    return (Z > 0) & (px >= MARGIN) & (px < IMG_W - MARGIN) & (py >= MARGIN) & (py < IMG_H - MARGIN)


def main():
    teams, pl, ball, _, names = tracking.load(GAME)
    n = len(pl)
    H = np.loadtxt(asset_paths(GAME)["homography"], delimiter=",", skiprows=1, usecols=range(1, 10)).reshape(-1, 3, 3)
    g = np.zeros(n, bool)
    for _, a, z in half_video_blocks(GAME):
        g[a:z + 1] = True
    idx = np.flatnonzero(g)

    # pitch coverage
    gx, gy = np.meshgrid(np.linspace(0, L, 43), np.linspace(0, W, 28))
    vis = in_view(H[idx], gx.ravel()[None], gy.ravel()[None])
    cover = vis.mean(axis=1)
    print("share of the pitch in view: median {:.0f}%, p10 {:.0f}%, whole pitch on {:.0f}% of frames".format(
        100 * np.median(cover), 100 * np.percentile(cover, 10), 100 * np.mean(cover > 0.99)))
    goals = in_view(H[idx], np.array([[0.0, L]]), np.array([[W / 2, W / 2]]))
    print("goal 0 (x=0) in view {:.0f}%, goal 1 (x=L) in view {:.0f}%, both {:.0f}%".format(
        100 * goals[:, 0].mean(), 100 * goals[:, 1].mean(), 100 * goals.all(axis=1).mean()))
    # are tracked players / ball always inside the view? (sanity: the tracker sees only the image)
    seen = ~np.isnan(pl[idx, :, 0])
    pv = in_view(H[idx], np.nan_to_num(pl[idx, :, 0]), np.nan_to_num(pl[idx, :, 1]))
    print("tracked players inside the view {:.1f}% (outside = kept/extrapolated by the tracker)".format(
        100 * pv[seen].mean()))
    bs = ~np.isnan(ball[idx, 0])
    bvw = in_view(H[idx], np.nan_to_num(ball[idx, 0])[:, None], np.nan_to_num(ball[idx, 1])[:, None])[:, 0]
    print("tracked ball inside the view {:.1f}%".format(100 * bvw[bs].mean()))

    # missing players vs coverage
    cnt = seen.sum(axis=1)
    for lo, hi in ((0, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.01)):
        m = (cover >= lo) & (cover < hi)
        if m.any():
            print("   pitch in view {:3.0f}-{:3.0f}%: {:5.1f}% of frames, tracked objects mean {:.1f}, ball seen {:.0f}%".format(
                100 * lo, min(100, 100 * hi), 100 * m.mean(), cnt[m].mean(), 100 * bs[m].mean()))

    # keepers: when no own player within 20 m of the own goal (ball far), is that goal in view?
    for half, a, z in half_video_blocks(GAME):
        sel = (idx >= a) & (idx <= z)
        ii = idx[sel]
        for k in (0, 1):
            mx = np.nanmean(pl[a:z + 1][:, teams == k, 0])
            own = 0 if mx < L / 2 else 1
            own_x = 0.0 if own == 0 else L
            dgl = np.abs(pl[ii][:, teams == k, 0] - own_x)
            deepest = np.nanmin(np.where(np.isnan(dgl), np.inf, dgl), axis=1)
            far = np.nan_to_num(np.abs(ball[ii, 0] - own_x), nan=0) > 40
            miss = far & (deepest > 20)
            gv = goals[sel, own]
            # area within 20 m of own goal in view?
            ax = np.full((1, 1), 10.0 if own == 0 else L - 10.0)
            av = in_view(H[ii], ax, np.full((1, 1), W / 2))[:, 0]
            print("half {} {:12s}: keeper area missing on {:4.1f}% of far-ball frames; of those, own goal in view "
                  "{:3.0f}% (area 10 m out in view {:3.0f}%); goal in view when the keeper is there {:3.0f}%".format(
                      half, names[k], 100 * miss[far].mean(), 100 * gv[miss].mean(), 100 * av[miss].mean(),
                      100 * gv[far & ~miss].mean()))

    # ball gaps: where was the ball relative to the view?
    st = np.flatnonzero(np.diff(np.concatenate(([0], (~np.isnan(ball[:, 0]) | ~g).astype(np.int8) ^ 1, [0]))))
    s0, e0 = st[::2], st[1::2]
    long = (e0 - s0) >= 5 * FPS
    cov_full = np.full(n, np.nan)
    cov_full[idx] = cover
    gc = np.array([np.nanmean(cov_full[a:b]) for a, b in zip(s0[long], e0[long])])
    print("inside long ball gaps: pitch in view median {:.0f}% (all graded frames {:.0f}%)".format(
        100 * np.median(gc), 100 * np.median(cover)))
    # ball visible fraction vs ball being in view? last sighting in view
    last = s0[long] - 1
    last = last[last >= 0]
    lv = in_view(H[last], ball[last, 0][:, None], ball[last, 1][:, None])[:, 0]
    dist_edge = []
    for t in last:
        Hi = H[t]
        q = Hi @ np.array([ball[t, 0] / L, ball[t, 1] / W, 1.0])
        px, py = q[0] / q[2], q[1] / q[2]
        dist_edge.append(min(px, IMG_W - px, py, IMG_H - py))
    dist_edge = np.array(dist_edge)
    print("last sighting before a long gap: in view {:.0f}%, within 100 px of the image edge {:.0f}%".format(
        100 * lv.mean(), 100 * np.mean(dist_edge < 100)))


if __name__ == "__main__":
    main()
