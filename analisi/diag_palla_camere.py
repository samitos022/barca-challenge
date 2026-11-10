"""How the two cameras' balls disagree, aggregate numbers only (no labels used).

- offset b - t where both see the ball: is it a slowly varying bias (correctable) or noise?
- where they are > DIS m apart: which one is consistent with its own trajectory
  (distance from the straight line through its neighbours at -k and +k frames)?

    .venv/bin/python analisi/diag_palla_camere.py
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "starter", "solution"):
    sys.path.insert(0, os.path.join(ROOT, d))
from games import GAME, half_video_blocks  # noqa: E402
import tracking  # noqa: E402
import sync_broadcast  # noqa: E402

K = 3


def residual(ball, k=K):
    """Distance of each point from the midpoint of its neighbours at -k and +k (NaN at edges)."""
    n = len(ball)
    r = np.full(n, np.nan)
    r[k:n - k] = np.linalg.norm(ball[k:n - k] - (ball[:n - 2 * k] + ball[2 * k:]) / 2, axis=1)
    return r


def main():
    _, _, t_ball, _, _ = tracking.load(GAME)
    _, _, b_ball, _, _ = tracking.load(GAME, "broadcast")
    s = np.load(sync_broadcast.OUT)
    b_live = s["b_live"]
    n = len(t_ball)
    g = np.zeros(n, bool)
    for _, a, z in half_video_blocks(GAME):
        g[a:z + 1] = True
    bb = np.full((n, 2), np.nan, np.float32)
    ok = b_live >= 0
    bb[ok] = b_ball[b_live[ok]]
    tv, bv = ~np.isnan(t_ball[:, 0]), ~np.isnan(bb[:, 0])
    both = g & tv & bv
    off = bb - t_ball
    d = np.linalg.norm(off, axis=1)
    print("graded: tactical sees {:.1f}%, broadcast live sees {:.1f}%, both {:.1f}%, broadcast only {:.1f}%".format(
        100 * tv[g].mean(), 100 * bv[g].mean(), 100 * both[g].mean(), 100 * (bv & ~tv)[g].mean()))
    print("distance when both: median {:.2f} m, p75 {:.2f}, p90 {:.2f}, p95 {:.2f}".format(
        *np.percentile(d[both], [50, 75, 90, 95])))

    # bias: how much does the offset change over 1 s, where it is small (same object)?
    same = both & (d < 2.0)
    for lag in (1, 5, 25, 50):
        a = same[:-lag] & same[lag:]
        ch = np.linalg.norm(off[lag:] - off[:-lag], axis=1)[a]
        print("offset change over {:2d} frames (same object): median {:.2f} m p90 {:.2f}".format(
            lag, np.median(ch), np.percentile(ch, 90)))
    # does the offset grow with ball speed (sync lag) ?
    v = np.linalg.norm(np.gradient(t_ball, axis=0), axis=1) * 25
    for lo, hi in ((0, 3), (3, 8), (8, 15), (15, 40)):
        m = same & (v >= lo) & (v < hi)
        if m.sum():
            # component of the offset along the ball's motion
            vel = np.gradient(t_ball, axis=0)
            u = vel[m] / np.maximum(np.linalg.norm(vel[m], axis=1, keepdims=True), 1e-9)
            along = np.sum(off[m] * u, axis=1)
            print("ball {:2d}-{:2d} m/s: n {:6d}, distance median {:.2f} m, offset along the motion median {:+.2f} m".format(
                lo, hi, int(m.sum()), np.median(d[m]), np.median(along)))

    # disagreements: who is self-consistent?
    rt, rb = residual(t_ball), residual(bb)
    for dis in (2.0, 5.0):
        m = both & (d > dis) & np.isfinite(rt) & np.isfinite(rb)
        print("apart > {:.0f} m: {} frames ({:.1f}% of both); trajectory residual median tactical {:.2f} m, "
              "broadcast {:.2f} m; tactical smoother on {:.0f}%".format(
                  dis, int(m.sum()), 100 * m.sum() / both.sum(), np.median(rt[m]), np.median(rb[m]),
                  100 * np.mean(rt[m] < rb[m])))
        # how long are the disagreement runs
        e = np.flatnonzero(np.diff(np.concatenate(([0], (both & (d > dis)).astype(np.int8), [0]))))
        L = e[1::2] - e[::2]
        print("   runs: {}, length median {:.0f} frames, p90 {:.0f}, frames in runs >= 25: {:.0f}%".format(
            len(L), np.median(L), np.percentile(L, 90), 100 * L[L >= 25].sum() / max(L.sum(), 1)))
    m = both & (d < 2.0) & np.isfinite(rt) & np.isfinite(rb)
    print("apart < 2 m: residual median tactical {:.2f} m, broadcast {:.2f} m".format(
        np.median(rt[m]), np.median(rb[m])))

    # broadcast-only runs: lengths, and whether a tactical sighting is adjacent (offset estimable)
    bo = g & bv & ~tv
    e = np.flatnonzero(np.diff(np.concatenate(([0], bo.astype(np.int8), [0]))))
    st, en = e[::2], e[1::2]
    L = en - st
    adj = np.array([(s0 > 0 and both[s0 - 1]) or (e0 < n and both[e0]) for s0, e0 in zip(st, en)])
    near = np.array([both[max(0, s0 - 25):s0].any() or both[e0:e0 + 25].any() for s0, e0 in zip(st, en)])
    print("broadcast-only runs: {}, length median {:.0f} frames, p90 {:.0f}; touching a both-seen frame {:.0f}%, "
          "one within 1 s {:.0f}%".format(len(L), np.median(L), np.percentile(L, 90), 100 * adj.mean(),
                                         100 * near.mean()))
    # tactical gaps (>= 5 s): how many get broadcast frames inside
    miss = g & ~tv
    e = np.flatnonzero(np.diff(np.concatenate(([0], miss.astype(np.int8), [0]))))
    st, en = e[::2], e[1::2]
    long = (en - st) >= 125
    filled = np.array([bv[a:b].any() for a, b in zip(st, en)])
    print("tactical gaps >= 5 s: {}; with some broadcast ball inside {:.0f}%, broadcast fills {:.1f}% of their frames".format(
        int(long.sum()), 100 * filled[long].mean(),
        100 * np.mean(np.concatenate([bv[a:b] for a, b in zip(st[long], en[long])]))))


if __name__ == "__main__":
    main()
