"""Disagreement runs (cameras > DIS m apart): which camera jumps where the run starts / ends, and
which one stays near the players. Aggregate numbers only."""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "starter", "solution"):
    sys.path.insert(0, os.path.join(ROOT, d))
from games import GAME, half_video_blocks
import tracking, sync_broadcast

DIS = 2.0
_, t_pl, t_ball, _, _ = tracking.load(GAME)
_, _, b_ball, _, _ = tracking.load(GAME, "broadcast")
b_live = np.load(sync_broadcast.OUT)["b_live"]
n = len(t_ball)
g = np.zeros(n, bool)
for _, a, z in half_video_blocks(GAME):
    g[a:z + 1] = True
bb = np.full((n, 2), np.nan, np.float32)
ok = b_live >= 0
bb[ok] = b_ball[b_live[ok]]
tv, bv = ~np.isnan(t_ball[:, 0]), ~np.isnan(bb[:, 0])
both = g & tv & bv
d = np.linalg.norm(bb - t_ball, axis=1)
dis = both & (d > DIS)
e = np.flatnonzero(np.diff(np.concatenate(([0], dis.astype(np.int8), [0]))))
st, en = e[::2], e[1::2]

def step(b, t):
    return np.linalg.norm(b[t] - b[t - 1]) if t > 0 else np.nan

jt, jb, kinds = [], [], {"t jumps": 0, "b jumps": 0, "both jump": 0, "neither": 0, "edge unseen": 0}
J = 1.0  # m in one frame = 25 m/s on top of the motion: a switch of object
for s0, e0 in zip(st, en):
    for t in (s0, e0 if e0 < n else None):
        if t is None or t == 0:
            continue
        if not (tv[t] and tv[t - 1] and bv[t] and bv[t - 1]):
            kinds["edge unseen"] += 1
            continue
        a, b = step(t_ball, t), step(bb, t)
        jt.append(a); jb.append(b)
        kinds["both jump" if a > J and b > J else "t jumps" if a > J else "b jumps" if b > J else "neither"] += 1
print("disagreement runs > {} m: {}".format(DIS, len(st)))
print("run edges:", kinds)
# who is closer to the nearest player (the true ball is usually at someone's feet)
pd_t = np.nanmin(np.linalg.norm(t_pl - t_ball[:, None], axis=2), axis=1)
pd_b = np.nanmin(np.linalg.norm(t_pl - bb[:, None], axis=2), axis=1)
m = dis & np.isfinite(pd_t) & np.isfinite(pd_b)
print("in disagreement: nearest player median tactical ball {:.2f} m, broadcast ball {:.2f} m; tactical closer {:.0f}%".format(
    np.median(pd_t[m]), np.median(pd_b[m]), 100 * np.mean(pd_t[m] < pd_b[m])))
m2 = both & (d < 1.0)
print("in agreement: nearest player median {:.2f} m".format(np.median(pd_t[m2])))
# out of the pitch?
L, W = 105.3, 68.0
def outside(b):
    return (b[:, 0] < -1) | (b[:, 0] > L + 1) | (b[:, 1] < -1) | (b[:, 1] > W + 1)
print("in disagreement: tactical ball off the pitch {:.1f}%, broadcast ball off the pitch {:.1f}%".format(
    100 * outside(t_ball)[dis].mean(), 100 * outside(bb)[dis].mean()))
# per run: does it start from agreement (a switch) or from a gap?
pre = np.array([s0 > 0 and both[s0 - 1] and not dis[s0 - 1] for s0 in st])
print("runs starting right after agreement {:.0f}%".format(100 * pre.mean()))
v = np.linalg.norm(np.gradient(t_ball, axis=0), axis=1) * 25
print("ball speed median: agreement {:.1f} m/s, disagreement {:.1f} m/s; share over 12 m/s: {:.0f}% vs {:.0f}%".format(
    np.nanmedian(v[m2]), np.nanmedian(v[dis]), 100 * np.nanmean(v[m2] > 12), 100 * np.nanmean(v[dis] > 12)))
# distance profile inside a run: grows then shrinks (a flight) ?
prof = []
for s0, e0 in zip(st, en):
    if e0 - s0 >= 20:
        x = d[s0:e0]
        prof.append(np.interp(np.linspace(0, len(x) - 1, 5), np.arange(len(x)), x))
print("distance through a run (start..end, median of runs >= 20 frames):", np.round(np.median(prof, axis=0), 1))
