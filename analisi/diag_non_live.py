"""Is 'broadcast not live' a dead-ball signal? Aggregate numbers only, no labels.

Compares the share of tactical frames whose broadcast frame is not live (cut to a close-up, a
replay, another angle) in situations known without labels:
  - surely alive: ball seen, fast (> 8 m/s) and players running (> 2.5 m/s mean);
  - surely dead: 3-10 s after a confirmed ball out (rule 1);
  - rule 8 gaps of the team's v2 (long gap, players walking, supported);
and against the state of the v2 machine (tactical). Also the replays from replay.py.

    .venv/bin/python analisi/diag_non_live.py
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "starter", "solution"):
    sys.path.insert(0, os.path.join(ROOT, d))
from games import GAME, GAMES, half_video_blocks  # noqa: E402
import possession_v2 as v2  # noqa: E402
import tracking  # noqa: E402
import sync_broadcast  # noqa: E402
import whistle  # noqa: E402

FPS = 25


def main():
    teams, pl, ball, bs, _ = tracking.load(GAME)
    blocks = half_video_blocks(GAME)
    pitch = GAMES[GAME]["pitch"]
    ev, lv = whistle.load(GAME)
    labels, f, r = v2.predict(teams, pl, ball, bs, blocks, pitch, v2.PARAMS, ev, lv)
    s = np.load(sync_broadcast.OUT)
    b_of_t, b_live = s["b_of_t"], s["b_live"]
    rep = np.load(os.path.join(ROOT, "data", "derived", "{}_replays.npz".format(GAME)))["replay_t"]
    n = len(labels)
    g = np.zeros(n, bool)
    for _, a, z in blocks:
        g[a:z + 1] = True
    covered = g & (b_of_t >= 0)
    nonlive = covered & (b_live < 0)
    print("graded frames with a synced broadcast frame {:.1f}%; of those not live {:.1f}%".format(
        100 * covered[g].mean(), 100 * nonlive[covered].mean()))

    vis = f["visible"]
    alive = covered & vis & (np.nan_to_num(f["speed"]) > 8) & (np.nan_to_num(f["pspeed"]) > 2.5)
    out_after = np.zeros(n, bool)
    for t in np.flatnonzero(f["out_start"] >= 0):
        out_after[t + 3 * FPS:t + 10 * FPS] = True
    out_after &= covered
    gap8 = np.zeros(n, bool)
    for a, b, _ in f["gap_why"]:
        gap8[a:b] = True
    gap8 &= covered
    for name, m in (("surely alive (ball and players fast)", alive),
                    ("3-10 s after a ball out", out_after),
                    ("inside a rule 8 gap", gap8),
                    ("v2 says H or A", covered & np.isin(labels, ["H", "A"])),
                    ("v2 says D", covered & (labels == "D"))):
        print("{:40s} frames {:6d}  broadcast not live {:5.1f}%".format(name, int(m.sum()), 100 * nonlive[m].mean()))

    # not-live stretches: how long, and what v2 says inside
    e = np.flatnonzero(np.diff(np.concatenate(([0], nonlive.astype(np.int8), [0]))))
    st, en = e[::2], e[1::2]
    L = (en - st) / FPS
    print("not-live stretches: {}, length median {:.1f} s, p75 {:.1f}, p90 {:.1f}".format(
        len(L), np.median(L), np.percentile(L, 75), np.percentile(L, 90)))
    for lo, hi in ((0, 2), (2, 5), (5, 10), (10, 30), (30, 1e9)):
        k = (L >= lo) & (L < hi)
        if k.any():
            frames = np.concatenate([np.arange(a, b) for a, b in zip(st[k], en[k])])
            print("   {:>4.0f}-{:<4.0f} s: {:4d} stretches, {:5.1f}% of not-live frames, v2 D on {:4.1f}%".format(
                lo, min(hi, 999), int(k.sum()), 100 * len(frames) / nonlive.sum(),
                100 * np.mean(labels[frames] == "D")))
    rg = rep & g
    print("replays: {} frames, v2 D on {:.1f}%".format(int(rg.sum()), 100 * np.mean(labels[rg] == "D")))
    # replay stretches with v2 alive for >= 2 s inside
    e = np.flatnonzero(np.diff(np.concatenate(([0], rg.astype(np.int8), [0]))))
    k = sum(np.sum(labels[a:b] != "D") >= 2 * FPS for a, b in zip(e[::2], e[1::2]))
    print("replay stretches with >= 2 s of H/A in v2: {} of {}".format(k, len(e) // 2))


if __name__ == "__main__":
    main()
