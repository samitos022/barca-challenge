"""Audit of the provided tracking (ATD): where it is imprecise, aggregate numbers only, no labels.

Every check is a physical or structural impossibility, or a strong regularity of football, measured
on the graded frames. Nothing frame-level is printed.

    .venv/bin/python analisi/audit_tracking.py [--source tactical|broadcast]
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
GOAL_Y = W / 2


def runs(mask):
    """(starts, ends) of True runs, end exclusive."""
    e = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return e[::2], e[1::2]


def pct(x):
    return 100.0 * float(np.mean(x)) if np.size(x) else float("nan")


def main(source="tactical"):
    teams, pl, ball, bs, names = tracking.load(GAME, source)
    n = len(pl)
    if source == "broadcast":
        blocks = [(0, 0, n - 1)]
    else:
        blocks = half_video_blocks(GAME)
    g = np.zeros(n, bool)
    for _, a, z in blocks:
        g[a:z + 1] = True
    seen = ~np.isnan(pl[:, :, 0])
    print("== {} tracking, {} slots, {} graded frames ({})".format(source, pl.shape[1], int(g.sum()), names))

    # ---- 1. players per team -------------------------------------------------------------------
    print("\n1. players per team (graded)")
    for k in (0, 1):
        c = seen[:, teams == k].sum(axis=1)[g]
        h = {v: pct(c == v) for v in range(8, 14)}
        print("   team {} ({}): mean {:.2f}; <=9 {:.1f}%  10 {:.1f}%  11 {:.1f}%  12 {:.1f}%  >=13 {:.1f}%".format(
            k, names[k], c.mean(), pct(c <= 9), h[10], h[11], h[12], pct(c >= 13)))
    tot = seen.sum(axis=1)[g]
    print("   all objects: mean {:.1f}; 22 exactly {:.1f}%, > 22 {:.1f}%, < 20 {:.1f}%".format(
        tot.mean(), pct(tot == 22), pct(tot > 22), pct(tot < 20)))

    # ---- 2. goalkeepers --------------------------------------------------------------------------
    print("\n2. goalkeepers (which goal each team defends: side of its mean x in each half)")
    if source == "tactical":
        gk_missing, gk_other, deep = [], [], []
        for half, a, z in blocks:
            seg = slice(a, z + 1)
            mx = [np.nanmean(pl[seg][:, teams == k, 0]) for k in (0, 1)]
            own_x = [0.0 if mx[k] < L / 2 else L for k in (0, 1)]  # goal line each team defends
            bx = ball[seg, 0]
            for k in (0, 1):
                xs = pl[seg][:, teams == k, 0]
                ys = pl[seg][:, teams == k, 1]
                dgl = np.abs(xs - own_x[k])                      # distance to own goal line
                deepest = np.nanmin(np.where(np.isnan(dgl), np.inf, dgl), axis=1)
                deep.append(deepest)
                ball_far = np.nan_to_num(np.abs(bx - own_x[k]), nan=0.0) > 40  # ball in the other half
                # no own player within 20 m of the own goal line while the ball is far away:
                # the goalkeeper is not tracked (or carried by the other team)
                miss = ball_far & (deepest > 20)
                gk_missing.append((half, k, miss.sum(), ball_far.sum()))
                # a player of the OTHER team standing in this team's goal area (6 m of the goal
                # centre) with the ball > 40 m away: almost surely this team's keeper, wrong team
                o = 1 - k
                xo, yo = pl[seg][:, teams == o, 0], pl[seg][:, teams == o, 1]
                dgoal = np.hypot(xo - own_x[k], yo - GOAL_Y)
                in_box = np.nanmin(np.where(np.isnan(dgoal), np.inf, dgoal), axis=1) < 6
                wrong = ball_far & in_box
                gk_other.append((half, k, wrong.sum(), ball_far.sum(), (miss & wrong).sum()))
        for (half, k, m, bf), (_, _, w, _, mw) in zip(gk_missing, gk_other):
            print("   half {} team {}: ball in the far half on {} frames; no own player within 20 m of own goal "
                  "{:.1f}%; an opponent in its goal area {:.1f}% (both {:.1f}%)".format(
                      half, k, bf, 100 * m / max(bf, 1), 100 * w / max(bf, 1), 100 * mw / max(bf, 1)))
        dd = np.concatenate(deep)
        print("   deepest player to own goal line, all frames: median {:.1f} m, p90 {:.1f} m".format(
            np.median(dd[np.isfinite(dd)]), np.percentile(dd[np.isfinite(dd)], 90)))

    # ---- 3. track fragmentation ------------------------------------------------------------------
    print("\n3. track continuity")
    lens, starts_mid, starts = [], 0, 0
    for s in range(pl.shape[1]):
        st, en = runs(seen[:, s] & g)
        lens.extend((en - st).tolist())
        for a in st:
            starts += 1
            x, y = pl[a, s]
            if 5 < x < L - 5 and 5 < y < W - 5:
                starts_mid += 1
    lens = np.array(lens) / FPS
    minutes = g.sum() / FPS / 60
    print("   tracklets {}, {:.1f} per minute; length median {:.1f} s, p10 {:.1f} s, p90 {:.0f} s; under 10 s {:.0f}%".format(
        len(lens), len(lens) / minutes, np.median(lens), np.percentile(lens, 10), np.percentile(lens, 90),
        pct(lens < 10)))
    print("   tracklets starting inside the pitch (5 m from every line): {:.0f}% -> identity breaks, "
          "not players entering".format(100 * starts_mid / max(starts, 1)))
    # coverage of the 22 players by long tracklets
    print("   slot-frames covered: by tracklets >= 60 s {:.0f}%".format(
        100 * np.sum(lens[lens >= 60]) / max(np.sum(lens), 1)))

    # ---- 4. impossible motion --------------------------------------------------------------------
    print("\n4. impossible motion (players)")
    step = np.linalg.norm(np.diff(pl, axis=0), axis=2)          # m per frame
    gg = g[1:, None] & seen[1:] & seen[:-1]
    st = step[gg]
    print("   steps: median {:.3f} m/frame ({:.1f} m/s); > 0.5 m/frame (12.5 m/s) {:.3f}%; > 1 m/frame {:.3f}% ({} jumps)".format(
        np.median(st), np.median(st) * FPS, pct(st > 0.5), pct(st > 1.0), int(np.sum(st > 1.0))))
    # jitter: residual from the midpoint of +-2 frames for slow players
    mid = (pl[:-4] + pl[4:]) / 2
    res = np.linalg.norm(pl[2:-2] - mid, axis=2)
    spd = np.linalg.norm(pl[4:] - pl[:-4], axis=2) * FPS / 4
    m = g[2:-2, None] & np.isfinite(res) & (spd < 1.0)
    print("   jitter of near-still players (residual from +-2 frames): median {:.3f} m, p90 {:.3f} m".format(
        np.median(res[m]), np.percentile(res[m], 90)))

    # ---- 5. duplicates ---------------------------------------------------------------------------
    print("\n5. duplicates (two tracks of the same team within 0.5 m for >= 1 s)")
    dup = np.zeros(n, bool)
    for k in (0, 1):
        idx = np.flatnonzero(teams == k)
        P = pl[:, idx]
        d = np.linalg.norm(P[:, :, None] - P[:, None], axis=3)
        iu = np.triu_indices(len(idx), 1)
        close = np.nan_to_num(d[:, iu[0], iu[1]], nan=np.inf) < 0.5
        for j in range(close.shape[1]):
            a, b = runs(close[:, j])
            for s0, e0 in zip(a, b):
                if e0 - s0 >= FPS:
                    dup[s0:e0] = True
    print("   frames with a duplicate {:.1f}%".format(pct(dup[g])))
    cross = np.zeros(n, bool)
    i0, i1 = np.flatnonzero(teams == 0), np.flatnonzero(teams == 1)
    d = np.linalg.norm(pl[:, i0][:, :, None] - pl[:, i1][:, None], axis=3)
    cl = np.nan_to_num(d, nan=np.inf).min(axis=(1, 2)) < 0.3
    a, b = runs(cl)
    for s0, e0 in zip(a, b):
        if e0 - s0 >= 2 * FPS:
            cross[s0:e0] = True
    print("   frames with one person carried by both teams (two tracks < 0.3 m for >= 2 s) {:.1f}%".format(pct(cross[g])))

    # ---- 6. ball ---------------------------------------------------------------------------------
    print("\n6. ball")
    bv = ~np.isnan(ball[:, 0])
    print("   seen {:.1f}% of graded frames".format(pct(bv[g])))
    st, en = runs(~bv & g)
    gl = (en - st) / FPS
    print("   gaps {}: median {:.1f} s; frames in gaps >= 5 s {:.0f}%, >= 30 s {:.0f}%".format(
        len(gl), np.median(gl), 100 * gl[gl >= 5].sum() / gl.sum(), 100 * gl[gl >= 30].sum() / gl.sum()))
    # player pace inside gaps: is the ball missing during live play?
    vel = np.linalg.norm(np.gradient(pl, axis=0), axis=2) * FPS
    pace = np.nanmean(np.where(seen, vel, np.nan), axis=1)
    run_gap = [(a0, b0) for a0, b0 in zip(st, en) if b0 - a0 >= 5 * FPS]
    gp = np.array([np.nanmean(pace[a0:b0]) for a0, b0 in run_gap])
    print("   long gaps (>= 5 s): {}; players' mean pace in them median {:.2f} m/s (ball seen: {:.2f} m/s); "
          "gaps with pace > 1.8 m/s (looks like live play) {:.0f}% of the gap frames".format(
              len(run_gap), np.median(gp), np.nanmean(pace[g & bv]),
              100 * sum(b0 - a0 for (a0, b0), p_ in zip(run_gap, gp) if p_ > 1.8) /
              max(sum(b0 - a0 for a0, b0 in run_gap), 1)))
    bstep = np.linalg.norm(np.diff(ball, axis=0), axis=1)
    bb = g[1:] & bv[1:] & bv[:-1]
    print("   ball steps > 1.6 m/frame (40 m/s): {} ({:.2f}% of seen pairs)".format(
        int(np.sum(bstep[bb] > 1.6)), pct(bstep[bb] > 1.6)))
    near = np.nanmin(np.where(seen, np.linalg.norm(pl - ball[:, None], axis=2), np.inf), axis=1)
    bspd = np.zeros(n)
    bspd[1:-1] = np.linalg.norm(ball[2:] - ball[:-2], axis=1) * FPS / 2
    slow_far = g & bv & (np.nan_to_num(bspd, nan=99) < 1.0) & (near > 15)
    print("   ball still (< 1 m/s) and > 15 m from every player: {:.1f}% of seen frames (a wrong object, "
          "or a ball lying out of play)".format(100 * slow_far.sum() / max((g & bv).sum(), 1)))
    stuck = np.zeros(n, bool)
    a, b = runs(g & bv & (np.nan_to_num(bspd, nan=99) < 0.3))
    for s0, e0 in zip(a, b):
        if e0 - s0 >= 5 * FPS:
            stuck[s0:e0] = True
    print("   ball not moving at all (< 0.3 m/s) for >= 5 s: {:.1f}% of seen frames".format(
        100 * stuck.sum() / max((g & bv).sum(), 1)))
    off = g & bv & ((ball[:, 0] < -2) | (ball[:, 0] > L + 2) | (ball[:, 1] < -2) | (ball[:, 1] > W + 2))
    print("   ball more than 2 m off the pitch: {:.1f}% of seen frames".format(100 * off.sum() / max((g & bv).sum(), 1)))
    near_ok = g & bv & np.isfinite(near)
    print("   nearest player to the ball when seen: median {:.1f} m, p90 {:.1f} m".format(
        np.median(near[near_ok]), np.percentile(near[near_ok], 90)))
    m = g & bv & np.isfinite(bs) & np.isfinite(bspd)
    if m.any():
        print("   ATD ball speed vs speed from positions: median abs diff {:.2f} m/s".format(
            np.median(np.abs(bs[m] - bspd[m]))))
    # where is the ball lost: last position before a long gap
    last = np.array([a0 - 1 for a0, _ in run_gap if a0 > 0 and bv[a0 - 1]])
    if len(last):
        lx, ly = ball[last, 0], ball[last, 1]
        dline = np.minimum.reduce([lx, L - lx, ly, W - ly])
        box = ((lx < 16.5) | (lx > L - 16.5)) & (np.abs(ly - GOAL_Y) < 20.16)
        print("   last sighting before a long gap: within 3 m of a line {:.0f}%, inside a penalty area {:.0f}%, "
              "elsewhere {:.0f}%".format(pct(dline < 3), pct(box & (dline >= 3)), pct(~box & (dline >= 3))))

    # ---- 7. homography ---------------------------------------------------------------------------
    print("\n7. homography")
    h = np.loadtxt(asset_paths(GAME, source=source)["homography"], delimiter=",", skiprows=1, usecols=(10, 11))
    err, valid = h[:, 0], h[:, 1] == 1
    gm = g[:len(err)]
    e = err[gm & valid]
    print("   valid {:.1f}%; reprojection error median {:.2f}, p90 {:.2f}, p99 {:.2f}, > 2 px {:.2f}%".format(
        pct(valid[gm]), np.median(e), np.percentile(e, 90), np.percentile(e, 99), pct(e > 2)))
    H = np.loadtxt(asset_paths(GAME, source=source)["homography"], delimiter=",", skiprows=1, usecols=range(1, 10))
    Hg = H[gm & valid]
    dH = np.abs(np.diff(Hg[:, [0, 4]], axis=0)).max(axis=1)
    print("   camera motion: frames where the homography changes (> 0.1% of h00/h11) {:.0f}%".format(
        pct(dH > 1e-3 * np.abs(Hg[1:, [0, 4]]).max(axis=1))))


if __name__ == "__main__":
    main("broadcast" if "--source" in sys.argv and sys.argv[-1] == "broadcast" else "tactical")
