"""Recover the ball where the ATD has none: Roboflow ball detector on a 640x640 crop, then filters.

Two steps (the first needs the vision environment, the second only numpy):

    .venv-cv/bin/python solution/retrack/ball_gaps.py detect      # hours, resumable (checkpoints)
    .venv/bin/python    solution/retrack/ball_gaps.py filter      # seconds, writes the gap fill

detect: streams every graded frame of the tactical video (ffmpeg, as pilot.py) and runs the
detector on
  - every 2nd frame where the ATD ball is missing (kind 1 or 3), crop centred on the last good
    detection if it is recent (kind 3, follows the ball), else on the mean of the 6 tracked
    players closest to the last known ball (kind 1, as analisi/prova_palla_ritaglio.py);
  - every PREC_EVERY-th frame where the ATD DOES see the ball (kind 2), crop centred the way a gap
    would centre it (6 players closest to the ATD ball of LAG_PREC frames earlier): the agreement
    with the ATD there is the precision proxy, no labels involved.
All boxes above CONF_MIN are kept (raw cache, data/derived/retrack/, never printed).

filter: physical consistency only (pitch with margin, near a tracked player, speed between
consecutive accepted detections, isolated detections rejected), then linear interpolation of
short holes between accepted detections. Output data/derived/retrack/BAR-ATH_ball_gapfill.npz:
ball (N, 2) with the ATD ball where it has one and the recovered ball in the gaps, plus the
masks. Only aggregate numbers are printed.
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pilot  # noqa: E402

FPS = pilot.FPS
CROP = 640
CONF_MIN = 0.10          # raw cache threshold; the filter decides
EVERY = 2                # gap frames checked
PREC_EVERY = 10          # ATD-seen frames checked for the precision proxy
LAG_PREC = 25            # stale reference for the precision crops (1 s), like inside a gap
FOLLOW_S = 0.5           # a detection this recent centres the next crop
FOLLOW_CONF = 0.4
MAX_BOX = 5
CHUNK = 1500
RAW = os.path.join(pilot.OUT_DIR, "BAR-ATH_ball_gaps_raw.npz")
OUT = os.path.join(pilot.OUT_DIR, "BAR-ATH_ball_gapfill.npz")

# filter (physics, not tuned on this match)
PITCH_MARGIN = 2.0       # m outside the lines (throw-ins, corners)
NEAR_PLAYER = 3.0        # m: a ball on the ground near a player (ATD players lag ~0.5 m)
CONF_FAR = 0.5           # a ball far from every player needs a confident box AND a chain
V_MAX = 35.0             # m/s, hardest shots ~ 30-35 m/s
POS_SLACK = 1.5          # m, detector + homography error allowed on top of V_MAX * dt
CHAIN_GAP_S = 1.0        # detections further apart in time start a new chain
CHAIN_MIN = 3            # a chain needs >= 3 detections spanning >= CHAIN_MIN_S
CHAIN_MIN_S = 0.2
INTERP_S = 1.0           # holes between accepted detections filled if shorter


def graded_mask(n):
    g = np.zeros(n, bool)
    for _, a, z in pilot.BLOCKS:
        g[a:z + 1] = True
    return g


# ---- detect ------------------------------------------------------------------------------------
def _save(path, checks, dets, next_start):
    tmp = path + ".tmp.npz"
    np.savez(tmp, checks=np.array(checks, np.float64).reshape(-1, 6),
             dets=np.array(dets, np.float64).reshape(-1, 6), next_start=next_start)
    os.replace(tmp, path)


def detect(limit_chunks=None):
    from ultralytics import YOLO
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    m = YOLO(os.path.join(pilot.MODELS, "football-ball-detection.pt"))
    Hs = pilot.load_h()
    z = np.load(pilot.ATD_CACHE)
    pl, ball = z["players"], z["ball"]
    L, Wd = pilot.PITCH
    os.makedirs(pilot.OUT_DIR, exist_ok=True)
    checks, dets, start = [], [], pilot.BLOCKS[0][1]
    if os.path.exists(RAW):
        r = np.load(RAW)
        checks, dets, start = r["checks"].tolist(), r["dets"].tolist(), int(r["next_start"])
        print("resuming at frame {} ({} checks done)".format(start, len(checks)), flush=True)
    chunks = []
    for _, a, zf in pilot.BLOCKS:
        s = max(a, start)
        while s <= zf:
            chunks.append((s, min(CHUNK, zf + 1 - s)))
            s += CHUNK
    if limit_chunks:
        chunks = chunks[:limit_chunks]
    total = len(chunks)

    def px(f, x, y):
        q = Hs[f] @ np.array([x / L, y / Wd, 1.0])
        return q[0] / q[2], q[1] / q[2]

    t0 = time.time()
    n0 = len(checks)
    for ci, (first, count) in enumerate(chunks):
        # state from before the chunk: last ATD sighting (no detections carried across chunks)
        base = max(0, first - 3000)
        prev = np.flatnonzero(~np.isnan(ball[base:first, 0]))
        last = ball[base + prev[-1]].copy() if len(prev) else None
        last_det, last_det_f = None, -10 ** 9
        for k, im in enumerate(pilot.frames(first, count)):
            f = first + k
            seen = not np.isnan(ball[f, 0])
            if seen:
                last = ball[f].copy()
                if f % PREC_EVERY:
                    continue
                ref = ball[f - LAG_PREC] if not np.isnan(ball[f - LAG_PREC, 0]) else ball[f]
                kind = 2
            else:
                if f % EVERY:
                    continue
                if last_det is not None and f - last_det_f <= FOLLOW_S * FPS:
                    ref, kind = last_det, 3
                else:
                    ref, kind = last, 1
            P = pl[f][~np.isnan(pl[f, :, 0])]
            if not len(P):
                continue
            if ref is None:
                ref = P.mean(0)
            if kind == 3:
                c = ref
            else:
                c = P[np.argsort(np.linalg.norm(P - ref, axis=1))[:6]].mean(0)
            cx, cy = px(f, *c)
            if not (np.isfinite(cx) and np.isfinite(cy)):
                continue
            x0 = int(np.clip(cx - CROP / 2, 0, pilot.W_IMG - CROP))
            y0 = int(np.clip(cy - CROP / 2, 0, pilot.H_IMG - CROP))
            crop = np.ascontiguousarray(im[y0:y0 + CROP, x0:x0 + CROP])
            r = m.predict(crop, imgsz=CROP, conf=CONF_MIN, device=device, verbose=False)[0]
            nb = 0
            if len(r.boxes):
                conf = r.boxes.conf.cpu().numpy()
                xyxy = r.boxes.xyxy.cpu().numpy()
                order = np.argsort(-conf)[:MAX_BOX]
                bx = (xyxy[order, 0] + xyxy[order, 2]) / 2 + x0
                by = (xyxy[order, 1] + xyxy[order, 3]) / 2 + y0
                xm, ym = pilot.to_pitch(Hs[f], bx, by)
                for j, o in enumerate(order):
                    size = max(xyxy[o, 2] - xyxy[o, 0], xyxy[o, 3] - xyxy[o, 1])
                    dets.append((f, kind, xm[j], ym[j], conf[o], size))
                nb = len(order)
                if kind != 2:
                    j = 0
                    on = -PITCH_MARGIN <= xm[j] <= L + PITCH_MARGIN and -PITCH_MARGIN <= ym[j] <= Wd + PITCH_MARGIN
                    near = np.min(np.linalg.norm(P - np.array([xm[j], ym[j]]), axis=1)) < NEAR_PLAYER
                    if conf[order[0]] >= FOLLOW_CONF and on and near:
                        last_det, last_det_f = np.array([xm[j], ym[j]]), f
                        last = last_det.copy()
            checks.append((f, kind, c[0], c[1], nb, x0 * 10000 + y0))
        _save(RAW, checks, dets, first + count)
        el = time.time() - t0
        done = len(checks) - n0
        print("chunk {}/{} up to frame {}: {} checks, {:.1f} checks/s, elapsed {:.0f} min".format(
            ci + 1, total, first + count - 1, len(checks), done / max(el, 1e-9), el / 60), flush=True)
    print("detect done: {} checks, {} boxes".format(len(checks), len(dets)), flush=True)


# ---- filter ------------------------------------------------------------------------------------
def _runs(mask):
    e = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return e[::2], e[1::2]


def candidates(dets, pl, kinds):
    """Best box per checked frame passing the static gates: (frames, xy, conf)."""
    L, Wd = pilot.PITCH
    d = dets[np.isin(dets[:, 1], kinds)]
    on = (d[:, 2] >= -PITCH_MARGIN) & (d[:, 2] <= L + PITCH_MARGIN) & \
         (d[:, 3] >= -PITCH_MARGIN) & (d[:, 3] <= Wd + PITCH_MARGIN)
    d = d[on]
    f = d[:, 0].astype(int)
    P = pl[f]
    dist = np.linalg.norm(P - d[:, None, 2:4], axis=2)
    near = np.nanmin(np.where(np.isnan(dist), np.inf, dist), axis=1)
    keep = (near < NEAR_PLAYER) | (d[:, 4] >= CONF_FAR)
    d, near = d[keep], near[keep]
    # best per frame: highest confidence
    o = np.lexsort((-d[:, 4], d[:, 0]))
    d, near = d[o], near[o]
    first = np.concatenate(([True], d[1:, 0] != d[:-1, 0]))
    return d[first, 0].astype(int), d[first, 2:4], d[first, 4], near[first]


def chain_filter(fr, xy, near):
    """Temporal chains over the per-frame candidates (time order): a candidate joins the open chain
    if reachable at V_MAX (+ POS_SLACK) from its last member, otherwise it is an outlier and
    dropped; a time hole over CHAIN_GAP_S closes the chain. A chain is kept with >= CHAIN_MIN
    members spanning >= CHAIN_MIN_S and at least one member near a player."""
    acc = np.zeros(len(fr), bool)
    chain = []

    def close():
        if len(chain) >= CHAIN_MIN and (fr[chain[-1]] - fr[chain[0]]) / FPS >= CHAIN_MIN_S \
                and np.any(near[chain] < NEAR_PLAYER):
            acc[chain] = True

    for i in range(len(fr)):
        if chain and fr[i] - fr[chain[-1]] > CHAIN_GAP_S * FPS:
            close()
            chain = []
        if not chain:
            chain = [i]
            continue
        dt = (fr[i] - fr[chain[-1]]) / FPS
        if np.linalg.norm(xy[i] - xy[chain[-1]]) <= V_MAX * dt + POS_SLACK:
            chain.append(i)
        elif len(chain) == 1:
            chain = [i]          # a lone start that the next one cannot reach: restart there
    if chain:
        close()
    return acc


def build(fr, xy, n):
    """Accepted detections -> (n, 2) ball with short holes interpolated; source mask 1 det, 2 interp."""
    out = np.full((n, 2), np.nan, np.float32)
    src = np.zeros(n, np.int8)
    out[fr] = xy
    src[fr] = 1
    for a, b in zip(fr[:-1], fr[1:]):
        if 1 < b - a <= INTERP_S * FPS:
            w = (np.arange(a + 1, b) - a) / (b - a)
            out[a + 1:b] = xy[fr == a][0] * (1 - w[:, None]) + xy[fr == b][0] * w[:, None]
            src[a + 1:b] = 2
    return out, src


def filt():
    z = np.load(pilot.ATD_CACHE)
    pl, ball = z["players"], z["ball"]
    n = len(ball)
    r = np.load(RAW)
    checks, dets = r["checks"], r["dets"]
    g = graded_mask(n)
    print("raw: {} checks (gap {}, precision {}), {} boxes, scanned up to frame {}".format(
        len(checks), int(np.sum(checks[:, 1] != 2)), int(np.sum(checks[:, 1] == 2)), len(dets), int(r["next_start"])))
    atd = ~np.isnan(ball[:, 0])
    done = np.zeros(n, bool)
    done[:int(r["next_start"])] = True
    done &= g

    # precision proxy on ATD-seen frames: same gates, same chain logic on the sampled frames
    fr2, xy2, c2, nr2 = candidates(dets, pl, [2])
    pc = checks[checks[:, 1] == 2]
    d2 = np.linalg.norm(xy2 - ball[fr2], axis=1)
    print("precision proxy (ATD sees the ball, crop centred as in a gap, {} checks): a box passing the "
          "static gates on {:.0f}%; within 2 m of the ATD ball {:.0f}%, within 3 m {:.0f}%, median {:.2f} m".format(
              len(pc), 100 * len(fr2) / max(len(pc), 1), 100 * np.mean(d2 < 2), 100 * np.mean(d2 < 3), np.median(d2)))
    for lo in (0.25, 0.4, 0.6):
        s = c2 >= lo
        if s.any():
            print("   conf >= {:.2f}: {:.0f}% of checks, within 2 m {:.0f}%".format(
                lo, 100 * s.sum() / len(pc), 100 * np.mean(d2[s] < 2)))

    acc2 = chain_filter(fr2, xy2, nr2)
    if acc2.any():
        print("   after the chain filter (sampled every {} frames): kept {:.0f}% of the checks, within 2 m {:.0f}%, "
              "within 3 m {:.0f}%".format(PREC_EVERY, 100 * acc2.sum() / max(len(pc), 1), 100 * np.mean(d2[acc2] < 2),
                                          100 * np.mean(d2[acc2] < 3)))
    # false positives with the ball NOT in the crop (ATD ball projects outside it): any box kept
    # there is a wrong object (or an ATD error); this is the case of a gap crop that missed the ball
    Hs = pilot.load_h()
    L, Wd = pilot.PITCH
    pf = pc[:, 0].astype(int)
    q = np.einsum("nij,nj->ni", Hs[pf], np.stack([ball[pf, 0] / L, ball[pf, 1] / Wd, np.ones(len(pf))], 1))
    bu, bv = q[:, 0] / q[:, 2], q[:, 1] / q[:, 2]
    x0, y0 = pc[:, 5] // 10000, pc[:, 5] % 10000
    inside = (bu >= x0) & (bu < x0 + CROP) & (bv >= y0) & (bv < y0 + CROP)
    kept_static = np.isin(pf, fr2)
    kept_chain = np.isin(pf, fr2[acc2])
    print("   ball inside the crop on {:.0f}% of the precision checks; with the ball OUTSIDE the crop ({} checks) "
          "a box is kept on {:.1f}% (static gates) / {:.1f}% (chain): false-positive rate per check".format(
              100 * inside.mean(), int((~inside).sum()), 100 * kept_static[~inside].mean() if (~inside).any() else np.nan,
              100 * kept_chain[~inside].mean() if (~inside).any() else np.nan))
    print("   with the ball inside the crop: found (chain) {:.0f}%, within 2 m {:.0f}%".format(
        100 * kept_chain[inside].mean(), 100 * np.mean(d2[acc2][np.isin(fr2[acc2], pf[inside])] < 2)))
    fr, xy, conf, near = candidates(dets, pl, [1, 3])
    gc = checks[checks[:, 1] != 2]
    print("gap checks {}: a box passing the static gates on {:.0f}%".format(len(gc), 100 * len(fr) / max(len(gc), 1)))
    acc = chain_filter(fr, xy, near)
    print("   accepted after the temporal chain filter: {} ({:.0f}% of candidates)".format(int(acc.sum()), 100 * acc.mean()))
    rec, src = build(fr[acc], xy[acc], n)
    src[atd] = 0
    rec[atd] = np.nan
    fill = np.where(atd[:, None], ball, rec).astype(np.float32)
    seen_after = ~np.isnan(fill[:, 0])
    print("ball coverage of graded frames (scanned part): ATD {:.1f}% -> {:.1f}% ({:+.1f} points; detections "
          "{:.1f}, interpolated {:.1f})".format(100 * atd[done].mean(), 100 * seen_after[done].mean(),
                                                100 * (seen_after[done].mean() - atd[done].mean()),
                                                100 * np.mean(src[done] == 1), 100 * np.mean(src[done] == 2)))
    # coverage counting each checked gap frame as covering its unchecked neighbour (every 2nd frame)
    np.savez(OUT, ball=fill, recovered=(src > 0), source=src, scanned=done)
    print("saved", OUT)
    return fill, src, done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["detect", "filter"])
    ap.add_argument("--chunks", type=int, default=None, help="detect: only the first N chunks (test)")
    a = ap.parse_args()
    if a.step == "detect":
        detect(a.chunks)
    else:
        filt()


if __name__ == "__main__":
    main()
