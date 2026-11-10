"""Ball detector on one 640x640 crop per frame, centred where the ball should be. Aggregates only.

Centre: the ATD ball when it has one (measures recall and agreement), else the mean of the 6
tracked players closest to the last ATD sighting (what we would do inside a gap).

    .venv-cv/bin/python analisi/prova_palla_ritaglio.py --first 68185 --frames 1500 --every 2
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "solution", "retrack"))
import pilot  # noqa: E402

CROP = 640


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=68185)
    ap.add_argument("--frames", type=int, default=1500)
    ap.add_argument("--every", type=int, default=2)
    a = ap.parse_args()
    from ultralytics import YOLO
    m = YOLO(os.path.join(pilot.MODELS, "football-ball-detection.pt"))
    Hs = pilot.load_h()
    z = np.load(pilot.ATD_CACHE)
    pl, ball = z["players"], z["ball"]
    L, Wd = pilot.PITCH

    def px(f, x, y):
        q = Hs[f] @ np.array([x / L, y / Wd, 1.0])
        return q[0] / q[2], q[1] / q[2]

    rows = []   # frame, atd_seen, found, dist_m, nearest_player_m, centre_kind
    last = None
    t0 = time.time()
    for k, im in enumerate(pilot.frames(a.first, a.frames)):
        f = a.first + k
        if not np.isnan(ball[f, 0]):
            last = ball[f].copy()
        if k % a.every:
            continue
        seen = not np.isnan(ball[f, 0])
        if seen:
            cx, cy = px(f, *ball[f])
            kind = 0
        else:
            P = pl[f][~np.isnan(pl[f, :, 0])]
            if not len(P):
                continue
            ref = last if last is not None else P.mean(0)
            near = P[np.argsort(np.linalg.norm(P - ref, axis=1))[:6]].mean(0)
            cx, cy = px(f, *near)
            kind = 1
        x0 = int(np.clip(cx - CROP / 2, 0, pilot.W_IMG - CROP))
        y0 = int(np.clip(cy - CROP / 2, 0, pilot.H_IMG - CROP))
        crop = np.ascontiguousarray(im[y0:y0 + CROP, x0:x0 + CROP])
        r = m.predict(crop, imgsz=CROP, conf=0.25, device="mps", verbose=False)[0]
        found, dist, nearest = 0, np.nan, np.nan
        if len(r.boxes):
            j = int(np.argmax(r.boxes.conf.cpu().numpy()))
            b = r.boxes.xyxy.cpu().numpy()[j]
            bx, by = (b[0] + b[2]) / 2 + x0, (b[1] + b[3]) / 2 + y0
            xm, ym = pilot.to_pitch(Hs[f], np.array([bx]), np.array([by]))
            found = 1
            if seen:
                dist = float(np.hypot(xm[0] - ball[f, 0], ym[0] - ball[f, 1]))
            P = pl[f][~np.isnan(pl[f, :, 0])]
            if len(P):
                nearest = float(np.min(np.linalg.norm(P - np.array([xm[0], ym[0]]), axis=1)))
        rows.append((f, seen, found, dist, nearest, kind))
    R = np.array(rows, float)
    dt = time.time() - t0
    print("frames checked {} in {:.0f} s ({:.1f} checks/s incl. decoding)".format(len(R), dt, len(R) / dt))
    s, g = R[R[:, 1] == 1], R[R[:, 1] == 0]
    if len(s):
        d = s[s[:, 2] == 1, 3]
        print("ATD has the ball ({} checks): found {:.0f}%, within 2 m of the ATD ball {:.0f}% of those, median {:.2f} m".format(
            len(s), 100 * s[:, 2].mean(), 100 * np.mean(d < 2) if len(d) else 0, np.median(d) if len(d) else np.nan))
    if len(g):
        fg = g[g[:, 2] == 1]
        print("ATD gap ({} checks): found {:.0f}%; nearest player median {:.1f} m, within 3 m of a player {:.0f}%".format(
            len(g), 100 * g[:, 2].mean(), np.median(fg[:, 4]) if len(fg) else np.nan,
            100 * np.mean(fg[:, 4] < 3) if len(fg) else 0))
        # temporal coherence of the found balls inside gaps: consecutive checks
        np.save(os.path.join(pilot.OUT_DIR, "crop_{}.npy".format(a.first)), R)


if __name__ == "__main__":
    main()
