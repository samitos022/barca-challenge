"""Pilot of an independent tracking of the tactical video, compared with the provided ATD.

Runs locally (nothing leaves the computer) in the separate environment .venv-cv:

    .venv-cv/bin/python solution/retrack/pilot.py --auto            # two 1-minute windows
    .venv-cv/bin/python solution/retrack/pilot.py --first 30000 --frames 1500

Models (public, see analisi/ricerca_tracking.md), in models/:
  - people: Roboflow sports football-player-detection.pt (YOLOv8, classes ball / goalkeeper /
    player / referee), at imgsz 1280 on the full frame;
  - ball: WASB-SBDT HRNet trained on the Soccer set (ISSIA-CNR, wide fixed 1080p cameras), three
    consecutive frames in, three heatmaps out, at 512 x 288.
Pixels go to the pitch with the PROVIDED homography (inverse), people at the bottom centre of the
box. Only aggregate numbers are printed; per-frame detections are cached in
data/derived/retrack/ (git-ignored, never published).

Windows (--auto): picked by an algorithm from the ATD alone, outside the public five minutes:
the minute where the ATD ball is missing most while the players run (ball lost during live play),
and the minute closest to the match median of ball visibility.
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
WASB_SRC = os.path.join(ROOT, "third_party", "WASB-SBDT", "src")
MODELS = os.path.join(ROOT, "models")
OUT_DIR = os.path.join(ROOT, "data", "derived", "retrack")
VIDEO = os.path.join(ROOT, "data", "tactical", "BAR-ATH_tactical.mp4")
HOMOGRAPHY = os.path.join(ROOT, "data", "tactical", "BAR-ATH_tactical_homography.csv")
ATD_CACHE = os.path.join(ROOT, "data", "derived", "BAR-ATH_tactical_tracking.npz")

FPS = 25
W_IMG, H_IMG = 1920, 1080
PITCH = (105.3, 68.0)
BLOCKS = ((1, 11185, 90651), (2, 110808, 190197))
PUBLIC = (11185, 18684)
CLASSES = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}
WASB_THRESHOLD = 0.5    # heatmap score, the WASB default


# ---- video ------------------------------------------------------------------------------------
def frames(first, count):
    """Yield RGB frames (H, W, 3) uint8 of the tactical video, first .. first + count - 1."""
    cmd = ["ffmpeg", "-v", "error", "-hwaccel", "videotoolbox", "-ss", "{:.3f}".format(first / FPS),
           "-i", VIDEO, "-frames:v", str(count), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=W_IMG * H_IMG * 3 * 4)
    size = W_IMG * H_IMG * 3
    for _ in range(count):
        buf = p.stdout.read(size)
        if len(buf) < size:
            break
        yield np.frombuffer(buf, np.uint8).reshape(H_IMG, W_IMG, 3)
    p.stdout.close()
    p.wait()


# ---- geometry ---------------------------------------------------------------------------------
def load_h():
    h = np.loadtxt(HOMOGRAPHY, delimiter=",", skiprows=1, usecols=range(1, 10))
    return h.reshape(-1, 3, 3)


def to_pitch(Hm, px, py):
    """Pixels -> metres through the inverse of the provided pitch(0-1) -> pixel homography."""
    Hi = np.linalg.inv(Hm)
    q = Hi @ np.vstack([px, py, np.ones_like(px)])
    return q[0] / q[2] * PITCH[0], q[1] / q[2] * PITCH[1]


# ---- models -----------------------------------------------------------------------------------
def load_people(device):
    """CoreML export (NMS inside the model, Neural Engine / GPU) when present: on MPS the PyTorch
    NMS overran Ultralytics' time limit and detections were silently dropped."""
    from ultralytics import YOLO
    ml = os.path.join(MODELS, "football-player-detection.mlpackage")
    if os.path.exists(ml):
        return YOLO(ml, task="detect"), None
    return YOLO(os.path.join(MODELS, "football-player-detection.pt")), device


def pitch_mask(Hm, margin=3.0):
    """Pixels of the pitch plus a margin (throw-ins, corners), from the provided homography."""
    import cv2
    L, Wd = PITCH
    xs = np.concatenate([np.linspace(-margin, L + margin, 40), np.full(20, L + margin),
                         np.linspace(L + margin, -margin, 40), np.full(20, -margin)])
    ys = np.concatenate([np.full(40, -margin), np.linspace(-margin, Wd + margin, 20),
                         np.full(40, Wd + margin), np.linspace(Wd + margin, -margin, 20)])
    q = Hm @ np.vstack([xs / L, ys / Wd, np.ones_like(xs)])
    ok = q[2] > 0
    pts = np.stack([q[0, ok] / q[2, ok], q[1, ok] / q[2, ok]], 1)
    pts = np.clip(pts, -4 * W_IMG, 5 * W_IMG).astype(np.int32)
    m = np.zeros((H_IMG, W_IMG), np.uint8)
    cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def load_wasb(device):
    import torch
    import yaml
    sys.path.insert(0, WASB_SRC)
    from models.hrnet import HRNet
    with open(os.path.join(WASB_SRC, "configs", "model", "wasb.yaml")) as f:
        cfg = yaml.safe_load(f)

    class Cfg(dict):
        """The WASB code reads its (OmegaConf) config both as cfg['x'] and as cfg.x."""
        def __getattr__(self, k):
            v = self[k]
            return Cfg(v) if isinstance(v, dict) else v

        def __getitem__(self, k):
            v = dict.__getitem__(self, k)
            return Cfg(v) if isinstance(v, dict) else v

    net = HRNet(Cfg(cfg))
    ck = torch.load(os.path.join(MODELS, "wasb_soccer_best.pth.tar"), map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck)
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    net.load_state_dict(sd)
    return net.to(device).eval()


def wasb_affine():
    """The WASB test transform: whole frame, centred, longest side scaled to 512."""
    import cv2
    sys.path.insert(0, WASB_SRC)
    from utils.image import get_affine_transform
    c = np.array([W_IMG / 2.0, H_IMG / 2.0], np.float32)
    s = max(W_IMG, H_IMG) * 1.0
    fwd = get_affine_transform(c, s, 0, [512, 288])
    inv = get_affine_transform(c, s, 0, [512, 288], inv=1)
    return fwd, inv, cv2


def wasb_ball(net, triple, fwd, inv, cv2, device, masks=None):
    """3 RGB frames -> per frame list of (x_px, y_px, score) blobs above the threshold. With masks,
    everything off the pitch is painted grey first (the ISSIA-trained model fires on the stands)."""
    import torch
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    xs = []
    for k, im in enumerate(triple):
        if masks is not None:
            im = np.where(masks[k][:, :, None], im, np.uint8(128))
        w = cv2.warpAffine(im, fwd, (512, 288), flags=cv2.INTER_LINEAR)
        xs.append(((w.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1))
    x = torch.from_numpy(np.concatenate(xs, 0)[None]).to(device)
    with torch.no_grad():
        hm = net(x)[0].sigmoid()[0].cpu().numpy()   # (3, 288, 512)
    out = []
    for k in range(hm.shape[0]):
        h = hm[k]
        blobs = []
        if h.max() > WASB_THRESHOLD:
            _, th = cv2.threshold(h, WASB_THRESHOLD, 1, cv2.THRESH_BINARY)
            nl, lab = cv2.connectedComponents(th.astype(np.uint8))
            for m in range(1, nl):
                ys, xs_ = np.where(lab == m)
                ws = h[ys, xs_]
                bx, by = (xs_ * ws).sum() / ws.sum(), (ys * ws).sum() / ws.sum()
                p = inv @ np.array([bx, by, 1.0])
                blobs.append((p[0], p[1], float(ws.sum())))
        out.append(blobs)
    return out


# ---- windows ----------------------------------------------------------------------------------
def auto_windows(length=1500):
    z = np.load(ATD_CACHE)
    pl, ball = z["players"], z["ball"]
    seen = ~np.isnan(pl[:, :, 0])
    vel = np.linalg.norm(np.gradient(pl, axis=0), axis=2) * FPS
    pace = np.nanmean(np.where(seen, vel, np.nan), axis=1)
    bv = ~np.isnan(ball[:, 0])
    cands = []
    for _, a, z_ in BLOCKS:
        for s in range(a, z_ - length, length // 2):
            if s <= PUBLIC[1] and s + length > PUBLIC[0]:
                continue
            seg = slice(s, s + length)
            lost_live = np.mean(~bv[seg] & (np.nan_to_num(pace[seg]) > 1.8))
            cands.append((s, lost_live, bv[seg].mean()))
    cands = np.array(cands)
    hard = int(cands[np.argmax(cands[:, 1]), 0])
    med = np.median(cands[:, 2])
    typical = int(cands[np.argmin(np.abs(cands[:, 2] - med)), 0])
    return [("hard: ball lost in live play", hard), ("typical", typical)]


# ---- run --------------------------------------------------------------------------------------
def run_window(first, count, people, wasb, device, Hs):
    import cv2  # noqa: F401
    fwd, inv, cv2m = wasb_affine()
    yolo, dev = people
    det_people = []   # (frame, cls, x_m, y_m, conf, x_px, y_px)
    det_ball = []     # (frame, x_m, y_m, score, source) source 0 wasb, 1 yolo
    t_people = t_ball = 0.0
    buf, idx = [], first
    batch = []
    t0 = time.time()
    for im in frames(first, count):
        buf.append(im)
        batch.append((idx, im))
        idx += 1
        if len(buf) == 3:
            tb = time.time()
            res = wasb_ball(wasb, buf, fwd, inv, cv2m, device,
                            [pitch_mask(Hs[idx - 3 + k]) for k in range(3)])
            t_ball += time.time() - tb
            for k, blobs in enumerate(res):
                f = idx - 3 + k
                for bx, by, sc in blobs:
                    xm, ym = to_pitch(Hs[f], np.array([bx]), np.array([by]))
                    det_ball.append((f, xm[0], ym[0], sc, 0))
            buf = []
        if len(batch) == (1 if dev is None else 8):
            tp = time.time()
            _people(yolo, dev, batch, Hs, det_people, det_ball)
            t_people += time.time() - tp
            batch = []
    if batch:
        _people(yolo, dev, batch, Hs, det_people, det_ball)
    wall = time.time() - t0
    return np.array(det_people, np.float64).reshape(-1, 7), np.array(det_ball, np.float64).reshape(-1, 5), \
        {"wall": wall, "people": t_people, "ball": t_ball, "frames": idx - first}


def _people(yolo, dev, batch, Hs, det_people, det_ball):
    kw = {} if dev is None else {"device": dev}
    res = yolo.predict([im for _, im in batch], imgsz=1280, conf=0.3, verbose=False, **kw)
    for (f, _), r in zip(batch, res):
        b = r.boxes
        if b is None or not len(b):
            continue
        xyxy = b.xyxy.cpu().numpy()
        cls = b.cls.cpu().numpy().astype(int)
        conf = b.conf.cpu().numpy()
        fx = (xyxy[:, 0] + xyxy[:, 2]) / 2
        fy = np.where(cls == 0, (xyxy[:, 1] + xyxy[:, 3]) / 2, xyxy[:, 3])
        xm, ym = to_pitch(Hs[f], fx, fy)
        for c, x, y, cf, px, py in zip(cls, xm, ym, conf, fx, fy):
            if c == 0:
                det_ball.append((f, x, y, cf, 1))
            else:
                det_people.append((f, c, x, y, cf, px, py))


def compare(name, first, count, P, B, stats):
    """Aggregate comparison with the ATD over the window."""
    z = np.load(ATD_CACHE)
    pl, ball, teams = z["players"], z["ball"], z["teams"]
    fr = np.arange(first, first + stats["frames"])
    n = len(fr)
    print("\n== {} (frames {}..{}, {} s)".format(name, first, first + n - 1, n // FPS))
    print("speed: {:.1f} frames/s overall (people {:.1f} fps, ball {:.1f} fps of model time)".format(
        n / stats["wall"], n / max(stats["people"], 1e-9), n / max(stats["ball"], 1e-9)))
    # people
    cnt = {c: np.array([np.sum((P[:, 0] == f) & (P[:, 1] == c)) for f in fr]) for c in (1, 2, 3)}
    atd_n = (~np.isnan(pl[fr, :, 0])).sum(axis=1)
    print("people per frame: players {:.1f}, goalkeepers {:.2f}, referees {:.2f}; ATD objects {:.1f}".format(
        cnt[2].mean(), cnt[1].mean(), cnt[3].mean(), atd_n.mean()))
    print("frames with >= 1 goalkeeper detected {:.0f}%".format(100 * np.mean(cnt[1] > 0)))
    on = (P[:, 2] > -3) & (P[:, 2] < PITCH[0] + 3) & (P[:, 3] > -3) & (P[:, 3] < PITCH[1] + 3)
    print("people detections on the pitch (3 m margin) {:.0f}%".format(100 * on.mean() if len(on) else 0))
    # position agreement: each detected player/keeper to the nearest ATD object
    dd = []
    unmatched_atd = []
    for f in fr:
        q = P[(P[:, 0] == f) & (P[:, 1] != 3) & on, 2:4]
        a = pl[f][~np.isnan(pl[f, :, 0])]
        if len(q) and len(a):
            d = np.linalg.norm(q[:, None] - a[None], axis=2)
            dd.extend(d.min(axis=1).tolist())
            unmatched_atd.append(np.sum(d.min(axis=0) > 2.0))
    dd = np.array(dd)
    if len(dd):
        print("detected person -> nearest ATD object: median {:.2f} m, p90 {:.2f} m; detections with no ATD "
              "object within 2 m {:.1f}%; ATD objects with no detection within 2 m {:.2f} per frame".format(
                  np.median(dd), np.percentile(dd, 90), 100 * np.mean(dd > 2), np.mean(unmatched_atd)))
    # ball
    atd_seen = ~np.isnan(ball[fr, 0])
    for src, label in ((0, "WASB"), (1, "YOLO ball class")):
        Bs = B[B[:, 4] == src]
        has = np.array([np.any(Bs[:, 0] == f) for f in fr])
        best = {}
        for f, x, y, sc, _ in Bs:
            if f not in best or sc > best[f][2]:
                best[int(f)] = (x, y, sc)
        d = [np.hypot(best[f][0] - ball[f, 0], best[f][1] - ball[f, 1]) for f in fr if f in best and atd_seen[f - first]]
        d = np.array(d)
        print("{:15s}: ball found on {:.0f}% of frames; where the ATD has the ball {:.0f}%, where it has none {:.0f}%; "
              "agreement with the ATD median {:.2f} m (within 2 m {:.0f}%)".format(
                  label, 100 * has.mean(), 100 * has[atd_seen].mean() if atd_seen.any() else 0,
                  100 * has[~atd_seen].mean() if (~atd_seen).any() else 0,
                  np.median(d) if len(d) else np.nan, 100 * np.mean(d < 2) if len(d) else 0))
        # in the ATD gaps: are the new balls a coherent track (consecutive frames < 1.6 m apart)?
        gap_f = [f for f in fr if not atd_seen[f - first] and f in best]
        coh = sum(1 for f in gap_f if f - 1 in best and
                  np.hypot(best[f][0] - best[f - 1][0], best[f][1] - best[f - 1][1]) < 1.6)
        near = []
        for f in gap_f:
            a = pl[f][~np.isnan(pl[f, :, 0])]
            if len(a):
                near.append(np.min(np.linalg.norm(a - np.array(best[f][:2]), axis=1)))
        print("{:15s}  in ATD gaps: {} frames with a ball, {:.0f}% continuing the previous frame's ball, nearest "
              "player median {:.1f} m".format("", len(gap_f), 100 * coh / max(len(gap_f), 1),
                                            np.median(near) if near else np.nan))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int)
    ap.add_argument("--frames", type=int, default=1500)
    ap.add_argument("--auto", action="store_true")
    a = ap.parse_args()
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("device:", device)
    os.makedirs(OUT_DIR, exist_ok=True)
    Hs = load_h()
    people = load_people(device)
    wasb = load_wasb(device)
    wins = auto_windows(a.frames) if a.auto else [("window", a.first)]
    for name, first in wins:
        P, B, st = run_window(first, a.frames, people, wasb, device, Hs)
        np.savez(os.path.join(OUT_DIR, "pilot_{}.npz".format(first)), det_people=P, det_ball=B, **{"t_" + k: v for k, v in st.items()})
        compare(name, first, a.frames, P, B, st)


if __name__ == "__main__":
    main()
