"""Review video of possession_v2 against the public ground truth: what we predict, what is true,
and the two scores that decide every change.

Layout (1920 x 1080):
  top left    tactical video
  top right   2D pitch from the tracking + live values (labels, scores, rule inputs)
  bottom      - strips over the whole window: ground truth, v2, mismatches, with a cursor
              - chart 1, possession change: challenger score of each team (threshold 1)
              - chart 2, dead ball: dead-ball score while in play, restart score while dead
                (threshold 1), ball-missing frames shaded, long gaps of rule 8 outlined
              charts cover +-15 s around the cursor; solid ticks = our changes (at the dated
              frame), dashed = ground-truth changes.

Also writes a still overview of the whole window (strips + both charts) next to the video.

Local viewing only: built from match data, it stays in data/clips/ (git-ignored).

    .venv/bin/python viz/v2_review.py                      # public five minutes
    .venv/bin/python viz/v2_review.py --overview-only      # just the still image
"""

import argparse
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
sys.path.insert(0, os.path.join(ROOT, "solution"))
import possession_v2 as v2  # noqa: E402
import tracking  # noqa: E402
from games import GAME, GAMES, half_video_blocks, video_path  # noqa: E402

FPS = 25
W, H = 1920, 1080
VID_W, VID_H = 1280, 720
CLIPS = os.path.join(ROOT, "data", "clips")
GT_CSV = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")

BG, PANEL, TEXT, DIM = (16, 18, 22), (26, 29, 35), (235, 235, 235), (140, 145, 155)
COL = {"H": (90, 160, 255), "A": (255, 110, 100), "D": (175, 175, 175), "X": (45, 45, 50)}
ALIVE_C, DEAD_C, THR_C, MISS_C, GAP8_C = (110, 220, 120), (240, 240, 240), (255, 215, 90), \
    (70, 45, 95), (170, 120, 230)
REASON_IT = {
    "control": "controllo", "touch": "tocco", "ball out": "palla fuori",
    "players stopped": "giocatori fermi", "whistle": "fischio", "corner": "angolo",
    "corner kick": "angolo battuto", "restart touch": "tocco di ripresa",
    "alive play": "gioco ripreso", "dead ball too short": "palla morta troppo breve",
    "ball lost, players walking": "buco lungo, giocatori fermi", "ball back": "palla ricompare", "lightning touch": "tocco lampo",
}

WIN_S = 15.0                       # charts: +- seconds around the cursor
CH_X, CH_W = 20, W - 40
PXF = CH_W / (2 * WIN_S * FPS)     # px per frame in the charts
STRIP_Y, STRIP_H = 730, 16
C1_Y, C2_Y, CH_H = 806, 946, 120


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for d in ("/usr/share/fonts/truetype/dejavu", "/Library/Fonts"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


F_S, F_M, F_B, F_L = font(14), font(17), font(17, True), font(26, True)


def load_gt():
    lab = np.loadtxt(GT_CSV, delimiter=",", skiprows=1, dtype=str)[:, 1]
    idx = np.flatnonzero(lab != "X")
    return lab, int(idx[0]), int(idx[-1])


def changes_of(lab, first, last):
    """[(frame, new label)] where the label changes inside [first, last]."""
    seg = lab[first:last + 1]
    cut = np.flatnonzero(seg[1:] != seg[:-1]) + 1
    return [(first + int(c), seg[c]) for c in cut]


# ---- charts: pre-rendered over the whole window, cropped per frame ----------------------------

def wide_chart(first, last, lines, shade, ticks, height=CH_H, ymax=1.25):
    """One image for the window, PXF px per frame, margins of WIN_S on each side."""
    pad = int(WIN_S * FPS)
    n = last - first + 1 + 2 * pad
    width = int(np.ceil(n * PXF)) + 2
    img = Image.new("RGB", (width, height), PANEL)
    d = ImageDraw.Draw(img)

    def x_of(f):
        return (f - first + pad) * PXF

    def y_of(v):
        return height - 4 - np.clip(v / ymax, 0, 1) * (height - 10)

    for mask, colour, (y0, y1) in shade:
        idx = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
        for a, b in zip(idx[::2], idx[1::2]):
            d.rectangle([x_of(first + a), y0, x_of(first + b), y1], fill=colour)
    yt = y_of(1.0)
    for x in range(0, width, 8):
        d.line([x, yt, x + 4, yt], fill=THR_C)
    for f, colour, dashed in ticks:
        x = x_of(f)
        if dashed:
            for y in range(0, height, 9):
                d.line([x, y, x, y + 4], fill=colour, width=2)
        else:
            d.line([x, 0, x, height], fill=colour, width=2)
    for values, colour in lines:
        v = np.asarray(values, dtype=np.float64)
        pts = [(x_of(first + i), y_of(v[i])) for i in range(len(v)) if np.isfinite(v[i])]
        # draw in runs so masked (NaN) stretches leave a gap
        run = []
        for i in range(len(v)):
            if np.isfinite(v[i]):
                run.append((x_of(first + i), y_of(v[i])))
            elif run:
                if len(run) > 1:
                    d.line(run, fill=colour, width=2)
                run = []
        if len(run) > 1:
            d.line(run, fill=colour, width=2)
        del pts
    return img


def strips(gt, pred, first, last, width):
    """GT / v2 / mismatch strips over the whole window, as one image."""
    n = last - first + 1
    img = Image.new("RGB", (width, 3 * STRIP_H + 8), BG)
    d = ImageDraw.Draw(img)
    cols = np.minimum((np.arange(n) * width / n).astype(int), width - 1)
    for row, lab in enumerate((gt[first:last + 1], pred[first:last + 1])):
        y = row * (STRIP_H + 4)
        for x in range(width):
            sel = lab[cols == x]
            if len(sel):
                vals, cnt = np.unique(sel, return_counts=True)
                d.line([x, y, x, y + STRIP_H - 1], fill=COL[vals[np.argmax(cnt)]])
    y = 2 * (STRIP_H + 4)
    wrong = gt[first:last + 1] != pred[first:last + 1]
    for x in range(width):
        if wrong[cols == x].mean() > 0.5:
            d.line([x, y, x, y + STRIP_H - 1], fill=(230, 60, 60))
    return img


# ---- pitch ---------------------------------------------------------------------------------

def pitch_base(pitch, size):
    L, Wd = pitch
    sx = size[0] / (L + 6)
    img = Image.new("RGB", size, (30, 70, 40))
    d = ImageDraw.Draw(img)

    def p(x, y):
        return (3 * sx + x * sx, 3 * sx + y * sx)
    line = (200, 220, 200)
    d.rectangle([p(0, 0), p(L, Wd)], outline=line, width=2)
    d.line([p(L / 2, 0), p(L / 2, Wd)], fill=line, width=2)
    r = 9.15 * sx
    cx, cy = p(L / 2, Wd / 2)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=line, width=2)
    for x0, s in ((0, 1), (L, -1)):
        d.rectangle(sorted_box(p(x0, Wd / 2 - 20.16), p(x0 + s * 16.5, Wd / 2 + 20.16)), outline=line, width=2)
        d.rectangle(sorted_box(p(x0, Wd / 2 - 9.16), p(x0 + s * 5.5, Wd / 2 + 9.16)), outline=line, width=2)
    return img, p


def sorted_box(a, b):
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])]


# ---- main --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="BAR-ATH_v2_review_public.mp4")
    ap.add_argument("--overview-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(CLIPS, exist_ok=True)

    gt, first, last = load_gt()
    teams, players, ball, ball_speed, names = tracking.load(GAME)
    blocks = half_video_blocks(GAME)
    pitch = GAMES[GAME]["pitch"]
    print("running possession_v2 (no whistle, the calibrated configuration) ...", flush=True)
    pred, f, r = v2.predict(teams, players, ball, ball_speed, blocks, pitch, v2.PARAMS)
    n_win = last - first + 1
    sl = slice(first, last + 1)

    # our changes, dated, inside the window; ground-truth changes
    ours = [(start, {0: "H", 1: "A", 2: "D"}[new], reason)
            for decided, start, new, reason in r["changes"] if first <= start <= last]
    true = changes_of(gt, first, last)
    poss_ticks = [(s, COL[k], False) for s, k, _ in ours if k in "HA"] + \
                 [(s, COL[k], True) for s, k in true if k in "HA" and gt[s - 1] in "HA"]
    dead_ticks = [(s, COL[k], False) for s, k, _ in ours if k == "D" or pred[s - 1] == "D"] + \
                 [(s, COL[k], True) for s, k in true if k == "D" or gt[s - 1] == "D"]

    state = r["state"][sl]
    score = r["score"][:, sl]
    dead_score = np.where(state != 2, r["dead_score"][sl], np.nan)
    alive_score = np.where(state == 2, r["alive_score"][sl], np.nan)
    missing = ~f["visible"][sl]
    gap8 = np.zeros(n_win, dtype=bool)
    for a in np.flatnonzero(f["gap_dead_start"][sl]):
        b = np.flatnonzero(f["gap_dead_end"][first + a:last + 1])
        gap8[a:(a + b[0]) if len(b) else n_win] = True

    c1 = wide_chart(first, last, [(score[0], COL["H"]), (score[1], COL["A"])], [], poss_ticks)
    c2 = wide_chart(first, last, [(dead_score, DEAD_C), (alive_score, ALIVE_C)],
                    [(missing, MISS_C, (CH_H - 14, CH_H)), (gap8, GAP8_C, (CH_H - 14, CH_H - 10))],
                    dead_ticks)
    st = strips(gt, pred, first, last, CH_W)

    # ---- still overview of the whole window ------------------------------------------------
    rows = 5
    seg = n_win // rows
    ov_w = CH_W
    ov = Image.new("RGB", (ov_w + 40, 140 + rows * (2 * CH_H + 50)), BG)
    d = ImageDraw.Draw(ov)
    d.text((20, 10), "possession_v2 vs ground truth, 5 minuti pubblici (frame {}-{})".format(first, last),
           font=F_L, fill=TEXT)
    ov.paste(st, (20, 50))
    for lbl, y in (("GT", 50), ("v2", 70), ("errori", 90)):
        d.text((ov_w - 40, y), lbl, font=F_S, fill=DIM)
    pad = int(WIN_S * FPS)
    for k in range(rows):
        a = k * seg
        x0 = int((a + pad) * PXF)
        x1 = int((a + seg + pad) * PXF)
        y = 130 + k * (2 * CH_H + 50)
        d.text((20, y), "{:d}:{:02d} - {:d}:{:02d}".format(a // FPS // 60, a // FPS % 60,
                                                          (a + seg) // FPS // 60, (a + seg) // FPS % 60),
               font=F_B, fill=TEXT)
        for j, c in enumerate((c1, c2)):
            crop = c.crop((x0, 0, x1, CH_H)).resize((ov_w, CH_H))
            ov.paste(crop, (20, y + 22 + j * (CH_H + 4)))
    ov_path = os.path.join(CLIPS, os.path.splitext(os.path.basename(args.out))[0] + "_overview.png")
    ov.save(ov_path)
    print("->", os.path.relpath(ov_path, ROOT))
    if args.overview_only:
        return

    # ---- video -----------------------------------------------------------------------------
    pitch_size = (620, 410)
    base, p = pitch_base(pitch, pitch_size)
    out_path = os.path.join(CLIPS, os.path.basename(args.out))
    t0 = first / FPS
    reader = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-ss", "{:.3f}".format(t0), "-i", video_path(GAME),
         "-frames:v", str(n_win), "-vf", "scale={}:{}".format(VID_W, VID_H),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    writer = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", "{}x{}".format(W, H), "-r", str(FPS), "-i", "-",
         "-ss", "{:.3f}".format(t0), "-t", "{:.3f}".format(n_win / FPS), "-i", video_path(GAME),
         "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", out_path], stdin=subprocess.PIPE)

    pad = int(WIN_S * FPS)
    half = int(WIN_S * FPS * PXF)
    last_reason = ["inizio", first]
    ours_by_frame = {}
    for s, k, reason in ours:
        ours_by_frame.setdefault(s, (k, reason))
    decided = {dec: (start, new, reason) for dec, start, new, reason in r["changes"]}
    tick = 0
    for i in range(n_win):
        t = first + i
        raw = reader.stdout.read(VID_W * VID_H * 3)
        if len(raw) < VID_W * VID_H * 3:
            break
        frame = Image.new("RGB", (W, H), BG)
        frame.paste(Image.frombytes("RGB", (VID_W, VID_H), raw), (0, 0))
        d = ImageDraw.Draw(frame)

        # pitch
        pim = base.copy()
        pd_ = ImageDraw.Draw(pim)
        for s in range(players.shape[1]):
            x, y = players[t, s]
            if np.isfinite(x):
                cx, cy = p(x, y)
                pd_.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=COL["H" if teams[s] == 0 else "A"],
                            outline=(0, 0, 0))
        if f["visible"][t]:
            cx, cy = p(*ball[t])
            pd_.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(255, 255, 255), outline=(0, 0, 0))
            for k in (0, 1):
                sl_ = f["slot"][k, t]
                if sl_ >= 0 and np.isfinite(f["dist"][k, t]) and f["dist"][k, t] < 6:
                    px, py = p(*players[t, sl_])
                    pd_.line([cx, cy, px, py], fill=COL["HA"[k]], width=2)
        else:
            pd_.text((10, 10), "palla non tracciata", font=F_M, fill=(255, 200, 120))
        frame.paste(pim, (VID_W + 10, 10))

        # info panel
        if t in decided:
            s0, new, reason = decided[t]
            last_reason = [REASON_IT.get(reason, reason), t]
        x0, y0 = VID_W + 14, 430
        g, q = gt[t], pred[t]
        d.text((x0, y0), "verità", font=F_M, fill=DIM)
        d.rectangle([x0 + 90, y0 - 2, x0 + 150, y0 + 24], fill=COL[g])
        d.text((x0 + 112, y0 + 1), g, font=F_B, fill=(0, 0, 0))
        d.text((x0 + 180, y0), "v2", font=F_M, fill=DIM)
        d.rectangle([x0 + 220, y0 - 2, x0 + 280, y0 + 24], fill=COL[q])
        d.text((x0 + 242, y0 + 1), q, font=F_B, fill=(0, 0, 0))
        if g != q:
            d.text((x0 + 300, y0 + 1), "ERRORE", font=F_B, fill=(240, 80, 80))
        secs = i / FPS
        d.text((x0, y0 + 36), "t = {:d}:{:04.1f}   frame {}".format(int(secs // 60), secs % 60, t),
               font=F_M, fill=TEXT)
        d.text((x0, y0 + 62), "ultimo cambio: {} ({:.1f} s fa)".format(
            last_reason[0], (t - last_reason[1]) / FPS), font=F_M, fill=TEXT)
        own = r["state"][t]
        vals = [
            ("punti cambio H / A", "{:.2f} / {:.2f}".format(r["score"][0, t], r["score"][1, t])),
            ("punti palla morta", "{:.2f}".format(r["dead_score"][t]) if own != 2 else "--"),
            ("punti ripresa", "{:.2f}".format(r["alive_score"][t]) if own == 2 else "--"),
            ("controllo H / A", "{:.2f} / {:.2f}".format(f["ctrl"][0, t], f["ctrl"][1, t])),
            ("velocità palla", "--" if not np.isfinite(f["speed"][t]) else "{:.1f} m/s".format(f["speed"][t])),
            ("vel. media giocatori", "--" if not np.isfinite(f["pspeed"][t]) else "{:.2f} m/s".format(f["pspeed"][t])),
            ("distanza 4 più vicini", "--" if not np.isfinite(f["far"][t]) else "{:.1f} m".format(f["far"][t])),
            ("frenata (0-1)", "--" if not np.isfinite(f["f_rhythm"][t]) else "{:.2f}".format(f["f_rhythm"][t])),
        ]
        for j, (k_, v_) in enumerate(vals):
            d.text((x0, y0 + 100 + j * 23), k_, font=F_S, fill=DIM)
            d.text((x0 + 230, y0 + 100 + j * 23), v_, font=F_S, fill=TEXT)

        # strips with cursor
        frame.paste(st, (CH_X, STRIP_Y))
        cx = CH_X + int(i * CH_W / n_win)
        d.line([cx, STRIP_Y - 3, cx, STRIP_Y + 3 * STRIP_H + 11], fill=(255, 255, 255), width=2)
        for lbl, yy in (("verità", 0), ("v2", 1), ("errori", 2)):
            d.text((CH_X + CH_W - 52, STRIP_Y + yy * (STRIP_H + 4)), lbl, font=F_S, fill=(0, 0, 0))

        # charts around the cursor
        xc = int((i + pad) * PXF)
        for c, y, title in ((c1, C1_Y, "1  CAMBIO DI POSSESSO: punti della squadra sfidante (blu H, rosso A), soglia 1"),
                            (c2, C2_Y, "2  PALLA MORTA: punti palla morta (bianco, a gioco vivo) / ripresa (verde, a palla morta)   viola = palla non tracciata")):
            frame.paste(c.crop((xc - half, 0, xc - half + CH_W, CH_H)), (CH_X, y))
            d.text((CH_X, y - 20), title, font=F_S, fill=DIM)
            d.line([CH_X + CH_W // 2, y, CH_X + CH_W // 2, y + CH_H], fill=(255, 255, 255), width=1)
        d.text((CH_X + CH_W - 470, C1_Y - 20), "linea continua = nostro cambio, tratteggiata = vero",
               font=F_S, fill=DIM)

        writer.stdin.write(frame.tobytes())
        tick += 1
        if tick % 750 == 0:
            print("  {:.0f}%".format(100 * tick / n_win), flush=True)
    writer.stdin.close()
    writer.wait()
    reader.terminate()
    print("->", os.path.relpath(out_path, ROOT))


if __name__ == "__main__":
    main()
