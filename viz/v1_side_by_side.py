"""Review video of possession v1, with the decision rules live on screen.

Top: tactical video (left) and 2D pitch (right). On the pitch, around the ball: the control radii
(R_NEAR, R_FAR), the ball velocity arrow, and a line to the closest player of each team with its
distance. Bottom:
  - the v1 label (as written in the submission, change back-dated) and the live state of the machine;
  - three 5-minute strips: ground truth (public window only), v1, baseline smooth50;
  - the challenger's score over the page, with the threshold and the changes;
  - a +/-5 s zoom of the score and of the ball speed, and the live value of every rule.

Local viewing only: a derivative of the match data, it stays in data/clips/ (ignored by git).

    .venv/bin/python solution/possession_v1.py            # writes the diagnostics first
    .venv/bin/python viz/v1_side_by_side.py               # whole match
    .venv/bin/python viz/v1_side_by_side.py --first 11185 --last 18684 --out v1_gt_public.mp4
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
sys.path.insert(0, HERE)
import gt_side_by_side as gt  # noqa: E402  (pitch drawing, colours, fonts)
import match_side_by_side as ms  # noqa: E402  (page grid, label reader)
import possession_v1 as v1  # noqa: E402
import tracking  # noqa: E402

N_VIDEO = ms.N_VIDEO
PANEL_W, PANEL_H, OUT_W = gt.PANEL_W, gt.PANEL_H, gt.OUT_W
PAGE, ORIGIN, TL_X, TL_W, PX = ms.PAGE, ms.ORIGIN, ms.TL_X, ms.TL_W, ms.PX
V1_CSV = v1.OUT_CSV
TEAM_FILL = (gt.HOME_FILL, gt.AWAY_FILL)
TEAM_LINE = ((90, 160, 255), (255, 110, 100))  # lighter versions, readable on dark background
CHART_H = 70
ZOOM_S = 5.0
ZOOM_X, ZOOM_W, ZOOM_H = 30, 900, 190
INFO_X = 960
BAR_H = 550
OUT_H = PANEL_H + BAR_H
DIM, TEXT, ACCENT = (150, 155, 165), (240, 240, 240), (255, 220, 120)


def page_chart(score, live, switches, thr):
    """Whole-match image of the challenger score, x = (frame - ORIGIN) * PX."""
    n_pages = (N_VIDEO - ORIGIN + PAGE - 1) // PAGE
    width = int(n_pages * PAGE * PX) + 2
    img = np.full((CHART_H, width, 3), 22, dtype=np.uint8)
    top = thr * 1.25
    cols = ((np.arange(N_VIDEO) - ORIGIN) * PX).astype(np.int64)
    for team in (0, 1):
        col_max = np.zeros(width)
        np.maximum.at(col_max, cols, score[team])
        h = np.clip(col_max / top, 0, 1) * (CHART_H - 2)
        for x in np.flatnonzero(h >= 1):
            y0 = CHART_H - 1 - int(h[x])
            img[y0:CHART_H - 1, x] = np.maximum(img[y0:CHART_H - 1, x], TEAM_LINE[team])
    y_thr = CHART_H - 1 - int(thr / top * (CHART_H - 2))
    img[y_thr, ::3] = ACCENT
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    for decided, start, team in switches:
        x = int((start - ORIGIN) * PX)
        d.line([x, 0, x, CHART_H - 1], fill=(255, 255, 255), width=1)
        d.rectangle([x - 2, 0, x + 2, 4], fill=TEAM_LINE[team])
    return pil


def fmt(x, nd=1, unit=""):
    return "--" if x is None or not np.isfinite(x) else "{:.{}f}{}".format(x, nd, unit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--last", type=int, default=N_VIDEO - 1)
    ap.add_argument("--out", default="BAR-ATH_v1_side_by_side.mp4")
    args = ap.parse_args()
    first, last = max(0, args.first), min(N_VIDEO - 1, args.last)
    n = last - first + 1
    out_path = os.path.join(ms.CLIPS, os.path.basename(args.out))

    teams, players, ball, _, (home, away) = tracking.load()
    names = (home, away)
    z = np.load(v1.OUT_DIAG)
    p = v1.PARAMS
    thr = float(z["threshold"])
    score, live, dist, slot = z["score"], z["live"], z["dist"], z["slot"]
    speed, dv, angle, vel = z["speed"], z["dv"], z["angle"], z["vel"]
    ctrl, slow, conf = z["ctrl"], z["slow"], z["conf"]
    touch_team, is_touch = z["touch_team"], z["is_touch"]
    pts = {k: z["pts_" + k] for k in ("ctrl", "touch", "look", "own")}
    switches = [tuple(int(v) for v in row) for row in z["switches"]]
    reasons = list(z["reasons"])
    sw_start = np.array([s[1] for s in switches])
    sw_decided = np.array([s[0] for s in switches])

    gt_labels = ms.read_csv_labels(ms.GT_CSV)
    base = ms.read_csv_labels(ms.BASELINE)
    lab_v1 = ms.read_csv_labels(V1_CSV)
    is_gt = np.zeros(N_VIDEO, dtype=bool)
    is_gt[ms.GT_FIRST:ms.GT_LAST + 1] = True

    gt_img, base_img, _ = ms.long_strips(gt_labels, is_gt, base, [], 1, None)
    _, v1_img, _ = ms.long_strips(gt_labels, is_gt, lab_v1, [], 1, None)
    chart_img = page_chart(score, live, switches, thr)
    touch_frames = np.flatnonzero(is_touch)

    background = gt.pitch_background()
    font_big = ImageFont.truetype(gt.FONT, 70)
    font_mid = ImageFont.truetype(gt.FONT, 34)
    font_small = ImageFont.truetype(gt.FONT_REG, 22)
    font_bold = ImageFont.truetype(gt.FONT, 22)
    font_tag = ImageFont.truetype(gt.FONT_REG, 15)
    state_of = {"H": "{} in possesso".format(home), "A": "{} in possesso".format(away),
                "D": "Palla morta", "X": "Nessuna etichetta"}
    zoom_frames = int(ZOOM_S * 25)

    seek = ["-ss", "{:.2f}".format(first / 25)] if first > 0 else []
    reader = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-hwaccel", "videotoolbox"] + seek + ["-i", ms.VIDEO,
         "-frames:v", str(n), "-vf", "scale={}:{}".format(PANEL_W, PANEL_H),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE)
    writer = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", "{}x{}".format(OUT_W, OUT_H), "-r", "25", "-i", "-",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p", out_path],
        stdin=subprocess.PIPE)

    frame_bytes = PANEL_W * PANEL_H * 3
    written = 0
    y0 = PANEL_H
    for i in range(n):
        raw = reader.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        f = first + i
        canvas = Image.new("RGB", (OUT_W, OUT_H), (16, 18, 22))
        canvas.paste(Image.frombuffer("RGB", (PANEL_W, PANEL_H), raw), (0, 0))

        # ---- pitch -----------------------------------------------------------------------
        pitch = background.copy()
        d = ImageDraw.Draw(pitch)
        has_ball = not np.isnan(ball[f, 0])
        if has_ball:
            bx, by = gt.to_px(*ball[f])
            for r, col in ((p["R_FAR"], (200, 200, 200)), (p["R_NEAR"], (255, 255, 255))):
                rp = r * gt.SCALE
                d.ellipse([bx - rp, by - rp, bx + rp, by + rp], outline=col, width=1)
            for team in (0, 1):
                s_ = slot[team, f]
                if s_ >= 0 and dist[team, f] < 15:
                    qx, qy = gt.to_px(*players[f, s_])
                    d.line([bx, by, qx, qy], fill=TEAM_LINE[team], width=2)
                    d.text(((bx + qx) / 2 + 4, (by + qy) / 2 - 18), "{:.1f} m".format(dist[team, f]),
                           font=font_tag, fill=TEAM_LINE[team])
        for s_ in range(players.shape[1]):
            if np.isnan(players[f, s_, 0]):
                continue
            px, py = gt.to_px(*players[f, s_])
            fill, edge = (gt.HOME_FILL, gt.HOME_EDGE) if teams[s_] == 0 else (gt.AWAY_FILL, gt.AWAY_EDGE)
            d.ellipse([px - 8, py - 8, px + 8, py + 8], fill=fill, outline=edge, width=3)
        if has_ball:
            if np.isfinite(vel[f, 0]):
                ex, ey = gt.to_px(ball[f, 0] + vel[f, 0] * 0.5, ball[f, 1] + vel[f, 1] * 0.5)
                d.line([bx, by, ex, ey], fill=ACCENT, width=3)
            d.ellipse([bx - 6, by - 6, bx + 6, by + 6], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
        else:
            d.text((gt.OX + 8, gt.OY + 6), "palla non tracciata", font=font_small, fill=ACCENT)
        canvas.paste(pitch, (PANEL_W, 0))

        c = ImageDraw.Draw(canvas)
        # ---- label and state -------------------------------------------------------------
        lab = lab_v1[f]
        c.rectangle([30, y0 + 14, 290, y0 + 100], fill=gt.LABEL_COLOR.get(lab, (40, 40, 40)))
        c.text((160, y0 + 57), lab, font=font_big, fill=(255, 255, 255), anchor="mm")
        c.text((160, y0 + 108), "V1 (retrodatata)", font=font_small, fill=TEXT, anchor="mt")
        line = "v1: " + state_of.get(lab, lab)
        if is_gt[f]:
            line += "   ·   GT: {}".format(gt_labels[f])
        line += "   ·   baseline: {}".format(base[f])
        c.text((TL_X, y0 + 18), line, font=font_mid, fill=TEXT)
        secs = f / 25
        c.text((TL_X, y0 + 64), "fotogramma video {}   ·   {:d}:{:05.2f} del video".format(
            f, int(secs // 60), secs % 60), font=font_small, fill=(170, 175, 185))

        # ---- strips and page chart -------------------------------------------------------
        page_start = ORIGIN + ((f - ORIGIN) // PAGE) * PAGE
        x0 = int((page_start - ORIGIN) * PX)
        for k, (name, img) in enumerate((("Ground truth", gt_img), ("v1 possesso", v1_img),
                                         ("Baseline smooth50", base_img))):
            c.text((30, y0 + 136 + 26 * k), name, font=font_small, fill=(170, 175, 185))
            canvas.paste(img.crop((x0, 0, x0 + TL_W, 16)), (TL_X, y0 + 140 + 26 * k))
        c.text((30, y0 + 222), "Punteggio cambio", font=font_small, fill=(170, 175, 185))
        c.text((30, y0 + 248), "soglia {:.1f} (gialla)".format(thr), font=font_tag, fill=DIM)
        canvas.paste(chart_img.crop((x0, 0, x0 + TL_W, CHART_H)), (TL_X, y0 + 222))
        ps, pe = max(0, page_start) / 25, min(N_VIDEO, page_start + PAGE) / 25
        c.text((TL_X + TL_W, y0 + 108), "pagina {:d}:{:02d} - {:d}:{:02d}".format(
            int(ps // 60), int(ps % 60), int(pe // 60), int(pe % 60)), font=font_tag,
            fill=(130, 135, 145), anchor="ra")
        cur = TL_X + int((f - page_start) * PX)
        c.rectangle([cur - 1, y0 + 134, cur + 1, y0 + 222 + CHART_H], fill=(255, 255, 255))

        # ---- zoom chart (+/- 5 s) --------------------------------------------------------
        zy = y0 + 310
        c.rectangle([ZOOM_X, zy, ZOOM_X + ZOOM_W, zy + ZOOM_H], fill=(24, 26, 32))
        lo, hi = f - zoom_frames, f + zoom_frames
        fr = np.arange(max(0, lo), min(N_VIDEO - 1, hi) + 1)
        xs = ZOOM_X + (fr - lo) / (2 * zoom_frames) * ZOOM_W
        top = thr * 1.25

        def ys(vals, vmax):
            return zy + ZOOM_H - 4 - np.clip(vals / vmax, 0, 1) * (ZOOM_H - 24)

        y_thr = ys(np.array([thr]), top)[0]
        for xx in range(ZOOM_X, ZOOM_X + ZOOM_W, 8):
            c.line([xx, y_thr, xx + 4, y_thr], fill=ACCENT)
        sp = np.nan_to_num(speed[fr], nan=-1)
        pts_sp = [(x_, y_) for x_, y_, v_ in zip(xs, ys(sp, 25.0), sp) if v_ >= 0]
        if len(pts_sp) > 1:
            c.point(pts_sp, fill=(120, 125, 135))
        for team in (0, 1):
            yv = ys(score[team, fr], top)
            c.line(list(zip(xs, yv)), fill=TEAM_LINE[team], width=2)
        for tf in touch_frames[np.searchsorted(touch_frames, lo):np.searchsorted(touch_frames, hi + 1)]:
            tx = ZOOM_X + (tf - lo) / (2 * zoom_frames) * ZOOM_W
            team = touch_team[tf]
            col = TEAM_LINE[team] if team >= 0 else (160, 160, 160)
            c.polygon([(tx, zy + 4), (tx - 5, zy + 14), (tx + 5, zy + 14)], fill=col)
        for s0, s1 in zip(sw_start, sw_decided):
            if lo <= s0 <= hi:
                sx = ZOOM_X + (s0 - lo) / (2 * zoom_frames) * ZOOM_W
                c.line([sx, zy, sx, zy + ZOOM_H], fill=(255, 255, 255), width=2)
            if lo <= s1 <= hi:
                sx = ZOOM_X + (s1 - lo) / (2 * zoom_frames) * ZOOM_W
                c.line([sx, zy + ZOOM_H - 30, sx, zy + ZOOM_H], fill=ACCENT, width=2)
        c.line([ZOOM_X + ZOOM_W / 2, zy, ZOOM_X + ZOOM_W / 2, zy + ZOOM_H], fill=(255, 255, 255), width=1)
        c.text((ZOOM_X + 6, zy + ZOOM_H + 2), "-5 s   punteggio (colori squadra) · velocità palla 0-25 m/s "
               "(grigio) · ▲ tocchi · linea bianca = cambio datato, gialla = decisione   +5 s",
               font=font_tag, fill=DIM)

        # ---- live rules ------------------------------------------------------------------
        owner = int(live[f])
        lines = []
        if owner < 0:
            lines.append(("Macchina: nessun possesso ancora (inizio tempo)", TEXT, True))
            chal = int(np.argmax(score[:, f]))
        else:
            chal = 1 - owner
            lines.append(("Macchina: {} in possesso · sfidante {}".format(names[owner], names[chal]),
                          TEXT, True))
        sc = float(score[chal, f])
        lines.append(("Punteggio {}: {:.2f} / {:.2f}".format(names[chal], sc, thr),
                      ACCENT if sc >= 0.5 * thr else TEXT, True))
        if has_ball:
            dc, do = dist[chal, f], dist[1 - chal, f]
            lines.append(("1 Controllo: {} {} · {} {} · palla {} m/s → +{:.3f}".format(
                names[chal][:9], fmt(dc, 1, " m"), names[1 - chal][:9], fmt(do, 1, " m"),
                fmt(speed[f], 1), pts["ctrl"][chal, f]), TEXT, False))
            fast = np.isfinite(speed[f]) and slow[f] == 0
            if fast:
                lines.append(("3 Palla veloce ({} m/s), Δv {} m/s: prossimità ignorata".format(
                    fmt(speed[f], 1), fmt(dv[f], 1)), ACCENT, False))
            else:
                lines.append(("3 Palla lenta: peso velocità {:.2f}".format(slow[f]), DIM, False))
        else:
            lines.append(("1 Controllo: palla non tracciata, nessun punto", DIM, False))
            lines.append(("3 Palla veloce: --", DIM, False))
        k = np.searchsorted(touch_frames, f + 1) - 1
        if k >= 0 and f - touch_frames[k] <= 25:
            tf = touch_frames[k]
            who = names[touch_team[tf]] if touch_team[tf] >= 0 else "nessuno vicino"
            lines.append(("2 Tocco {:.1f} s fa: Δv {} m/s, angolo {}° · {}".format(
                (f - tf) / 25, fmt(dv[tf], 1), fmt(angle[tf], 0), who),
                TEAM_LINE[touch_team[tf]] if touch_team[tf] >= 0 else DIM, False))
            if touch_team[tf] >= 0 and touch_team[tf] != owner:
                lines.append(("4 Conferma nei {:.0f} s dopo: {:+.2f} → {:+.2f} punti".format(
                    p["LOOK_S"], conf[touch_team[tf], tf], p["W_LOOK"] * conf[touch_team[tf], tf]),
                    TEXT, False))
            else:
                lines.append(("4 Conferma: tocco della squadra in possesso, -{:.1f}".format(
                    p["W_TOUCH_OWN"]) if touch_team[tf] == owner and owner >= 0 else
                    "4 Conferma: --", DIM, False))
        else:
            lines.append(("2 Tocco: nessuno nell'ultimo secondo", DIM, False))
            lines.append(("4 Conferma: --", DIM, False))
        own = float(pts["own"][chal, f])
        lines.append(("Possesso proprio: -{:.3f} (controllo e tocchi di chi ha palla)".format(own),
                      TEXT if own > 0 else DIM, False))
        j = np.searchsorted(sw_decided, f + 1) - 1
        if j >= 0 and f - sw_decided[j] <= 50:
            dec, st, team = switches[j]
            lines.append(("CAMBIO → {} ({}), retrodatato di {:.2f} s".format(
                names[team], "tocco" if reasons[j] == "touch" else "controllo", (dec - st) / 25),
                TEAM_LINE[team], True))
        for li, (txt, col, bold) in enumerate(lines):
            c.text((INFO_X, zy + li * 25), txt, font=font_bold if bold else font_small, fill=col)

        writer.stdin.write(canvas.tobytes())
        written += 1
        if written % 10000 == 0:
            print("  {} / {}".format(written, n), flush=True)
    writer.stdin.close()
    writer.wait()
    reader.stdout.close()
    reader.wait()
    print("wrote {} frames -> {}".format(written, out_path))


if __name__ == "__main__":
    main()
