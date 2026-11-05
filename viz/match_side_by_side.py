"""Side-by-side review video of the whole match (or a frame range of it).

Left: the tactical video. Right: the 2D pitch from the tracking. Bottom: the label of the frame --
the public ground truth inside its 5-minute window, the reference baseline everywhere else -- and
a timeline of the labels (ground truth and baseline on two synchronised rows) and of the Smart
Tagging phases. The timeline shows one 5-minute page at a time with a moving cursor; pages are
aligned so that the public ground-truth window is exactly one page.

Local viewing only: the output is a derivative of the match data, so it stays in data/ (ignored
by git) and must not be published or uploaded anywhere.

    .venv/bin/python viz/match_side_by_side.py                       # whole match
    .venv/bin/python viz/match_side_by_side.py --first 11000 --last 11600 --out test.mp4
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
sys.path.insert(0, HERE)
from games import GAME, GAMES, require, source_frames  # noqa: E402
from baseline import team_of_each_track, team_names  # noqa: E402
from load_smart_tagging import load as load_tags  # noqa: E402
import gt_side_by_side as gt  # noqa: E402  (pitch drawing, colours, fonts)

N_VIDEO = source_frames("tactical")
GT_FIRST, GT_LAST = gt.FIRST, gt.LAST
VIDEO = os.path.join(ROOT, "data", "tactical", "BAR-ATH_tactical.mp4")
GT_CSV = gt.GT
BASELINE = os.path.join(ROOT, "evaluation", "results", "baseline_smooth50.csv")
CLIPS = os.path.join(ROOT, "data", "clips")

PANEL_W, PANEL_H, OUT_W = gt.PANEL_W, gt.PANEL_H, gt.OUT_W
LENGTH, WIDTH = gt.LENGTH, gt.WIDTH
PAGE = GT_LAST - GT_FIRST + 1   # 7500 frames = 5 minutes per timeline page
ORIGIN = GT_FIRST - ((GT_FIRST + PAGE - 1) // PAGE) * PAGE  # page grid aligned on the GT window
TL_X = 330
TL_W = OUT_W - TL_X - 40
PX = TL_W / PAGE                # timeline pixels per frame
LANE_H = 24


def read_csv_labels(path):
    out = np.full(N_VIDEO, "X", dtype="<U1")
    with open(path) as handle:
        next(handle)
        for line in handle:
            frame, label = line.strip().split(",")
            out[int(frame)] = label
    return out


def read_tracking(first, last):
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    teams = team_of_each_track(meta_path)
    n = last - first + 1
    players = np.full((n, len(teams), 2), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1  # tracking frame_id counts from 1, video frames from 0
            if frame < first:
                continue
            if frame > last:
                break
            i = frame - first
            for slot, group in enumerate(parts[1].split(";")):
                vals = group.split(",")
                if vals[0][0] != "N":
                    players[i, slot] = (float(vals[0]) * LENGTH, float(vals[1]) * WIDTH)
            vals = parts[2].split(",")
            if vals[0][0] != "N":
                ball[i] = (float(vals[0]) * LENGTH, float(vals[1]) * WIDTH)
    return teams, players, ball


def read_all_tags(home, away):
    """All Smart Tagging phases in video frames (end exclusive), with a lane each."""
    tags = []
    for ev in load_tags():
        team = (ev["tags"].get("Team") or "").strip()
        code = (ev["code"] or "").strip()
        kind = (ev["tags"].get("Type") or "").strip()
        if code.lower() == "set pieces":
            color = gt.SET_PIECE
        elif team == home:
            color = gt.HOME_FILL
        elif team == away:
            color = gt.AWAY_FILL
        else:
            color = gt.NEUTRAL
        tags.append({"s": ev["start_frame"], "e": ev["end_frame"], "code": code,
                     "team": team or "N/A", "type": kind, "color": color})
    tags.sort(key=lambda t: (t["s"], t["e"]))
    lane_end = []
    for t in tags:
        for k, end in enumerate(lane_end):
            if t["s"] >= end:
                t["lane"], lane_end[k] = k, t["e"]
                break
        else:
            t["lane"] = len(lane_end)
            lane_end.append(t["e"])
    return tags, max(1, len(lane_end))


def long_strips(gt_labels, is_gt, base, tags, n_lanes, font_tag):
    """Whole-match timeline images at PX px/frame, x = (frame - ORIGIN) * PX."""
    n_pages = (N_VIDEO - ORIGIN + PAGE - 1) // PAGE
    width = int(n_pages * PAGE * PX) + 2
    frames = (np.arange(width) / PX).astype(np.int64) + ORIGIN
    valid = (frames >= 0) & (frames < N_VIDEO)
    fr = np.clip(frames, 0, N_VIDEO - 1)

    def row(labels, mask):
        rgb = np.full((width, 3), 30, dtype=np.uint8)
        for lab, col in gt.LABEL_COLOR.items():
            rgb[(labels[fr] == lab) & mask] = col
        return Image.fromarray(np.repeat(rgb[None], 16, axis=0))

    gt_img = row(gt_labels, valid & is_gt[fr])
    base_img = row(base, valid)

    lanes = Image.new("RGB", (width, n_lanes * LANE_H), (16, 18, 22))
    d = ImageDraw.Draw(lanes)
    for t in tags:
        x0 = int((t["s"] - ORIGIN) * PX)
        x1 = max(x0 + 2, int((t["e"] - ORIGIN) * PX) - 1)
        y0 = t["lane"] * LANE_H
        d.rectangle([x0, y0, x1, y0 + LANE_H - 4], fill=t["color"])
        name = t["code"].title() if not t["type"] else t["type"]
        if d.textlength(name, font=font_tag) < x1 - x0 - 8:
            d.text((x0 + 4, y0 + 2), name, font=font_tag,
                   fill=(20, 20, 20) if t["color"] == gt.SET_PIECE else (245, 245, 245))
    return gt_img, base_img, lanes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--last", type=int, default=N_VIDEO - 1)
    ap.add_argument("--out", default="BAR-ATH_match_side_by_side.mp4")
    args = ap.parse_args()
    first, last = max(0, args.first), min(N_VIDEO - 1, args.last)
    n = last - first + 1
    out_path = os.path.join(CLIPS, os.path.basename(args.out))

    home, away = team_names(require(["atd_meta"])[0])
    gt_labels = read_csv_labels(GT_CSV)
    base = read_csv_labels(BASELINE)
    is_gt = np.zeros(N_VIDEO, dtype=bool)
    is_gt[GT_FIRST:GT_LAST + 1] = True
    labels = np.where(is_gt, gt_labels, base)
    tags, n_lanes = read_all_tags(home, away)
    teams, players, ball = read_tracking(first, last)
    print("frames {}..{} ({}), ground truth frames in range {}, smart tagging {} phases on {} lanes".format(
        first, last, n, int(is_gt[first:last + 1].sum()), len(tags), n_lanes))

    tag_y = PANEL_H + 236
    bar_h = 236 + n_lanes * LANE_H + 30
    bar_h += bar_h % 2
    out_h = PANEL_H + bar_h

    background = gt.pitch_background()
    font_big = ImageFont.truetype(gt.FONT, 70)
    font_mid = ImageFont.truetype(gt.FONT, 34)
    font_small = ImageFont.truetype(gt.FONT_REG, 22)
    font_tag = ImageFont.truetype(gt.FONT_REG, 15)
    text_of = {"H": "{} in possesso".format(home), "A": "{} in possesso".format(away),
               "D": "Palla morta", "X": "Nessuna etichetta"}
    gt_img, base_img, lanes_img = long_strips(gt_labels, is_gt, base, tags, n_lanes, font_tag)

    # Hardware decode; seek straight to the first frame (constant 25 fps, accurate seek decodes
    # from the previous keyframe and drops frames up to the exact timestamp).
    seek = ["-ss", "{:.2f}".format(first / 25)] if first > 0 else []
    reader = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-hwaccel", "videotoolbox"] + seek + ["-i", VIDEO,
         "-frames:v", str(n), "-vf", "scale={}:{}".format(PANEL_W, PANEL_H),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE)
    writer = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", "{}x{}".format(OUT_W, out_h), "-r", "25", "-i", "-",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p", out_path],
        stdin=subprocess.PIPE)

    frame_bytes = PANEL_W * PANEL_H * 3
    written = 0
    tag_starts = np.array([t["s"] for t in tags])
    for i in range(n):
        raw = reader.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        f = first + i
        canvas = Image.new("RGB", (OUT_W, out_h), (16, 18, 22))
        canvas.paste(Image.frombuffer("RGB", (PANEL_W, PANEL_H), raw), (0, 0))

        pitch = background.copy()
        d = ImageDraw.Draw(pitch)
        for slot in range(players.shape[1]):
            if np.isnan(players[i, slot, 0]):
                continue
            px, py = gt.to_px(*players[i, slot])
            fill, edge = (gt.HOME_FILL, gt.HOME_EDGE) if teams[slot] == 0 else (gt.AWAY_FILL, gt.AWAY_EDGE)
            d.ellipse([px - 8, py - 8, px + 8, py + 8], fill=fill, outline=edge, width=3)
        if np.isnan(ball[i, 0]):
            d.text((gt.OX + 8, gt.OY + 6), "palla non tracciata", font=font_small, fill=(255, 220, 120))
        else:
            bx, by = gt.to_px(*ball[i])
            d.ellipse([bx - 6, by - 6, bx + 6, by + 6], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
        canvas.paste(pitch, (PANEL_W, 0))

        c = ImageDraw.Draw(canvas)
        lab = labels[f]
        c.rectangle([30, PANEL_H + 14, 290, PANEL_H + 100], fill=gt.LABEL_COLOR.get(lab, (40, 40, 40)))
        c.text((160, PANEL_H + 57), lab, font=font_big, fill=(255, 255, 255), anchor="mm")
        source = "GROUND TRUTH" if is_gt[f] else "BASELINE smooth50"
        c.text((160, PANEL_H + 108), source, font=font_small, fill=(255, 255, 255) if is_gt[f] else (150, 155, 165), anchor="mt")
        state = text_of.get(lab, lab)
        if is_gt[f]:
            state += "   ·   baseline: {}".format(base[f])
        c.text((TL_X, PANEL_H + 18), state, font=font_mid, fill=(240, 240, 240))
        secs = f / 25
        c.text((TL_X, PANEL_H + 64), "fotogramma video {}   ·   {:d}:{:05.2f} del video".format(
            f, int(secs // 60), secs % 60), font=font_small, fill=(170, 175, 185))

        page_start = ORIGIN + ((f - ORIGIN) // PAGE) * PAGE
        x0 = int((page_start - ORIGIN) * PX)
        c.text((30, PANEL_H + 136), "Ground truth", font=font_small, fill=(170, 175, 185))
        c.text((30, PANEL_H + 164), "Baseline smooth50", font=font_small, fill=(170, 175, 185))
        canvas.paste(gt_img.crop((x0, 0, x0 + TL_W, 16)), (TL_X, PANEL_H + 140))
        canvas.paste(base_img.crop((x0, 0, x0 + TL_W, 16)), (TL_X, PANEL_H + 166))
        canvas.paste(lanes_img.crop((x0, 0, x0 + TL_W, n_lanes * LANE_H)), (TL_X, tag_y))
        ps, pe = max(0, page_start) / 25, min(N_VIDEO, page_start + PAGE) / 25
        c.text((TL_X + TL_W, PANEL_H + 108), "pagina {:d}:{:02d} - {:d}:{:02d}".format(
            int(ps // 60), int(ps % 60), int(pe // 60), int(pe % 60)), font=font_tag,
            fill=(130, 135, 145), anchor="ra")

        lo = np.searchsorted(tag_starts, f - 20000)
        active = [t for t in tags[lo:] if t["s"] <= f < t["e"]]
        c.text((30, PANEL_H + 206), "Smart Tagging", font=font_small, fill=(170, 175, 185))
        now = "   |   ".join(gt.tag_text(t) for t in active) if active else "nessuna fase"
        c.text((TL_X, PANEL_H + 204), now, font=font_small, fill=(240, 240, 240))
        for t in active:
            tx0 = TL_X + int((t["s"] - page_start) * PX)
            tx1 = TL_X + int((t["e"] - page_start) * PX) - 1
            ty0 = tag_y + t["lane"] * LANE_H
            c.rectangle([max(TL_X, tx0), ty0, min(TL_X + TL_W, max(tx1, tx0 + 2)), ty0 + LANE_H - 4],
                        outline=(255, 255, 255), width=2)
        cur = TL_X + int((f - page_start) * PX)
        c.rectangle([cur - 1, PANEL_H + 134, cur + 1, PANEL_H + 186], fill=(255, 255, 255))
        c.rectangle([cur - 1, tag_y - 4, cur + 1, tag_y + n_lanes * LANE_H], fill=(255, 255, 255))

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
