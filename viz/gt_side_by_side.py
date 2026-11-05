"""Side-by-side review video of the public ground-truth window.

Left: the tactical video (already cut to the window). Right: a 2D pitch drawn from the tracking of
the same frames. Bottom: the public ground-truth label of the frame (H / A / D), a timeline of
the whole window with a cursor, and under it the Smart Tagging phases of the window (one lane per
overlap level, coloured by team, set pieces in yellow) with the phases active on the frame.

Local viewing only: the output is a derivative of the match data, so it stays in data/ (ignored
by git) and must not be published or uploaded anywhere.

    .venv/bin/python viz/gt_side_by_side.py
"""

import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAMES, GAME, require  # noqa: E402
from baseline import team_of_each_track, team_names  # noqa: E402
from load_smart_tagging import load as load_tags  # noqa: E402

FIRST, LAST = 11185, 18684  # video frames of the public ground truth, inclusive
N = LAST - FIRST + 1
CLIP = os.path.join(ROOT, "data", "clips", "BAR-ATH_tactical_gt_public.mp4")
OUT = os.path.join(ROOT, "data", "clips", "BAR-ATH_gt_public_side_by_side.mp4")
GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")

PANEL_W, PANEL_H = 960, 540
BAR_H = 180
OUT_W, OUT_H = PANEL_W * 2, PANEL_H + BAR_H
LENGTH, WIDTH = GAMES[GAME]["pitch"]

FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FONT_REG = "/System/Library/Fonts/Supplemental/Arial.ttf"
GRASS, LINE = (38, 112, 58), (235, 245, 235)
HOME_FILL, HOME_EDGE = (0, 77, 152), (165, 0, 68)      # Barcelona blue, garnet edge
AWAY_FILL, AWAY_EDGE = (238, 37, 35), (255, 255, 255)  # Athletic red, white edge
LABEL_COLOR = {"H": HOME_FILL, "A": AWAY_FILL, "D": (110, 110, 110), "X": (40, 40, 40)}

# pitch placement inside the right panel
MARGIN = 30
SCALE = min((PANEL_W - 2 * MARGIN) / LENGTH, (PANEL_H - 2 * MARGIN) / WIDTH)
OX = (PANEL_W - LENGTH * SCALE) / 2
OY = (PANEL_H - WIDTH * SCALE) / 2


def to_px(x_m, y_m):
    return OX + x_m * SCALE, OY + y_m * SCALE


def pitch_background():
    img = Image.new("RGB", (PANEL_W, PANEL_H), (24, 70, 38))
    d = ImageDraw.Draw(img)
    d.rectangle([to_px(0, 0), to_px(LENGTH, WIDTH)], fill=GRASS)
    w = 2

    def rect(x0, y0, x1, y1):
        d.rectangle([to_px(x0, y0), to_px(x1, y1)], outline=LINE, width=w)

    rect(0, 0, LENGTH, WIDTH)
    d.line([to_px(LENGTH / 2, 0), to_px(LENGTH / 2, WIDTH)], fill=LINE, width=w)
    cx, cy = to_px(LENGTH / 2, WIDTH / 2)
    r = 9.15 * SCALE
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=LINE, width=w)
    d.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=LINE)
    for side in (0, 1):
        x_goal = 0 if side == 0 else LENGTH
        sgn = 1 if side == 0 else -1
        for depth, span in ((16.5, 40.32), (5.5, 18.32)):
            x1 = x_goal + sgn * depth
            rect(min(x_goal, x1), WIDTH / 2 - span / 2, max(x_goal, x1), WIDTH / 2 + span / 2)
        px, py = to_px(x_goal + sgn * 11, WIDTH / 2)
        d.ellipse([px - 3, py - 3, px + 3, py + 3], fill=LINE)
        gx = x_goal - sgn * 1.5
        rect(min(x_goal, gx), WIDTH / 2 - 3.66, max(x_goal, gx), WIDTH / 2 + 3.66)
    return img


def read_tracking():
    """Positions in metres for the window: players (N, slots, 2) and ball (N, 2), NaN if missing."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    teams = team_of_each_track(meta_path)
    players = np.full((N, len(teams), 2), np.nan)
    ball = np.full((N, 2), np.nan)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1  # tracking frame_id counts from 1, video frames from 0
            if frame < FIRST:
                continue
            if frame > LAST:
                break
            i = frame - FIRST
            for slot, group in enumerate(parts[1].split(";")):
                vals = group.split(",")
                if vals[0][0] != "N":
                    players[i, slot] = (float(vals[0]) * LENGTH, float(vals[1]) * WIDTH)
            vals = parts[2].split(",")
            if vals[0][0] != "N":
                ball[i] = (float(vals[0]) * LENGTH, float(vals[1]) * WIDTH)
    return teams, players, ball


def read_labels():
    labels = []
    with open(GT) as handle:
        next(handle)
        for line in handle:
            frame, label = line.strip().split(",")
            if FIRST <= int(frame) <= LAST:
                labels.append(label)
    return np.array(labels)


def timeline_strip(labels, width):
    """One pixel column per slice of the window, coloured by label."""
    strip = np.zeros((14, width, 3), dtype=np.uint8)
    for x in range(width):
        lab = labels[min(N - 1, int(x * N / width))]
        strip[:, x] = LABEL_COLOR.get(lab, (40, 40, 40))
    return Image.fromarray(strip)


TAG_LANE_H = 24
SET_PIECE = (230, 190, 40)
NEUTRAL = (120, 120, 120)


def read_tags(home, away):
    """Smart Tagging intervals overlapping the window, in window frames, with a lane each.

    end is exclusive (load_smart_tagging's convention). Lanes are assigned greedily so that
    overlapping phases sit on different rows.
    """
    tags = []
    for ev in load_tags():
        s, e = ev["start_frame"] - FIRST, ev["end_frame"] - FIRST
        if e <= 0 or s >= N:
            continue
        team = (ev["tags"].get("Team") or "").strip()
        code = (ev["code"] or "").strip()
        kind = (ev["tags"].get("Type") or "").strip()
        if code.lower() == "set pieces":
            color = SET_PIECE
        elif team == home:
            color = HOME_FILL
        elif team == away:
            color = AWAY_FILL
        else:
            color = NEUTRAL
        tags.append({"s": max(0, s), "e": min(N, e), "code": code, "team": team or "N/A",
                     "type": kind, "color": color})
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


def tag_text(t):
    name = t["code"].title()
    if t["type"]:
        name += " ({})".format(t["type"])
    return "{} · {}".format(name, t["team"])


def main():
    global BAR_H, OUT_H
    home, away = team_names(require(["atd_meta"])[0])
    teams, players, ball = read_tracking()
    labels = read_labels()
    tags, n_lanes = read_tags(home, away)
    print("window {}..{} ({} frames), labels {}, smart tagging {} phases on {} lanes".format(
        FIRST, LAST, N, len(labels), len(tags), n_lanes))
    if len(labels) != N:
        raise SystemExit("ground truth window does not match")
    tag_y = PANEL_H + 200            # top of the first Smart Tagging lane
    BAR_H = 200 + n_lanes * TAG_LANE_H + 24
    BAR_H += BAR_H % 2               # even height for yuv420p
    OUT_H = PANEL_H + BAR_H

    background = pitch_background()
    font_big = ImageFont.truetype(FONT, 70)
    font_mid = ImageFont.truetype(FONT, 34)
    font_small = ImageFont.truetype(FONT_REG, 22)
    text_of = {"H": "{} in possesso".format(home), "A": "{} in possesso".format(away),
               "D": "Palla morta", "X": "Non valutato"}
    tl_x, tl_w = 330, OUT_W - 330 - 40
    strip = timeline_strip(labels, tl_w)
    font_tag = ImageFont.truetype(FONT_REG, 15)

    def tag_box(t):
        x0 = tl_x + int(t["s"] * tl_w / N)
        x1 = max(x0 + 2, tl_x + int(t["e"] * tl_w / N) - 1)
        y0 = tag_y + t["lane"] * TAG_LANE_H
        return x0, y0, x1, y0 + TAG_LANE_H - 4

    # static Smart Tagging lanes, drawn once and pasted on every frame
    lanes = Image.new("RGB", (OUT_W, n_lanes * TAG_LANE_H), (16, 18, 22))
    ld = ImageDraw.Draw(lanes)
    for t in tags:
        x0, y0, x1, y1 = tag_box(t)
        ld.rectangle([x0, y0 - tag_y, x1, y1 - tag_y], fill=t["color"])
        label = t["code"].title() if not t["type"] else t["type"]
        if ld.textlength(label, font=font_tag) < x1 - x0 - 8:
            ld.text((x0 + 4, y0 - tag_y + 2), label, font=font_tag,
                    fill=(20, 20, 20) if t["color"] == SET_PIECE else (245, 245, 245))

    reader = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", CLIP, "-vf", "scale={}:{}".format(PANEL_W, PANEL_H),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    writer = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", "{}x{}".format(OUT_W, OUT_H), "-r", "25", "-i", "-",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", OUT],
        stdin=subprocess.PIPE)

    frame_bytes = PANEL_W * PANEL_H * 3
    written = 0
    for i in range(N):
        raw = reader.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        canvas = Image.new("RGB", (OUT_W, OUT_H), (16, 18, 22))
        canvas.paste(Image.frombuffer("RGB", (PANEL_W, PANEL_H), raw), (0, 0))

        pitch = background.copy()
        d = ImageDraw.Draw(pitch)
        for slot in range(players.shape[1]):
            if np.isnan(players[i, slot, 0]):
                continue
            px, py = to_px(*players[i, slot])
            fill, edge = (HOME_FILL, HOME_EDGE) if teams[slot] == 0 else (AWAY_FILL, AWAY_EDGE)
            d.ellipse([px - 8, py - 8, px + 8, py + 8], fill=fill, outline=edge, width=3)
        if np.isnan(ball[i, 0]):
            d.text((OX + 8, OY + 6), "palla non tracciata", font=font_small, fill=(255, 220, 120))
        else:
            bx, by = to_px(*ball[i])
            d.ellipse([bx - 6, by - 6, bx + 6, by + 6], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
        canvas.paste(pitch, (PANEL_W, 0))

        c = ImageDraw.Draw(canvas)
        lab = labels[i]
        c.rectangle([30, PANEL_H + 14, 30 + 260, PANEL_H + 100], fill=LABEL_COLOR.get(lab))
        c.text((30 + 130, PANEL_H + 57), lab, font=font_big, fill=(255, 255, 255), anchor="mm")
        c.text((tl_x, PANEL_H + 18), text_of.get(lab, lab), font=font_mid, fill=(240, 240, 240))
        secs = i / 25
        c.text((tl_x, PANEL_H + 64), "fotogramma video {}   ·   {:d}:{:05.2f} della finestra".format(
            FIRST + i, int(secs // 60), secs % 60), font=font_small, fill=(170, 175, 185))
        c.text((30, PANEL_H + 112), "Possesso (ground truth)", font=font_small, fill=(170, 175, 185))
        canvas.paste(strip, (tl_x, PANEL_H + 118))

        active = [t for t in tags if t["s"] <= i < t["e"]]
        c.text((30, PANEL_H + 160), "Smart Tagging", font=font_small, fill=(170, 175, 185))
        now = "   |   ".join(tag_text(t) for t in active) if active else "nessuna fase"
        c.text((tl_x, PANEL_H + 158), now, font=font_small, fill=(240, 240, 240))
        canvas.paste(lanes, (0, tag_y))
        for t in active:
            c.rectangle(tag_box(t), outline=(255, 255, 255), width=2)

        cur = tl_x + int(i * tl_w / N)
        c.rectangle([cur - 1, PANEL_H + 112, cur + 1, PANEL_H + 138], fill=(255, 255, 255))
        c.rectangle([cur - 1, tag_y - 4, cur + 1, tag_y + n_lanes * TAG_LANE_H], fill=(255, 255, 255))

        writer.stdin.write(canvas.tobytes())
        written += 1
    writer.stdin.close()
    writer.wait()
    reader.stdout.close()
    reader.wait()
    print("wrote {} frames -> {}".format(written, OUT))


if __name__ == "__main__":
    main()
