"""Review video of possession v2 (possession changes + dead ball), with every rule live on screen.

Top: tactical video (left), broadcast video synchronised to it by sync_broadcast.py (centre, with
LIVE / NOT LIVE from the tracking check), and the 2D pitch (right, tactical or fused tracking). On the pitch: control radii around the ball, ball
velocity arrow, a line to the closest player of each team, and a white ring on the K players used
by the "players far" rule. Bottom:
  - the v2 label (as written in the submission, changes back-dated) and the live machine state;
  - three 5-minute strips: ground truth (public window only), v2, baseline smooth50;
  - the scores over the page: possession challenger (team colours), dead ball (white) while in
    play, alive play (green) while dead, with the changes;
  - a +/-5 s zoom with the same scores, ball speed, touches and whistle-like tones;
  - the live value of every rule.

Local viewing only: a derivative of the match data, it stays in data/clips/ (ignored by git).

    .venv/bin/python solution/possession_v2.py
    .venv/bin/python viz/v2_side_by_side.py                                   # public 5 minutes
    .venv/bin/python viz/v2_side_by_side.py --first 0 --last 199318 --out v2_match.mp4
    .venv/bin/python viz/v2_side_by_side.py --tracking fused        # v2 run on fused tracking
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
import possession_v2 as v2  # noqa: E402
import sync_broadcast  # noqa: E402
import tracking  # noqa: E402
from games import video_path  # noqa: E402

N_VIDEO = ms.N_VIDEO
PANEL_W, PANEL_H, OUT_W = gt.PANEL_W, gt.PANEL_H, gt.OUT_W
PAGE, ORIGIN, TL_X, TL_W, PX = ms.PAGE, ms.ORIGIN, ms.TL_X, ms.TL_W, ms.PX
TEAM_LINE = ((90, 160, 255), (255, 110, 100))
DEAD_LINE, ALIVE_LINE = (235, 235, 235), (110, 220, 120)
STATE_LINE = {0: TEAM_LINE[0], 1: TEAM_LINE[1], 2: (170, 170, 170)}
CHART_H = 70
ZOOM_S = 5.0
ZOOM_X, ZOOM_W, ZOOM_H = 30, 900, 210
INFO_X = 960
COL_VAL = 1270     # "misura adesso"
COL_NOW = 1770     # "punti adesso", right-aligned
COL_TOT = 1890     # "totale", right-aligned
BAR_H = 600
TOP_W, TOP_H = 640, 360   # three panels side by side: tactical | broadcast | 2D
OUT_H = TOP_H + BAR_H
DIM, TEXT, ACCENT = (150, 155, 165), (240, 240, 240), (255, 220, 120)
DECAY = 0.5 ** (1.0 / (v2.PARAMS["HALF_LIFE"] * 25))
REASON_IT = {"control": "controllo", "touch": "tocco", "ball out": "palla fuori",
             "players stopped": "giocatori fermi", "whistle": "fischio", "corner": "calcio d'angolo",
             "corner kick": "calcio d'angolo battuto", "restart touch": "tocco di ripresa",
             "alive play": "gioco ripreso", "dead ball too short": "palla morta troppo breve, annullata"}


def page_chart(series, changes):
    """Whole-match image of the scores (each 0..1.25 of its threshold), x = (frame - ORIGIN) * PX."""
    n_pages = (N_VIDEO - ORIGIN + PAGE - 1) // PAGE
    width = int(n_pages * PAGE * PX) + 2
    img = np.full((CHART_H, width, 3), 22, dtype=np.uint8)
    cols = ((np.arange(N_VIDEO) - ORIGIN) * PX).astype(np.int64)
    for values, colour in series:
        col_max = np.zeros(width)
        np.maximum.at(col_max, cols, values)
        h = np.clip(col_max / 1.25, 0, 1) * (CHART_H - 2)
        for x in np.flatnonzero(h >= 1):
            y0 = CHART_H - 1 - int(h[x])
            img[y0:CHART_H - 1, x] = np.maximum(img[y0:CHART_H - 1, x], colour)
    y_thr = CHART_H - 1 - int(1 / 1.25 * (CHART_H - 2))
    img[y_thr, ::3] = ACCENT
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    for decided, start, new in changes:
        x = int((start - ORIGIN) * PX)
        d.line([x, 0, x, CHART_H - 1], fill=(255, 255, 255), width=1)
        d.rectangle([x - 2, 0, x + 2, 4], fill=STATE_LINE[new])
    return pil


def fmt(x, nd=1, unit=""):
    return "--" if x is None or not np.isfinite(x) else "{:.{}f}{}".format(x, nd, unit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=ms.GT_FIRST)
    ap.add_argument("--last", type=int, default=ms.GT_LAST)
    ap.add_argument("--out", default="BAR-ATH_v2_gt_public.mp4")
    ap.add_argument("--tracking", choices=("tactical", "fused"), default="tactical")
    args = ap.parse_args()
    first, last = max(0, args.first), min(N_VIDEO - 1, args.last)
    n = last - first + 1
    out_path = os.path.join(ms.CLIPS, os.path.basename(args.out))

    teams, players, ball, _, (home, away) = tracking.load(source=args.tracking)
    names = (home, away)
    csv_path, diag_path = v2.outputs(args.tracking)
    z = np.load(diag_path)
    n_tac_slots = len(tracking.load()[0])
    ball_src = None
    if args.tracking == "fused":
        ball_src = np.load(os.path.join(ROOT, "data", "derived", "BAR-ATH_fused_tracking.npz"))["ball_src"]
    sync = np.load(sync_broadcast.OUT) if os.path.exists(sync_broadcast.OUT) else None
    if sync is not None:
        b_of_t, b_live_flag = sync["b_of_t"], sync["live"]
    else:
        b_of_t, b_live_flag = np.full(N_VIDEO, -1), np.zeros(1, bool)
    p = v2.PARAMS
    thr, thr_dead = float(z["threshold"]), float(z["threshold_dead"])
    score, live, dist, slot = z["score"], z["live"], z["dist"], z["slot"]
    dead_score, alive_score = z["dead_score"], z["alive_score"]
    speed, dv, angle, vel = z["speed"], z["dv"], z["angle"], z["vel"]
    slow, conf, ctrl = z["slow"], z["conf"], z["ctrl"]
    touch_team, is_touch = z["touch_team"], z["is_touch"]
    out_dist, far, pspeed = z["out_dist"], z["far"], z["pspeed"]
    f_dist, f_speed, deadness = z["f_dist"], z["f_speed"], z["deadness"]
    rhythm, f_rhythm, deadness_exit = z["rhythm"], z["f_rhythm"], z["deadness_exit"]
    duel = z["duel"]
    duel_frames = np.flatnonzero(duel)
    gap_out = z["gap_out"]
    offs = [tuple(int(v) for v in row) for row in z["offsides"]]
    refused = [tuple(int(v) for v in row) for row in z["refused"]]
    cause_s = p["CAUSE_S"]
    wlevel, wconf = z["whistle_level"], z["whistle_conf"]
    changes = [tuple(int(v) for v in row) for row in z["changes"]]
    reasons = list(z["reasons"])
    corners = [tuple(int(v) for v in row) for row in z["corners"]]
    ch_start = np.array([c[1] for c in changes])
    ch_decided = np.array([c[0] for c in changes])

    gt_labels = ms.read_csv_labels(ms.GT_CSV)
    base = ms.read_csv_labels(ms.BASELINE)
    lab_v2 = ms.read_csv_labels(csv_path)
    is_gt = np.zeros(N_VIDEO, dtype=bool)
    is_gt[ms.GT_FIRST:ms.GT_LAST + 1] = True

    gt_img, base_img, _ = ms.long_strips(gt_labels, is_gt, base, [], 1, None)
    _, v2_img, _ = ms.long_strips(gt_labels, is_gt, lab_v2, [], 1, None)
    in_play = (live == 0) | (live == 1)
    chart_img = page_chart([(score[0] / thr, TEAM_LINE[0]), (score[1] / thr, TEAM_LINE[1]),
                            (np.where(in_play, dead_score / thr_dead, 0), DEAD_LINE),
                            (np.where(live == 2, alive_score, 0), ALIVE_LINE)],
                           [(a, b, c) for a, b, c in changes])
    touch_frames = np.flatnonzero(is_touch)
    whistle_on = wlevel > 0

    background = gt.pitch_background()
    font_big = ImageFont.truetype(gt.FONT, 70)
    font_mid = ImageFont.truetype(gt.FONT, 34)
    font_small = ImageFont.truetype(gt.FONT_REG, 21)
    font_bold = ImageFont.truetype(gt.FONT, 21)
    font_tag = ImageFont.truetype(gt.FONT_REG, 15)
    font_row = ImageFont.truetype(gt.FONT_REG, 18)
    state_of = {"H": "{} in possesso".format(home), "A": "{} in possesso".format(away),
                "D": "Palla morta", "X": "Nessuna etichetta"}
    zoom_frames = int(ZOOM_S * 25)
    k_near = p["K_NEAR"]

    ptot = {k: z["ptot_" + k] for k in v2.P_KEYS}
    dtot = {k: z["dtot_" + k] for k in v2.D_KEYS}
    atot = {k: z["atot_" + k] for k in v2.A_KEYS}
    d_reset, a_reset = z["d_reset"], z["a_reset"]
    visible = ~np.isnan(ball[:, 0])
    dead_reason_at = {}
    for (dec, st, new), why in zip(changes, reasons):
        if new == 2:
            dead_reason_at[dec] = why
    dead_decided = np.array(sorted(dead_reason_at))
    reset_it = {v2.RESET_FAST: "palla veloce: il gioco è vivo",
                v2.RESET_TOUCH: "tocco in campo: il gioco è vivo",
                v2.RESET_NO_CAUSE: "nessun contrasto né fuorigioco",
                v2.RESET_OUT: "palla fuori: niente è vivo",
                v2.RESET_KICKOFF: "prima del calcio d'inizio si riparte solo dal centro"}

    def code(team):
        return names[team][:3].upper()

    def pts(x, per_s=False):
        if x is None or not np.isfinite(x):
            return "--"
        return "{:+.2f}{}".format(x, "/s" if per_s else "")

    def recent(arr, f, span=12):
        """The latest non-zero reset code in the last `span` frames, or 0."""
        seg = arr[max(0, f - span):f + 1]
        nz = seg[seg != 0]
        return int(nz[-1]) if len(nz) else 0

    def last_touch_of(team, f, span=25):
        k = np.searchsorted(touch_frames, f + 1) - 1
        while k >= 0 and f - touch_frames[k] <= span:
            if touch_team[touch_frames[k]] == team:
                return int(touch_frames[k])
            k -= 1
        return None

    def possession_rows(f, own, chal, colour):
        sc = float(score[chal, f])
        rows = [("head", ("SOGLIA CAMBIO POSSESSO → {}".format(names[chal]),
                          "{:.2f} / {:.2f}".format(sc, thr), sc / thr), colour),
                ("cols", ("parametro", "misura adesso", "punti adesso", "totale"), DIM)]
        cv = float(ctrl[chal, f])
        if visible[f]:
            meas = "{} {} · {} {} · palla {} m/s → {:.2f}".format(
                code(chal), fmt(dist[chal, f], 1, " m"), code(own),
                fmt(dist[own, f], 1, " m"), fmt(speed[f], 1), cv)
        else:
            meas = "palla non tracciata"
        rows.append(("row", ("1 Controllo di " + code(chal), meas,
                             pts(p["RATE_CTRL"] * cv, True), pts(ptot["ctrl"][chal, f])),
                     TEXT if cv > 0 else DIM))
        tt = last_touch_of(chal, f)
        rows.append(("row", ("2 Tocco di " + code(chal),
                             "Δv {} m/s, {}° ({:.1f} s fa)".format(fmt(dv[tt], 1), fmt(angle[tt], 0),
                                                                   (f - tt) / 25) if tt is not None
                             else "nessun tocco nell'ultimo secondo",
                             pts(p["W_TOUCH"]) if tt is not None else "--",
                             pts(ptot["touch"][chal, f])), TEXT if tt is not None else DIM))
        rows.append(("row", ("4 Conferma (2 s dopo il tocco)",
                             "{:+.2f}: poi la tiene {}".format(conf[chal, tt], code(chal) if conf[chal, tt] > 0
                                                               else code(own)) if tt is not None else "--",
                             pts(p["W_LOOK"] * conf[chal, tt]) if tt is not None else "--",
                             pts(ptot["look"][chal, f])), TEXT if tt is not None else DIM))
        fast_now = visible[f] and np.isfinite(speed[f]) and slow[f] == 0
        rows.append(("row", ("3 Palla veloce senza deviazione",
                             "sì ({} m/s): il controllo non conta".format(fmt(speed[f], 1)) if fast_now
                             else "no", "", ""), ACCENT if fast_now else DIM))
        ot = last_touch_of(own, f)
        cvo = float(ctrl[own, f])
        rows.append(("row", ("− Chi ha palla ({}): controllo, tocchi".format(code(own)),
                             "controllo {:.2f}{}".format(cvo, ", tocco" if ot is not None and f - ot < 3 else ""),
                             pts(-p["RATE_OWN"] * cvo, True), pts(ptot["own"][chal, f])),
                     TEXT if cvo > 0 else DIM))
        rows.append(("row", ("− Decadimento", "il punteggio si dimezza in 1 s",
                             pts(-sc * (1 - DECAY) * 25, True), pts(ptot["decay"][chal, f])), DIM))
        rows.append(("row", ("", "", "somma", "= {:.2f}".format(sc)), TEXT))
        return rows

    def dead_rows(f, colour):
        sd = float(dead_score[f])
        rows = [("head", ("SOGLIA PALLA MORTA", "{:.2f} / {:.2f}".format(sd, thr_dead), sd / thr_dead),
                 colour),
                ("cols", ("parametro (+ sopra 0.5, − sotto)", "misura adesso", "punti adesso", "totale"),
                 DIM)]
        od = out_dist[f]
        rows.append(("row", ("1 Palla fuori",
                             "fuori di {:.1f} m: palla morta subito".format(od) if visible[f] and od > p["OUT_MARGIN"]
                             else ("in campo" if visible[f] else "non tracciata: la posizione non decide"),
                             "", ""), ACCENT if visible[f] and od > p["OUT_MARGIN"] else DIM))
        rw = p["RATE_DEAD"] * (1.0 if visible[f] else p["GAP_WEIGHT"])
        fd, fr = (float(f_dist[f]) if visible[f] else np.nan), float(f_rhythm[f])
        both = np.isfinite(fd) and np.isfinite(fr)
        wr = p["W_RHYTHM"]
        nd = rw * (2 * (1 - wr) * (fd - 0.5) if both else (2 * fd - 1)) if np.isfinite(fd) else np.nan
        nr = rw * (2 * wr * (fr - 0.5) if both else (2 * fr - 1)) if np.isfinite(fr) else np.nan
        if recent(d_reset, f, 0):
            nd = nr = 0.0
        rows.append(("row", ("2 Distanza {} più vicini ({:.0f}%)".format(k_near, 100 * (1 - p["W_RHYTHM"])),
                             "{} → {}".format(fmt(far[f], 1, " m"), fmt(fd, 2)) if visible[f]
                             else "serve la palla", pts(nd, True), pts(dtot["dist"][f])),
                     TEXT if visible[f] else DIM))
        rows.append(("row", ("3 Frenata del ritmo ({:.0f}%)".format(100 * p["W_RHYTHM"]),
                             "{} ora, {}× i 4 s prima → {}".format(fmt(pspeed[f], 2, " m/s"),
                                                                 fmt(rhythm[f], 2), fmt(fr, 2)),
                             pts(nr, True), pts(dtot["rhythm"][f])), TEXT))
        wk = np.flatnonzero(whistle_on[max(0, f - 25):f + 1])
        if len(wk):
            wf = max(0, f - 25) + wk[-1]
            wtxt = "{:.0f} dB {:.1f} s fa, frenata dopo {:.2f}".format(wlevel[wf], (f - wf) / 25, wconf[wf])
            wnow = pts(p["W_WHISTLE"] * wconf[wf])
        else:
            wtxt, wnow = "nessuno nell'ultimo secondo", "--"
        rows.append(("row", ("4 Fischio (vale solo con frenata)", wtxt, wnow, pts(dtot["whistle"][f])),
                     TEXT if len(wk) else DIM))
        if not visible[f]:
            rows.append(("row", ("6 Palla non tracciata", "decidono solo gli altri parametri", "", ""), ACCENT))
        k = np.searchsorted(duel_frames, f + 1) - 1
        last_duel = (f - duel_frames[k]) / 25 if k >= 0 else np.inf
        last_off = [o for o in offs if 0 <= f - o[0] <= cause_s * 25 * 2]
        if gap_out[f]:
            cause, ok = "la palla ricompare fuori dal campo", True
        elif last_off:
            cause, ok = "fuorigioco di {} ({:.1f} s fa)".format(names[last_off[-1][2]],
                                                               (f - last_off[-1][0]) / 25), True
        elif last_duel <= cause_s:
            cause, ok = "contrasto vicino alla palla {:.1f} s fa".format(last_duel), True
        else:
            cause, ok = "nessun contrasto né fuorigioco", False
        pen = 0.0 if ok else float(z["no_cause_penalty"])
        rows.append(("row", ("7 Causa (se manca: −{:.2f})".format(float(z["no_cause_penalty"])), cause,
                             "" if ok else pts(-pen), "" if ok else pts(-pen)),
                     ACCENT if ok else TEXT))
        rr = recent(d_reset, f)
        if rr:
            rows.append(("text", "Azzerato: " + reset_it[rr], ACCENT))
        rows.append(("row", ("", "", "somma", "= {:.2f}".format(sd - pen)), TEXT))
        rows[0] = ("head", ("SOGLIA PALLA MORTA", "{:.2f} / {:.2f}".format(sd - pen, thr_dead),
                            (sd - pen) / thr_dead), colour)
        return rows

    def restart_rows(f):
        sa = float(alive_score[f])
        k = np.searchsorted(dead_decided, f + 1) - 1
        why = dead_reason_at[dead_decided[k]] if k >= 0 else "inizio tempo"
        st = ch_start[ch_decided <= f]
        since = (f - st[-1]) / 25 if len(st) else 0
        rows = [("head", ("SOGLIA RIPRESA DEL GIOCO", "{:.2f} / 1.00".format(sa), sa), ALIVE_LINE),
                ("text", "palla morta da {:.1f} s, per: {}".format(since, REASON_IT.get(why, why)), TEXT),
                ("cols", ("parametro", "misura adesso", "punti adesso", "totale"), DIM)]
        tt = touch_frames[np.searchsorted(touch_frames, f + 1) - 1] if len(touch_frames) else None
        rows.append(("row", ("Tocco di ripresa (in campo)", "nessuno: si aspetta" if tt is None or f - tt > 12
                             else "tocco di {}".format(names[touch_team[tt]] if touch_team[tt] >= 0 else "?"),
                             "riprende subito", ""), DIM))
        ex = float(deadness_exit[f])
        alive = max(0.0, 1.0 - 2.0 * ex)
        on = visible[f] and np.nan_to_num(out_dist[f], nan=1) == 0
        rows.append(("row", ("Gioco vivo (giocatori ripartono)",
                             "{} (lenti {}), dist. {} → indice {:.2f}".format(
                                 fmt(pspeed[f], 2, " m/s"), fmt(f_speed[f], 2), fmt(f_dist[f], 2), ex)
                             if on else ("palla fuori" if visible[f] else "palla non tracciata"),
                             pts(p["ALIVE_RATE"] * alive if on else 0, True), pts(atot["alive"][f])),
                     TEXT if alive > 0 and on else DIM))
        fast_on = on and speed[f] > p["V_FAST"] and alive > 0
        rows.append(("row", ("Palla veloce, giocatori in moto",
                             "{} m/s".format(fmt(speed[f], 1)) if visible[f] else "--",
                             pts(p["ALIVE_RATE"] if fast_on else 0, True), pts(atot["fast"][f])),
                     TEXT if fast_on else DIM))
        rows.append(("row", ("− Decadimento", "il punteggio si dimezza in 1 s",
                             pts(-sa * (1 - DECAY) * 25, True), pts(atot["decay"][f])), DIM))
        if why in ("whistle", "players stopped"):
            rows.append(("text", "Se il gioco riprende prima di {:.0f} s (o entro {:.0f} s dalla decisione) "
                         "la palla morta viene annullata".format(p["D_MIN_S"], p["D_MIN_AFTER_S"]), DIM))
        rr = recent(a_reset, f)
        if rr:
            rows.append(("text", "Azzerato: " + reset_it[rr], ACCENT))
        rows.append(("row", ("", "", "somma", "= {:.2f}".format(sa)), TEXT))
        return rows

    def panel_rows(f, owner):
        if owner == 2:
            rows = restart_rows(f)
        elif owner in (0, 1):
            chal = 1 - owner
            fp, fd = score[chal, f] / thr, dead_score[f] / thr_dead
            if fd > fp:
                rows = dead_rows(f, DEAD_LINE)
                rows.append(("text", "Altra soglia: cambio possesso → {} {:.2f} / {:.2f}".format(
                    names[chal], score[chal, f], thr), DIM))
            else:
                rows = possession_rows(f, owner, chal, TEAM_LINE[chal])
                rows.append(("text", "Altra soglia: palla morta {:.2f} / {:.2f}".format(
                    dead_score[f], thr_dead), DIM))
        else:
            rows = [("bold", "Inizio tempo", TEXT)]
        j = np.searchsorted(ch_decided, f + 1) - 1
        if j >= 0 and f - ch_decided[j] <= 50:
            dec, st, new = changes[j]
            what = "PALLA MORTA" if new == 2 else names[new]
            rows.append(("bold", "CAMBIO → {} ({}), retrodatato di {:.2f} s".format(
                what, REASON_IT.get(reasons[j], reasons[j]), (dec - st) / 25), STATE_LINE[new]))
        return rows

    seek = ["-ss", "{:.2f}".format(first / 25)] if first > 0 else []
    reader = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-hwaccel", "videotoolbox"] + seek + ["-i", ms.VIDEO,
         "-frames:v", str(n), "-vf", "scale={}:{}".format(TOP_W, TOP_H),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE)
    # broadcast: decoded forward from the first synchronised frame of the range
    need = b_of_t[first:last + 1]
    need = need[need >= 0]
    b_reader, b_cur, b_img = None, -1, None
    if len(need):
        b_first = int(need.min())
        b_reader = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-hwaccel", "videotoolbox", "-ss", "{:.2f}".format(b_first / 25),
             "-i", video_path(source="broadcast"), "-vf", "scale={}:{}".format(TOP_W, TOP_H),
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
        b_cur = b_first - 1
    writer = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", "{}x{}".format(OUT_W, OUT_H), "-r", "25", "-i", "-",
         # the tactical video's own audio, cut to the same frames
         "-ss", "{:.2f}".format(first / 25), "-t", "{:.2f}".format(n / 25), "-i", ms.VIDEO,
         "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "160k",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
         "-shortest", out_path],
        stdin=subprocess.PIPE)

    widest = [(0, ""), (0, ""), (0, "")]
    frame_bytes = TOP_W * TOP_H * 3
    written = 0
    y0 = TOP_H
    font_panel = ImageFont.truetype(gt.FONT, 18)
    for i in range(n):
        raw = reader.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        f = first + i
        canvas = Image.new("RGB", (OUT_W, OUT_H), (16, 18, 22))
        canvas.paste(Image.frombuffer("RGB", (TOP_W, TOP_H), raw), (0, 0))

        # ---- broadcast, synchronised ------------------------------------------------------
        target = int(b_of_t[f])
        shown = False
        if b_reader is not None and target >= 0 and target >= b_cur:
            while b_cur < target:
                braw = b_reader.stdout.read(frame_bytes)
                if len(braw) < frame_bytes:
                    b_reader = None
                    break
                b_img, b_cur = braw, b_cur + 1
            if b_cur == target and b_img is not None:
                canvas.paste(Image.frombuffer("RGB", (TOP_W, TOP_H), b_img), (TOP_W, 0))
                shown = True
        cb = ImageDraw.Draw(canvas)
        if shown:
            is_live = bool(b_live_flag[target])
            tag = "LIVE · broadcast {}".format(target) if is_live else \
                "NON LIVE (replay, primo piano o stacco) · {}".format(target)
            cb.rectangle([TOP_W + 8, TOP_H - 34, TOP_W + 16 + font_panel.getlength(tag), TOP_H - 8],
                         fill=(30, 120, 60) if is_live else (140, 100, 20))
            cb.text((TOP_W + 12, TOP_H - 31), tag, font=font_panel, fill=(255, 255, 255))
        else:
            cb.text((TOP_W + TOP_W / 2, TOP_H / 2), "broadcast: nessun fotogramma sincronizzato",
                    font=font_panel, fill=DIM, anchor="mm")
        for k_, title in enumerate(("TATTICA", "BROADCAST sincronizzata",
                                    "2D " + ("fuso tattica + broadcast" if args.tracking == "fused" else "tattica"))):
            cb.rectangle([k_ * TOP_W + 6, 6, k_ * TOP_W + 14 + font_panel.getlength(title), 30], fill=(0, 0, 0))
            cb.text((k_ * TOP_W + 10, 8), title, font=font_panel, fill=(255, 255, 255))

        # ---- pitch -----------------------------------------------------------------------
        pitch = background.copy()
        d = ImageDraw.Draw(pitch)
        has_ball = not np.isnan(ball[f, 0])
        ring = set()
        if has_ball:
            bx, by = gt.to_px(*ball[f])
            for r, col in ((p["R_FAR"], (200, 200, 200)), (p["R_NEAR"], (255, 255, 255))):
                rp = r * gt.SCALE
                d.ellipse([bx - rp, by - rp, bx + rp, by + rp], outline=col, width=1)
            dd = np.linalg.norm(players[f] - ball[f], axis=1)
            dd = np.where(np.isnan(dd), np.inf, dd)
            ring = set(int(s_) for s_ in np.argsort(dd)[:k_near] if np.isfinite(dd[s_]))
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
            if s_ >= n_tac_slots:  # seen only by the broadcast
                edge = (255, 255, 255)
            if s_ in ring:
                d.ellipse([px - 13, py - 13, px + 13, py + 13], outline=(255, 255, 255), width=2)
            d.ellipse([px - 8, py - 8, px + 8, py + 8], fill=fill, outline=edge, width=3)
        if has_ball:
            if np.isfinite(vel[f, 0]):
                ex, ey = gt.to_px(ball[f, 0] + vel[f, 0] * 0.5, ball[f, 1] + vel[f, 1] * 0.5)
                d.line([bx, by, ex, ey], fill=ACCENT, width=3)
            d.ellipse([bx - 6, by - 6, bx + 6, by + 6], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
        else:
            d.text((gt.OX + 8, gt.OY + 6), "palla non tracciata", font=font_mid, fill=ACCENT)
        if ball_src is not None and has_ball:
            src_txt = {1: "palla: solo tattica", 2: "palla: solo broadcast", 3: "palla: media delle due"}
            d.text((gt.OX + 8, gt.OY + 6), src_txt.get(int(ball_src[f]), ""), font=font_mid, fill=ACCENT)
        canvas.paste(pitch.resize((TOP_W, TOP_H), Image.LANCZOS), (2 * TOP_W, 0))

        c = ImageDraw.Draw(canvas)
        # ---- label and state -------------------------------------------------------------
        lab = lab_v2[f]
        c.rectangle([30, y0 + 14, 290, y0 + 100], fill=gt.LABEL_COLOR.get(lab, (40, 40, 40)))
        c.text((160, y0 + 57), lab, font=font_big, fill=(255, 255, 255), anchor="mm")
        c.text((160, y0 + 108), "V2 (retrodatata)", font=font_small, fill=TEXT, anchor="mt")
        line = "v2: " + state_of.get(lab, lab)
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
        for k, (name, img) in enumerate((("Ground truth", gt_img),
                                         ("v2 fuso" if args.tracking == "fused" else "v2 (nostro)", v2_img),
                                         ("Baseline smooth50", base_img))):
            c.text((30, y0 + 136 + 26 * k), name, font=font_small, fill=(170, 175, 185))
            canvas.paste(img.crop((x0, 0, x0 + TL_W, 16)), (TL_X, y0 + 140 + 26 * k))
        c.text((30, y0 + 222), "Punteggi", font=font_small, fill=(170, 175, 185))
        c.text((30, y0 + 246), "soglia = linea gialla", font=font_tag, fill=DIM)
        c.text((30, y0 + 266), "bianco palla morta, verde ripresa", font=font_tag, fill=DIM)
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

        def ys(vals, vmax):
            return zy + ZOOM_H - 4 - np.clip(vals / vmax, 0, 1) * (ZOOM_H - 30)

        y_thr = ys(np.array([1.0]), 1.25)[0]
        for xx in range(ZOOM_X, ZOOM_X + ZOOM_W, 8):
            c.line([xx, y_thr, xx + 4, y_thr], fill=ACCENT)
        sp = np.nan_to_num(speed[fr], nan=-1)
        pts_sp = [(x_, y_) for x_, y_, v_ in zip(xs, ys(sp, 25.0), sp) if v_ >= 0]
        if len(pts_sp) > 1:
            c.point(pts_sp, fill=(120, 125, 135))
        lv = live[fr]
        for values, colour in ((score[0, fr] / thr, TEAM_LINE[0]), (score[1, fr] / thr, TEAM_LINE[1]),
                               (np.where((lv == 0) | (lv == 1), dead_score[fr] / thr_dead, 0), DEAD_LINE),
                               (np.where(lv == 2, alive_score[fr], 0), ALIVE_LINE)):
            c.line(list(zip(xs, ys(values, 1.25))), fill=colour, width=2)
        for wf in fr[whistle_on[fr]]:
            wx = ZOOM_X + (wf - lo) / (2 * zoom_frames) * ZOOM_W
            c.line([wx, zy + 18, wx, zy + 26], fill=ACCENT, width=1)
        for tf in touch_frames[np.searchsorted(touch_frames, lo):np.searchsorted(touch_frames, hi + 1)]:
            tx = ZOOM_X + (tf - lo) / (2 * zoom_frames) * ZOOM_W
            team = touch_team[tf]
            col = TEAM_LINE[team] if team >= 0 else (160, 160, 160)
            c.polygon([(tx, zy + 4), (tx - 5, zy + 14), (tx + 5, zy + 14)], fill=col)
        for (s1, s0, new) in changes:
            if lo <= s0 <= hi:
                sx = ZOOM_X + (s0 - lo) / (2 * zoom_frames) * ZOOM_W
                c.line([sx, zy, sx, zy + ZOOM_H], fill=STATE_LINE[new], width=2)
            if lo <= s1 <= hi:
                sx = ZOOM_X + (s1 - lo) / (2 * zoom_frames) * ZOOM_W
                c.line([sx, zy + ZOOM_H - 30, sx, zy + ZOOM_H], fill=ACCENT, width=2)
        c.line([ZOOM_X + ZOOM_W / 2, zy, ZOOM_X + ZOOM_W / 2, zy + ZOOM_H], fill=(255, 255, 255), width=1)
        c.text((ZOOM_X + 6, zy + ZOOM_H + 4), "-5 s   punteggi · velocità palla 0-25 m/s (grigio) · ▲ tocchi · "
               "| giallo = fischio · linea = cambio datato, gialla in basso = decisione   +5 s",
               font=font_tag, fill=DIM)

        # ---- live rules: the breakdown of the score closest to its threshold -------------
        owner = int(live[f])
        rows = panel_rows(f, owner)
        y = zy
        for kind, cells, col in rows:
            if kind == "row":
                for ci, txt in enumerate(cells[:2]):
                    widest[ci] = max(widest[ci], (font_row.getlength(txt), txt))
            elif kind in ("text", "bold"):
                widest[2] = max(widest[2], (font_bold.getlength(cells), cells))
            if kind == "head":
                title, value, frac = cells
                c.text((INFO_X, y), title, font=font_bold, fill=col)
                c.text((COL_TOT, y), value, font=font_bold, fill=col, anchor="ra")
                bx0, bx1 = COL_TOT - 330, COL_TOT - 120
                c.rectangle([bx0, y + 5, bx1, y + 17], outline=(90, 95, 105))
                c.rectangle([bx0, y + 5, bx0 + int((bx1 - bx0) * min(1.0, max(0.0, frac))), y + 17], fill=col)
            elif kind == "cols":
                c.text((INFO_X, y), cells[0], font=font_row, fill=DIM)
                c.text((COL_VAL, y), cells[1], font=font_row, fill=DIM)
                c.text((COL_NOW, y), cells[2], font=font_row, fill=DIM, anchor="ra")
                c.text((COL_TOT, y), cells[3], font=font_row, fill=DIM, anchor="ra")
            elif kind == "row":
                name, value, now, tot = cells
                c.text((INFO_X, y), name, font=font_row, fill=col)
                c.text((COL_VAL, y), value, font=font_row, fill=col)
                c.text((COL_NOW, y), now, font=font_row, fill=col, anchor="ra")
                c.text((COL_TOT, y), tot, font=font_bold if tot.startswith("=") else font_row,
                       fill=col, anchor="ra")
            else:
                c.text((INFO_X, y), cells, font=font_bold if kind == "bold" else font_row, fill=col)
            y += 22

        writer.stdin.write(canvas.tobytes())
        written += 1
    writer.stdin.close()
    writer.wait()
    reader.stdout.close()
    reader.wait()
    if b_reader is not None:
        b_reader.kill()
    print("wrote {} frames -> {}".format(written, out_path))
    print("widest name {:.0f} px (room {}), value {:.0f} px (room {}), note {:.0f} px (room {})".format(
        widest[0][0], COL_VAL - INFO_X - 10, widest[1][0], COL_NOW - COL_VAL - 75,
        widest[2][0], COL_TOT - INFO_X))


if __name__ == "__main__":
    main()
