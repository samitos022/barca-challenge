"""PFF FC (now Gradient Sports) FIFA World Cup 2022: broadcast tracking of the 64 matches in the
shape idsse.load() returns, plus H/A/D labels from PFF's game events. A fourth calibration set
next to Metrica, IDSSE and SkillCorner (labelled data from OTHER games).

Source: the Google Drive folder that kloppy's documentation lists as PFF's public data
(https://kloppy.pysport.org/user-guide/loading-data/ , "Public Data" column, PFF FC row):
https://drive.google.com/drive/folders/1_a_q1e9CXeEPJ3GdCv_3-rNO3gPqacfa . PFF announced the
dataset as free to use (blog post "PFF FC's 2022 World Cup Dataset Now Available"); no licence
file is in the folder. Files used, in data/external/pff/ (git-ignored):
    Tracking Data/<id>.jsonl.bz2   one JSON line per frame, 29.97 fps
    Metadata/<id>.json             teams, pitch size, fps
(Event Data/ and Rosters/ are not needed: every tracking frame carries its game event.)

What this tracking is:
  - made from the BROADCAST video. Every frame gives all the players, but only the ones in the
    picture are measured (visibility VISIBLE): the others are ESTIMATED by PFF. About 9 of 22 are
    really seen in play and about 2 of 22 during stoppages (close-ups, replays).
  - the ball comes twice: "ballsSmoothed" (filtered) and "balls" (as detected). The filtered one
    has NO coordinates while play is stopped (0.2% of the dead frames), the detected one has them
    on about a third of the dead frames (9 in 10 during the first second of a stoppage, 3 in 10
    after 6 s) and there the ball is really followed while it is carried or kicked to the restart
    spot (faster than 2 m/s on more than half of those frames). So the ball here is the filtered
    one where it has coordinates and the detected one elsewhere. Reading only the filtered ball
    (the first version of this loader did) makes every stoppage look ball-less. Same flag as the
    players: VISIBLE (seen) or ESTIMATED.
  - rows here are 25 fps: row r of a period is its elapsed time r / 25 s, positions linearly
    interpolated between the two nearest source frames, never across more than MAX_GAP_S.

Labels (an algorithm applied to PFF's human-annotated game events):
  - dead from every OUT event (ball out of play: also fouls and goals) and from END; alive from a
    kick-off and from every OTB ("on the ball") event, so a stoppage ends with the first on-ball
    event after it, which is the restart being taken (measured: the ball is moving 0.5 s later in
    two cases out of three). Event times are annotated on video frames: about +-0.5 s.
  - team: the team of the OTB events (home_ball). As in idsse.py and skillcorner.py, an on-ball
    spell that carries a pass, a shot or a cross, or lasts FIRM_S or more, gives the ball to its
    team; a shorter one only proposes a change, confirmed and dated to it if that team has the next
    firm spell before the other one does. Each alive run starts with its first spell's team.
  - graded rows: from the kick-off to the last in-play row of each period (X elsewhere).

    .venv/bin/python solution/external/pff.py --download   # Tracking Data + Metadata (about 2.8 GB), resumable
    .venv/bin/python solution/external/pff.py              # parse what is on disk, print the aggregates

Google Drive stops serving files after about 130 downloads in a row ("Failed to retrieve file
url"): run --download again some hours later (it skips what is already there), or download the
"Tracking Data" and "Metadata" folders from the browser and unzip them into data/external/pff/.
"""

import bz2
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DIR = os.path.join(ROOT, "data", "external", "pff")
FPS = 25
MAX_GAP_S = 0.5          # never interpolate a position across a hole longer than this
FIRM_S = 1.0             # an on-ball spell this long is controlled whatever happens in it
FIRM_EVENTS = ("PA", "SH", "CR")   # pass, shot, cross: the player was in control
HOME, AWAY = 0, 1
SLOTS = 30               # shirt numbers kept per team (26-man squads)
VIS = {"VISIBLE": 1, "ESTIMATED": 2}


FOLDER = "1_a_q1e9CXeEPJ3GdCv_3-rNO3gPqacfa"     # the Drive folder listed in kloppy's documentation


def download(max_failures=2):
    """Tracking Data/ and Metadata/ of the public folder into DIR (needs the gdown package)."""
    import time
    import gdown
    import requests
    walker = sys.modules["gdown.download_folder"]
    root = walker._download_and_parse_google_drive_link(sess=requests.session(), folder_id=FOLDER, quiet=True,
                                                         verify=True, timeout=60)
    want = []
    for folder in root.children:
        if folder.is_folder() and folder.name in ("Tracking Data", "Metadata"):
            want += [(folder.name, f.name, f.id) for f in folder.children if not f.is_folder() and "google-apps" not in f.type]
    want.sort(key=lambda x: (x[0] != "Tracking Data", x[1]))   # the big files first: they are the ones that matter
    done = failed = 0
    for folder, name, fid in want:
        out = os.path.join(DIR, folder, name)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            done += 1
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        try:
            gdown.download(id=fid, output=out, quiet=True)
            done += 1
        except Exception as e:
            failed += 1
            print("failed: {} ({})".format(name, str(e).strip().splitlines()[0][:80]), flush=True)
            if failed >= max_failures:
                print("Drive is refusing downloads for now: run again later (it resumes)")
                break
            time.sleep(20)
    print("{} of {} files on disk".format(done, len(want)))


def _games():
    return tuple(sorted(os.path.basename(p).split(".")[0] for p in glob.glob(os.path.join(DIR, "Tracking Data", "*.jsonl.bz2"))
                        if os.path.exists(os.path.join(DIR, "Metadata", os.path.basename(p).split(".")[0] + ".json"))))


GAMES = _games()


def match_info(game):
    with open(os.path.join(DIR, "Metadata", "{}.json".format(game))) as h:
        m = json.load(h)
    m = m[0] if isinstance(m, list) else m
    pitch = (m.get("stadium") or {}).get("pitches") or [{}]
    return float(pitch[0].get("length") or 105.0), float(pitch[0].get("width") or 68.0), float(m.get("fps") or 29.97)


def pitch(game):
    return match_info(game)[:2]


def _resample(t_src, v, t_dst):
    """v (n, k) at times t_src (increasing) with NaN holes -> values at t_dst: linear between the
    two nearest tracked source frames when they are at most MAX_GAP_S apart, NaN elsewhere."""
    out = np.full((len(t_dst), v.shape[1]), np.nan, dtype=np.float32)
    ok = np.flatnonzero(~np.isnan(v[:, 0]))
    if len(ok) < 2:
        return out
    ts = t_src[ok]
    j = np.searchsorted(ts, t_dst, side="right")
    lo, hi = np.clip(j - 1, 0, len(ok) - 1), np.clip(j, 0, len(ok) - 1)
    span = ts[hi] - ts[lo]
    inside = (j > 0) & (j < len(ok)) & (span <= MAX_GAP_S) & (span > 0)
    w = ((t_dst[inside] - ts[lo[inside]]) / span[inside]).astype(np.float32)[:, None]
    out[inside] = (1 - w) * v[ok[lo[inside]]] + w * v[ok[hi[inside]]]
    return out


def _build(game):
    L, W, _ = match_info(game)
    per, tm, ev = [], [], []          # per frame; ev: (frame index, type, home_ball, firm)
    P = [[], []]                      # per team: (frame index, shirt, x, y, visible)
    B = []                            # (frame index, x, y, visible)
    with bz2.open(os.path.join(DIR, "Tracking Data", "{}.jsonl.bz2".format(game)), "rt") as h:
        for i, line in enumerate(h):
            d = json.loads(line)
            per.append(d.get("period") or 0)
            tm.append(d.get("periodElapsedTime") if d.get("periodElapsedTime") is not None else np.nan)
            ge = d.get("game_event")
            pe = d.get("possession_event")
            if ge and ge.get("game_event_type"):
                ev.append((i, ge["game_event_type"], ge.get("home_ball"),
                           bool(pe and pe.get("possession_event_type") in FIRM_EVENTS), ge.get("start_frame")))
            for k, key in ((HOME, "homePlayersSmoothed"), (AWAY, "awayPlayersSmoothed")):
                got = set()
                for src in (key, key.replace("Smoothed", "")):   # filtered first, the detected one for who is missing
                    for p in d.get(src) or []:
                        if p.get("x") is None or p.get("jerseyNum") is None or p["jerseyNum"] in got:
                            continue
                        got.add(p["jerseyNum"])
                        P[k].append((i, int(p["jerseyNum"]), p["x"], p["y"], VIS.get(p.get("visibility"), 2)))
            b = d.get("ballsSmoothed")
            if not b or b.get("x") is None:          # no filtered ball (always so in a stoppage): the detected one
                b = (d.get("balls") or [None])[0]
            if b and b.get("x") is not None:
                B.append((i, b["x"], b["y"], VIS.get(b.get("visibility"), 2)))
    per, tm = np.array(per), np.array(tm, dtype=np.float64)
    n_src = len(per)
    periods = [p for p in np.unique(per) if p in (1, 2)]      # extra time is not used
    # rows: each period on its own 25 fps grid, one after the other
    grids, offs, row0 = {}, {}, 0
    for p in periods:
        m = (per == p) & ~np.isnan(tm)
        t0, t1 = tm[m].min(), tm[m].max()
        grids[p] = (t0, np.arange(int(np.floor((t1 - t0) * FPS)) + 1) / FPS + t0)
        offs[p] = row0
        row0 += len(grids[p][1])
    n = row0
    row_of = np.full(n_src, -1, dtype=np.int64)
    for p in periods:
        m = (per == p) & ~np.isnan(tm)
        row_of[m] = offs[p] + np.clip(np.round((tm[m] - grids[p][0]) * FPS).astype(np.int64), 0, len(grids[p][1]) - 1)
    shirts = [sorted({s for _, s, *_ in P[k]})[:SLOTS] for k in (HOME, AWAY)]
    slot = [{s: j for j, s in enumerate(shirts[k])} for k in (HOME, AWAY)]
    S = len(shirts[0]) + len(shirts[1])
    players = np.full((n, S, 2), np.nan, dtype=np.float32)
    pdet = np.zeros((n, S), dtype=np.uint8)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    bdet = np.zeros(n, dtype=np.uint8)
    for k in (HOME, AWAY):
        if not P[k]:
            continue
        A = np.array(P[k], dtype=np.float64)
        for s, j in slot[k].items():
            a = A[A[:, 1] == s]
            col = j + (len(shirts[0]) if k == AWAY else 0)
            for p in periods:
                fi = a[:, 0].astype(np.int64)
                m = per[fi] == p
                if m.sum() < 2:
                    continue
                v = np.stack([a[m, 2] + L / 2, a[m, 3] + W / 2], axis=1)
                r0, t_dst = offs[p], grids[p][1]
                players[r0:r0 + len(t_dst), col] = _resample(tm[fi[m]], v, t_dst)
                vis = _resample(tm[fi[m]], (a[m, 4:5] == 1).astype(np.float64), t_dst)[:, 0]
                has = ~np.isnan(players[r0:r0 + len(t_dst), col, 0])
                pdet[r0:r0 + len(t_dst), col] = np.where(has, np.where(np.nan_to_num(vis) >= 0.5, 1, 2), 0)
    if B:
        A = np.array(B, dtype=np.float64)
        fi = A[:, 0].astype(np.int64)
        for p in periods:
            m = per[fi] == p
            if m.sum() < 2:
                continue
            r0, t_dst = offs[p], grids[p][1]
            ball[r0:r0 + len(t_dst)] = _resample(tm[fi[m]], np.stack([A[m, 1] + L / 2, A[m, 2] + W / 2], axis=1), t_dst)
            vis = _resample(tm[fi[m]], (A[m, 3:4] == 1).astype(np.float64), t_dst)[:, 0]
            has = ~np.isnan(ball[r0:r0 + len(t_dst), 0])
            bdet[r0:r0 + len(t_dst)] = np.where(has, np.where(np.nan_to_num(vis) >= 0.5, 1, 2), 0)
    # events -> on-ball spells and out / end / kick-off rows
    spells, marks = [], []      # spells: [first row, last row, team, firm]; marks: (row, kind)
    cur = None
    for i, typ, hb, firm, sf in ev:
        r = row_of[i]
        if r < 0:
            continue
        if typ == "OTB":
            if cur is not None and cur[4] == sf and cur[2] == (HOME if hb else AWAY):
                cur[1] = r
                cur[3] = cur[3] or firm
            else:
                cur = [r, r, HOME if hb else AWAY, firm, sf]
                spells.append(cur)
        elif typ in ("OUT", "END"):
            marks.append((r, 0))
            cur = None
        elif typ.endswith("KICKOFF"):
            marks.append((r, 1))
            cur = None
    teams = np.array([HOME] * len(shirts[0]) + [AWAY] * len(shirts[1]), dtype=np.int8)
    return dict(teams=teams, players=players, ball=ball, pdet=pdet, bdet=bdet,
                spells=np.array([s[:4] for s in spells], dtype=np.int64).reshape(-1, 4),
                marks=np.array(marks, dtype=np.int64).reshape(-1, 2),
                period_rows=np.array([[p, offs[p], offs[p] + len(grids[p][1]) - 1] for p in periods], dtype=np.int64))


def _cache(game):
    return os.path.join(DIR, "cache", "{}.npz".format(game))


def build_cache(game):
    if not os.path.exists(_cache(game)):
        os.makedirs(os.path.dirname(_cache(game)), exist_ok=True)
        np.savez_compressed(_cache(game), **_build(game))
    return game


def raw(game):
    build_cache(game)
    return np.load(_cache(game))


def load(game, estimated=True):
    """(teams, players, ball, ball_speed, periods). estimated=False blanks the positions PFF did
    not measure (players out of the picture, ball not seen)."""
    z = raw(game)
    players, ball = z["players"], z["ball"]
    if not estimated:
        players = np.where((z["pdet"] == 1)[:, :, None], players, np.nan).astype(np.float32)
        ball = np.where((z["bdet"] == 1)[:, None], ball, np.nan).astype(np.float32)
    return z["teams"], players, ball, np.full(len(ball), np.nan, dtype=np.float32), z["period_rows"]


def labels(game):
    z = raw(game)
    n = len(z["ball"])
    lab = np.full(n, "X", dtype="<U1")
    spells, marks = z["spells"], z["marks"]
    for _, a, b in z["period_rows"]:
        # change points inside the period: alive from a kick-off or an on-ball spell, dead from OUT / END
        pts = [(int(r), 1) for r, k in marks if a <= r <= b and k == 1] + \
              [(int(r), 0) for r, k in marks if a <= r <= b and k == 0] + \
              [(int(s[0]), 1) for s in spells if a <= s[0] <= b]
        pts.sort()
        if not pts:
            continue
        alive = np.zeros(b - a + 1, dtype=bool)
        state, last = None, None
        for r, k in pts:
            if state is not None and state:
                alive[last - a:r - a] = True
            state, last = (k == 1), r
        # after the last change point the period is over (END, or the half-time whistle after an OUT)
        first = next((r for r, k in pts if k == 1), None)
        if first is None:
            continue
        idx = np.flatnonzero(alive)
        if not len(idx):
            continue
        lab[first:a + idx[-1] + 1] = "D"
        e = np.flatnonzero(np.diff(np.concatenate(([0], alive.astype(np.int8), [0]))))
        mine = [s for s in spells if a <= s[0] <= b]
        starts = np.array([s[0] for s in mine]) if mine else np.zeros(0, dtype=np.int64)
        cur = HOME
        for s0, e0 in zip(a + e[::2], a + e[1::2]):
            run = [s for s in mine[np.searchsorted(starts, s0):np.searchsorted(starts, e0)]]
            if run:
                cur = int(run[0][2])
            t0, pending = s0, None
            for st, en, team, firm in run:
                firm = bool(firm) or (en - st) >= FIRM_S * FPS
                if team == cur:
                    pending = None
                    continue
                if not firm:
                    if pending is None or pending[0] != team:
                        pending = (team, st)
                    continue
                at = pending[1] if pending is not None and pending[0] == team else st
                at = min(max(int(at), s0), e0)
                lab[t0:at] = "HA"[cur]
                cur, t0, pending = int(team), at, None
            lab[t0:e0] = "HA"[cur]
    return lab


def blocks_from_labels(lab):
    """Graded blocks [(k, first, last)]: the non-X runs, one per period."""
    idx = np.flatnonzero(np.diff(np.concatenate(([0], (lab != "X").astype(np.int8), [0]))))
    return [(k + 1, int(a), int(b) - 1) for k, (a, b) in enumerate(zip(idx[::2], idx[1::2]))]


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    import sanity
    if "--download" in sys.argv:
        download()
        GAMES = _games()
    with Pool(min(6, os.cpu_count())) as pool:
        pool.map(build_cache, GAMES)
    rows, seen_rows = [], []
    for g in GAMES:
        z = raw(g)
        lab = labels(g)
        s = sanity.aggregates(lab, z["players"], z["ball"])
        seen = sanity.aggregates(lab, np.where((z["pdet"] == 1)[:, :, None], z["players"], np.nan),
                                 np.where((z["bdet"] == 1)[:, None], z["ball"], np.nan))
        rows.append(s)
        seen_rows.append(seen)
        print(sanity.line(g, s), flush=True)
    if rows:
        print(sanity.line("mean of {}".format(len(rows)), sanity.mean(rows)))
        m = sanity.mean(seen_rows)
        print("really seen by the camera: ball {:.1f}% of graded, {:.1f}% of dead (moving {:.1f}%), players {:.1f} per frame".format(
            m["ball_pct"], m["dead_ball_pct"], m["dead_ball_moving_pct"], m["players"]))
