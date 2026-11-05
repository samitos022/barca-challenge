"""Align the broadcast recording to the tactical one, with no manual step.

The two recordings share nothing but the match: different cameras, different cuts, replays only in
the broadcast, and no mapping table. Three independent signals are combined:

  1. Match clock (coarse).  Both videos show the broadcaster's clock. It is read with on-device OCR
     (Apple Vision, ocr_vision.swift) every STEP frames; its position on screen is found by the
     algorithm on a first pass over a few frames. Every pair of readings with the same clock value
     votes for an offset (tactical frame - broadcast frame); the vote peaks are the true offsets,
     one per stretch of continuous broadcast (normally one per half). Spurious pairs (45:xx shown
     in both halves, misreads) spread out and do not form a peak.
  2. Audio (fine).  Both videos carry the stadium sound. Band energies at 25 Hz are compared by
     normalised cross-correlation in windows of AUDIO_WIN_S around the clock offset, which pins it
     to the frame.
  3. Tracking (decision, frame by frame).  On broadcast frames with a usable homography, the
     broadcast players (pitch coordinates, same convention as the tactical ATD) are compared with
     the tactical players at each candidate offset: median distance of each broadcast player to
     the closest tactical player of the same team. The candidate under LIVE_COST metres wins and
     the frame is LIVE. A clock peak the tracking never confirms is dropped. Replays, close-ups and
     cuts fail this test and are not used.

Output, data/derived/<game>_broadcast_sync.npz:
    t_of_b  (n_broadcast,)  tactical frame shown by each broadcast frame at the stretch offset
                            (-1 outside the stretches)
    live    (n_broadcast,)  True where the tracking confirms it is live play at that offset
    b_of_t  (n_tactical,)   broadcast frame synchronised to each tactical frame (-1 if none)
    b_live  (n_tactical,)   b_of_t[t] where that broadcast frame is live, else -1
    segments                (first broadcast frame, last broadcast frame, offset) per stretch

    .venv/bin/python solution/sync_broadcast.py
"""

import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, GAMES, asset_paths, source_frames, video_path  # noqa: E402
import tracking  # noqa: E402

FPS = 25
DERIVED = os.path.join(ROOT, "data", "derived")
OCR_BIN = os.path.join(DERIVED, "ocr_vision")
OCR_SRC = os.path.join(HERE, "ocr_vision.swift")
OUT = os.path.join(DERIVED, "{}_broadcast_sync.npz".format(GAME))

P = {
    "PROBE_EVERY": 2500,    # frames between the probe images used to find the clock on screen
    "PROBE_N": 40,
    "STEP": 50,             # frames between clock readings (2 s)
    "CROP_W": 0.16,         # crop around the clock, fraction of the frame
    "CROP_H": 0.08,
    "VOTE_BIN": 25,         # frames per bin of the offset vote (the clock ticks once a second)
    "MIN_VOTES": 30,        # a stretch needs this many agreeing clock pairs
    "AUDIO_SR": 8000,
    "AUDIO_WIN_S": 30.0,
    "AUDIO_EVERY_S": 60.0,
    "AUDIO_SEARCH": 75,     # frames searched each side of the clock offset (3 s)
    "TRACK_EVERY": 5,       # broadcast frames between tracking checks
    "TRACK_MIN_PLAYERS": 4,
    "TRACK_SEARCH": 12,     # frames searched each side when refining the offset by tracking
    "LIVE_COST": 1.5,       # m, median distance under which a broadcast frame is live
    "NEAREST_S": 60.0,      # s: a non-live frame takes the offset of a live check this close
}
CLOCK = re.compile(r"^(\d{1,3})[:.](\d{2})$")


# ---- 1. match clock ---------------------------------------------------------------------------
def _ensure_ocr():
    if not os.path.exists(OCR_BIN):
        os.makedirs(DERIVED, exist_ok=True)
        subprocess.run(["swiftc", "-O", OCR_SRC, "-o", OCR_BIN], check=True)


def _extract(source, every, out_dir, vf_extra="", limit=None):
    """Every `every`-th frame of a video as PNG, named by its frame index."""
    os.makedirs(out_dir, exist_ok=True)
    done = sorted(f for f in os.listdir(out_dir) if f.endswith(".png"))
    if done:
        return
    vf = "select='not(mod(n\\,{}))'".format(every) + vf_extra
    cmd = ["ffmpeg", "-v", "error", "-y", "-hwaccel", "videotoolbox", "-i", video_path(GAME, source=source),
           "-vf", vf, "-fps_mode", "passthrough"]
    if limit:
        cmd += ["-frames:v", str(limit)]
    subprocess.run(cmd + [os.path.join(out_dir, "%06d.png")], check=True)
    # ffmpeg numbers outputs 1, 2, ...: rename to the frame index
    for name in sorted(os.listdir(out_dir)):
        if name.endswith(".png") and not name.startswith("f"):
            k = int(name[:-4]) - 1
            os.rename(os.path.join(out_dir, name), os.path.join(out_dir, "f{:06d}.png".format(k * every)))


def _ocr(out_dir):
    tsv = out_dir + ".tsv"
    if not os.path.exists(tsv):
        paths = sorted(os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".png"))
        with open(tsv, "w") as handle:
            subprocess.run([OCR_BIN], input="\n".join(paths), text=True, stdout=handle, check=True)
    rows = []
    for line in open(tsv):
        parts = line.rstrip("\n").split("\t")
        if len(parts) == 7:
            rows.append((parts[0], parts[1].strip(), *map(float, parts[2:])))
    return rows


def clock_readings(source):
    """[(frame, clock seconds)] for one recording."""
    _ensure_ocr()
    base = os.path.join(DERIVED, "ocr", source)
    probe = base + "_probe"
    _extract(source, P["PROBE_EVERY"], probe, limit=P["PROBE_N"])
    hits = [r for r in _ocr(probe) if CLOCK.match(r[1])]
    if not hits:
        raise RuntimeError("no clock found on the {} probe frames".format(source))
    # the clock sits where most clock-like texts are (box centre, normalised, origin bottom-left)
    cx = np.median([r[2] + r[4] / 2 for r in hits])
    cy = np.median([r[3] + r[5] / 2 for r in hits])
    w, h = P["CROP_W"], P["CROP_H"]
    x0, y0 = max(0.0, cx - w / 2), max(0.0, 1 - cy - h / 2)  # to top-left origin
    crop = ",crop=iw*{:.4f}:ih*{:.4f}:iw*{:.4f}:ih*{:.4f}".format(w, h, x0, y0)
    full = base + "_clock"
    _extract(source, P["STEP"], full, vf_extra=crop)
    out = []
    for path, text, *_ in _ocr(full):
        m = CLOCK.match(text)
        if m and int(m.group(2)) < 60:
            frame = int(os.path.basename(path)[1:-4])
            out.append((frame, int(m.group(1)) * 60 + int(m.group(2))))
    return out, (cx, cy)


def clock_offsets(tac, brd):
    """Offset peaks from equal-clock pairs: [(offset, votes)], and per broadcast reading its peak."""
    by_clock = {}
    for t, c in tac:
        by_clock.setdefault(c, []).append(t)
    d = np.array([t - b for b, c in brd for t in by_clock.get(c, [])])
    lo, hi = d.min(), d.max() + P["VOTE_BIN"]
    hist, edges = np.histogram(d, bins=np.arange(lo, hi + P["VOTE_BIN"], P["VOTE_BIN"]))
    # merge each peak with its neighbour bin (a clock second straddles two bins)
    peaks = []
    used = np.zeros(len(hist), bool)
    for k in np.argsort(hist)[::-1]:
        if used[k]:
            continue
        votes = hist[max(0, k - 1):k + 2].sum()
        if votes < P["MIN_VOTES"]:
            break
        used[max(0, k - 2):k + 3] = True
        sel = d[(d >= edges[max(0, k - 1)]) & (d < edges[min(len(hist), k + 2)])]
        peaks.append((int(np.median(sel)), int(votes)))
    labels = []
    for b, c in brd:
        cands = [t - b for t in by_clock.get(c, [])]
        best = None
        for i, (off, _) in enumerate(peaks):
            if any(abs(x - off) <= P["VOTE_BIN"] * 2 for x in cands):
                best = i
        labels.append((b, best))
    return peaks, labels


def stretches(peaks, labels, n_b):
    """Contiguous broadcast ranges per offset: each broadcast frame takes the peak of the nearest
    labelled readings (majority of the 15 nearest, so a stray reading cannot cut a stretch)."""
    lab = [(b, k) for b, k in labels if k is not None]
    bs = np.array([b for b, _ in lab])
    ks = np.array([k for _, k in lab])
    smooth = ks.copy()
    for i in range(len(ks)):
        win = ks[max(0, i - 7):i + 8]
        smooth[i] = np.bincount(win).argmax()
    segs = []
    start = 0
    for i in range(1, len(smooth) + 1):
        if i == len(smooth) or smooth[i] != smooth[start]:
            b0 = 0 if not segs else (bs[start - 1] + bs[start]) // 2 + 1
            b1 = n_b - 1 if i == len(smooth) else (bs[i - 1] + bs[i]) // 2
            segs.append([int(b0), int(b1), peaks[smooth[start]][0]])
            start = i
    return segs


# ---- 2. audio -----------------------------------------------------------------------------------
def audio_bands(source):
    """Per video frame, log energy in four bands, minus its own 5 s rolling median."""
    path = os.path.join(DERIVED, "{}_{}_audio_bands.npy".format(GAME, source))
    if os.path.exists(path):
        return np.load(path)
    sr = P["AUDIO_SR"]
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", video_path(GAME, source=source), "-vn", "-ac", "1",
                          "-ar", str(sr), "-f", "s16le", "-"], stdout=subprocess.PIPE, check=True).stdout
    x = np.frombuffer(raw, np.int16).astype(np.float32)
    hop = sr // FPS
    n = len(x) // hop
    frames = x[:n * hop].reshape(n, hop) * np.hanning(hop).astype(np.float32)
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(hop, 1 / sr)
    bands = [(100, 400), (400, 1200), (1200, 2500), (2500, 4000)]
    e = np.stack([np.log10(spec[:, (freqs >= a) & (freqs < b)].sum(axis=1) + 1e-3) for a, b in bands], 1)
    med = np.empty_like(e)
    for k in range(0, n, FPS):
        med[k:k + FPS] = np.median(e[max(0, k - 62):k + 63], axis=0)
    out = (e - med).astype(np.float32)
    np.save(path, out)
    return out


def refine_audio(seg, at, ab):
    """Best audio offset for a stretch: median of per-window cross-correlation peaks."""
    b0, b1, off = seg
    win, every, s = int(P["AUDIO_WIN_S"] * FPS), int(P["AUDIO_EVERY_S"] * FPS), P["AUDIO_SEARCH"]
    best, peaks = [], []
    for b in range(b0, b1 - win, every):
        x = ab[b:b + win]
        if len(x) < win:
            break
        x = (x - x.mean(0)) / (x.std(0) + 1e-6)
        scores = []
        for lag in range(-s, s + 1):
            t = b + off + lag
            if t < 0 or t + win > len(at):
                scores.append(-1)
                continue
            y = at[t:t + win]
            y = (y - y.mean(0)) / (y.std(0) + 1e-6)
            scores.append(float((x * y).mean()))
        scores = np.array(scores)
        k = int(np.argmax(scores))
        best.append(off + k - s)
        peaks.append(scores[k])
    best, peaks = np.array(best), np.array(peaks)
    good = peaks >= np.median(peaks)  # keep the clearer half of the windows
    return int(np.median(best[good])) if good.any() else off, best, peaks


# ---- 3. tracking ----------------------------------------------------------------------------------
def tracking_assign(candidates, tac, brd, valid_h):
    """Let the tracking pick, frame by frame, which candidate offset is right.

    1. Each candidate is refined on a subsample of checks (+/- TRACK_SEARCH frames); a candidate the
       tracking never confirms is dropped (a spurious clock peak dies here).
    2. Every TRACK_EVERY-th broadcast frame with a usable homography and enough players is checked
       against every surviving candidate (best of +/- 2 frames); the closest wins if it is under
       LIVE_COST metres.
    3. Frames between two checks confirmed with the same offset are live. Every other frame (replay,
       close-up) takes the offset of the nearest confirmed check, within NEAREST_S, so that what the
       broadcast is showing at that moment can still be displayed (flagged not live).
    """
    t_teams, t_pl = tac
    b_teams, b_pl = brd
    n_b, n_t = len(b_pl), len(t_pl)
    s = P["TRACK_SEARCH"]
    samples = [b for b in range(0, n_b, P["TRACK_EVERY"])
               if valid_h[b] and np.sum(~np.isnan(b_pl[b, :, 0])) >= P["TRACK_MIN_PLAYERS"]]

    def cost(b, t):
        return _cost(b_pl[b], b_teams, t_pl[t], t_teams) if 0 <= t < n_t else np.inf

    refined = []
    for off in candidates:
        shifts = []
        for b in samples[::20]:
            row = np.array([cost(b, b + off + dl) for dl in range(-s, s + 1)])
            if row.min() <= P["LIVE_COST"]:
                shifts.append(int(np.argmin(row)) - s)
        if shifts:
            refined.append((off + int(np.median(shifts)), len(shifts), off))
    offs = [r[0] for r in refined]

    label = np.full(n_b, -1, np.int64)
    cost_at = np.full(n_b, np.nan, np.float32)
    for b in samples:
        best, best_k = np.inf, -1
        for k, off in enumerate(offs):
            c = min(cost(b, b + off + dl) for dl in range(-2, 3))
            if c < best:
                best, best_k = c, k
        cost_at[b] = best
        if best <= P["LIVE_COST"]:
            label[b] = best_k

    live = np.zeros(n_b, bool)
    off_of_b = np.full(n_b, np.iinfo(np.int64).min, np.int64)
    conf = np.array([b for b in samples if label[b] >= 0])
    for a, b in zip(conf[:-1], conf[1:]):
        if b - a <= 2 * P["TRACK_EVERY"] and label[a] == label[b]:
            live[a:b + 1] = True
    live &= valid_h
    if len(conf):
        frames = np.arange(n_b)
        k = np.clip(np.searchsorted(conf, frames), 1, len(conf) - 1)
        left, right = conf[k - 1], conf[k]
        nearest = np.where(frames - left <= right - frames, left, right)
        near_ok = np.abs(nearest - frames) <= int(P["NEAREST_S"] * FPS)
        off_arr = np.array(offs, np.int64)
        off_of_b[near_ok] = off_arr[label[nearest[near_ok]]]
    return refined, live, off_of_b, cost_at


def _cost(bp, bt, tp, tt):
    """Median distance (m) from each broadcast player to the closest tactical player of his team."""
    out = []
    for team in (0, 1):
        a = bp[(bt == team) & ~np.isnan(bp[:, 0])]
        b = tp[(tt == team) & ~np.isnan(tp[:, 0])]
        if len(a) and len(b):
            out.append(np.min(np.linalg.norm(a[:, None] - b[None], axis=2), axis=1))
    if not out:
        return np.inf
    return float(np.median(np.concatenate(out)))


def main():
    n_t, n_b = source_frames("tactical"), source_frames("broadcast")
    print("1. clock: OCR of both recordings (on device) ...", flush=True)
    tac_clock, tpos = clock_readings("tactical")
    brd_clock, bpos = clock_readings("broadcast")
    print("   clock found at ({:.2f}, {:.2f}) tactical, ({:.2f}, {:.2f}) broadcast (fraction of frame)".format(
        tpos[0], 1 - tpos[1], bpos[0], 1 - bpos[1]))
    print("   readings: tactical {} of {}, broadcast {} of {}".format(
        len(tac_clock), n_t // P["STEP"], len(brd_clock), n_b // P["STEP"]))
    peaks, labels = clock_offsets(tac_clock, brd_clock)
    for off, votes in peaks:
        print("   offset peak {:+d} frames ({:+.1f} s), {} votes".format(off, off / FPS, votes))
    segs = stretches(peaks, labels, n_b)

    print("2. audio ...", flush=True)
    at, ab = audio_bands("tactical"), audio_bands("broadcast")
    for seg in segs:
        old = seg[2]
        seg[2], best, pk = refine_audio(seg, at, ab)
        agree = np.mean(np.abs(best - seg[2]) <= 2) if len(best) else 0
        print("   stretch b {}..{}: clock {:+d} -> audio {:+d} frames ({} windows, {:.0f}% within 2 frames, "
              "median peak {:.2f})".format(seg[0], seg[1], old, seg[2], len(best), 100 * agree,
                                           np.median(pk) if len(pk) else 0))

    print("3. tracking ...", flush=True)
    t_teams, t_pl, *_ = tracking.load(GAME)
    b_teams, b_pl, *_ = tracking.load(GAME, "broadcast")
    hcsv = asset_paths(GAME, source="broadcast")["homography"]
    valid_h = np.loadtxt(hcsv, delimiter=",", skiprows=1, usecols=(11,)) == 1
    candidates = sorted(set(seg[2] for seg in segs))
    refined, live, off_of_b, cost_at = tracking_assign(candidates, (t_teams, t_pl), (b_teams, b_pl), valid_h)
    dropped = set(candidates) - set(r[2] for r in refined)
    for off, n_conf, old in refined:
        print("   offset {:+d} -> tracking {:+d} frames ({:+.2f} s): confirmed on {} of the refining checks, "
              "live on {:.1f}% of broadcast frames".format(old, off, off / FPS, n_conf,
                                                           100 * np.mean(live & (off_of_b == off))))
    for off in sorted(dropped):
        print("   offset {:+d}: never confirmed by the tracking, dropped".format(off))
    checked = np.isfinite(cost_at)
    print("   tracking distance on checked frames: median {:.2f} m (live) vs {:.2f} m (not live)".format(
        np.median(cost_at[checked & live]) if (checked & live).any() else np.nan,
        np.median(cost_at[checked & ~live]) if (checked & ~live).any() else np.nan))

    has = off_of_b != np.iinfo(np.int64).min
    t_of_b = np.full(n_b, -1, np.int64)
    t_of_b[has] = np.arange(n_b)[has] + off_of_b[has]
    t_of_b[(t_of_b < 0) | (t_of_b >= n_t)] = -1
    live &= t_of_b >= 0
    b_of_t = np.full(n_t, -1, np.int64)
    b_live = np.full(n_t, -1, np.int64)
    ok = t_of_b >= 0
    b_of_t[t_of_b[ok]] = np.flatnonzero(ok)
    b_live[t_of_b[live]] = np.flatnonzero(live)
    # stretches of constant offset, for the record
    segs_out, start = [], None
    for b in range(n_b + 1):
        cur = off_of_b[b] if b < n_b else None
        if start is None or b == n_b or cur != off_of_b[start]:
            if start is not None and off_of_b[start] != np.iinfo(np.int64).min:
                segs_out.append((start, b - 1, int(off_of_b[start])))
            start = b
    np.savez(OUT, t_of_b=t_of_b, live=live, b_of_t=b_of_t, b_live=b_live,
             segments=np.array(segs_out, np.int64).reshape(-1, 3), cost=cost_at)
    print("   {} stretches of constant offset".format(len(segs_out)))
    print("broadcast frames live: {:.1f}%; tactical frames with a live broadcast frame: {:.1f}%".format(
        100 * live.mean(), 100 * (b_live >= 0).mean()))
    print("-> {}".format(os.path.relpath(OUT, ROOT)))


if __name__ == "__main__":
    main()
