"""Whistle-like tones on the tactical video's own audio track (same timeline as the video).

A referee whistle is a loud, almost pure tone between about 2.5 and 4.5 kHz that lasts a few tenths
of a second and keeps its pitch. The detector:

  - per video frame (40 ms) takes the spectrum of a 64 ms window, 2-4.5 kHz band;
  - subtracts from every frequency bin its own rolling median over 10 s, so steady tones (hum, the
    PA, a camera fan) vanish and only sounds that start and stop remain ("transient", dB);
  - keeps frames where one bin stands TONE_DB over its background and NARROW_DB over the band's
    median transient (a narrow tone, not a crowd roar that lifts every bin);
  - an event is a run of at least MIN_FRAMES such frames whose pitch moves less than STEP_HZ from
    one frame to the next.

Measured limit: the tactical camera hears the crowd far better than the referee, and fans
whistling and chanting make the same kind of tone. On this match the detector fires about 12
times a minute during play and 7 a minute outside it, so an event is weak evidence on its own and
the state machine only uses it when the players actually stop afterwards.

Cached in data/derived/ (a derivative of the match audio: never published).
"""

import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, half_video_blocks, source_frames, video_path  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "data", "derived")
SR = 16000
FPS = 25
HOP = SR // FPS          # 640 samples = one video frame
WIN = 1024               # 64 ms window centred on the frame

PARAMS = {
    "F_LO": 2000.0,      # Hz, band stored
    "F_MIN": 2500.0,     # Hz, lowest pitch accepted as a whistle
    "F_HI": 4500.0,      # Hz
    "BG_S": 10.0,        # s, rolling window of each bin's background
    "TONE_DB": 15.0,     # dB, peak bin over its own background
    "NARROW_DB": 12.0,   # dB, peak bin over the band's median transient
    "STEP_HZ": 80.0,     # Hz, a whistle keeps its pitch from frame to frame
    "MIN_FRAMES": 5,     # 0.2 s: shorter tones are not a whistle blast
}


def _decode(game, source="tactical"):
    path = video_path(game) if source == "tactical" else video_path(game, source=source)
    if not os.path.exists(path):
        raise FileNotFoundError("{} missing: the whistle needs the {} video's audio "
                                "(or run possession_v2.py --no-whistle)".format(path, source))
    cmd =["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(SR),
           "-f", "s16le", "-"]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16)


def features(audio, n_frames):
    """Per video frame: band spectrum in dB (float16), and the band's frequencies."""
    freqs = np.fft.rfftfreq(WIN, 1.0 / SR)
    band = (freqs >= PARAMS["F_LO"]) & (freqs <= PARAMS["F_HI"])
    window = np.hanning(WIN).astype(np.float32)
    # the audio track is a few hundredths of a second shorter than the video: pad with silence
    tail = max(WIN, n_frames * HOP + 2 * WIN - len(audio))
    pad = np.concatenate((np.zeros(WIN // 2, np.int16), audio, np.zeros(tail, np.int16)))
    spec_db = np.zeros((n_frames, int(band.sum())), np.float16)
    chunk = 4096
    for a in range(0, n_frames, chunk):
        idx = np.arange(a, min(n_frames, a + chunk))
        starts = idx * HOP + HOP // 2  # window centred on the middle of the video frame
        frames = pad[starts[:, None] + np.arange(WIN)[None, :]].astype(np.float32) * window
        spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
        spec_db[idx] = 10 * np.log10(spec[:, band] + 1e-6)
    return spec_db, freqs[band]


def transient(spec_db, p=PARAMS):
    """Each bin minus its rolling median over BG_S seconds (sampled every second)."""
    x = spec_db.astype(np.float32)
    n = len(x)
    half = int(p["BG_S"] * FPS) // 2
    centres = np.arange(0, n, FPS)
    bg = np.empty((len(centres), x.shape[1]), np.float32)
    for k, c in enumerate(centres):
        bg[k] = np.median(x[max(0, c - half):c + half + 1:5], axis=0)
    rows = np.minimum(np.round(np.arange(n) / FPS).astype(np.int64), len(centres) - 1)
    return x - bg[rows]


def detect(spec_db, bf, p=PARAMS):
    """events [(first, last, pitch Hz, peak dB)], per-frame peak dB of the tone (0 outside events)."""
    trans = transient(spec_db, p)
    keep = (bf >= p["F_MIN"]) & (bf <= p["F_HI"])
    sub, fsub = trans[:, keep], bf[keep]
    peak = sub.max(axis=1)
    pitch = fsub[np.argmax(sub, axis=1)]
    ok = (peak >= p["TONE_DB"]) & (peak - np.median(trans, axis=1) >= p["NARROW_DB"])
    n = len(ok)
    events = []
    level = np.zeros(n, np.float32)
    i = 0
    idx = np.flatnonzero(ok)
    pos = 0
    while pos < len(idx):
        i = idx[pos]
        j = i
        while j + 1 < n and ok[j + 1] and abs(pitch[j + 1] - pitch[j]) <= p["STEP_HZ"]:
            j += 1
        if j - i + 1 >= p["MIN_FRAMES"]:
            events.append((int(i), int(j), float(np.median(pitch[i:j + 1])), float(peak[i:j + 1].max())))
            level[i:j + 1] = peak[i:j + 1]
        pos = np.searchsorted(idx, j + 1)
    return events, level


def load(game=GAME):
    """(events, level): cached spectrum, detection re-run on every call."""
    n = source_frames("tactical", game)
    path = os.path.join(CACHE_DIR, "{}_tactical_audio_features.npz".format(game))
    if os.path.exists(path):
        z = np.load(path)
        spec_db, bf = z["spec_db"], z["bf"]
    else:
        print("decoding the tactical audio once (cached in data/derived/) ...", flush=True)
        spec_db, bf = features(_decode(game), n)
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(path, spec_db=spec_db, bf=bf)
    return detect(spec_db, bf)


# ---- the same tone in the broadcast audio --------------------------------------------------------
# The tactical camera hears the crowd better than the referee, and fans whistle too: 12 events a
# minute during play. The broadcast has its own microphones, by the pitch: the referee's whistle is
# in both sound tracks at the same moment and pitch, a fan whistling next to the tactical camera
# is only in one. second_microphone() gives, for every tactical event, how loud a tone of the same
# pitch is in the broadcast audio at that moment (dB over its background, like TONE_DB).
PITCH_TOL = 150.0     # Hz between the two recordings of the same blast
TIME_TOL = 10         # frames (0.4 s): sound reaches the far tactical camera up to 0.3 s late


def broadcast_transient(game=GAME):
    """(transient dB per broadcast frame and frequency bin, bin frequencies), cached."""
    path = os.path.join(CACHE_DIR, "{}_broadcast_audio_features.npz".format(game))
    if os.path.exists(path):
        z = np.load(path)
        spec_db, bf = z["spec_db"], z["bf"]
    else:
        print("decoding the broadcast audio once (cached in data/derived/) ...", flush=True)
        spec_db, bf = features(_decode(game, "broadcast"), source_frames("broadcast", game))
        np.savez(path, spec_db=spec_db, bf=bf)
    return transient(spec_db), bf


def second_microphone(events, game=GAME):
    """(k,) dB of the broadcast audio at each tactical event's pitch and time (NaN where the
    broadcast does not cover that moment). Uses the stretch offsets of sync_broadcast."""
    import sync_broadcast
    segs = np.load(sync_broadcast.OUT)["segments"]
    trans, bf = broadcast_transient(game)
    out = np.full(len(events), np.nan, dtype=np.float32)
    for i, (a, b, pitch, _) in enumerate(events):
        for b0, b1, off in segs:
            lo, hi = a - off - TIME_TOL, b - off + TIME_TOL
            if hi < b0 or lo > b1:
                continue
            lo, hi = max(lo, b0, 0), min(hi, b1, len(trans) - 1)
            cols = np.abs(bf - pitch) <= PITCH_TOL
            if hi >= lo and cols.any():
                out[i] = float(trans[lo:hi + 1][:, cols].max())
    return out


if __name__ == "__main__":
    events, level = load()
    blocks = half_video_blocks(GAME)
    n = source_frames("tactical", GAME)
    graded = np.zeros(n, dtype=bool)
    for _, a, z in blocks:
        graded[a:z + 1] = True
    inside = np.array([graded[s] for s, *_ in events])
    print("whistle-like events: {}  ({:.1f}/min in play, {:.1f}/min outside the halves)".format(
        len(events), inside.sum() / graded.sum() * FPS * 60,
        (~inside).sum() / (~graded).sum() * FPS * 60))
    print("duration median {:.2f} s".format(np.median([(e - s + 1) / FPS for s, e, *_ in events])))
