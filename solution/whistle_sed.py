"""The referee's whistle heard by a pretrained sound-event detector (separate environment).

whistle.py finds whistle-like TONES, and on this match almost all of them are the crowd (fans
whistling): after one of its events the players brake no more often than at a random moment. A
general sound-event model trained on AudioSet tells apart two classes that matter here: "Whistle"
(the device: a referee's whistle) and "Whistling" (a person whistling). This script runs PANNs
(Kong et al. 2020, CNN14 with frame-level output, public weights, MIT licence) over a video's
sound track and stores, per video frame (25 fps), the probability of each of the two classes.

Needs the separate environment (.venv-cv: torch, panns-inference) and the weights in
models/panns/panns_data/ (class_labels_indices.csv, Cnn14_DecisionLevelMax.pth, from
https://zenodo.org/record/3987831). Output: data/derived/<game>_<source>_sed.npz (a derivative of
the match audio: never published).

    HOME=models/panns .venv-cv/bin/python solution/whistle_sed.py [tactical|broadcast]
"""

import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, source_frames, video_path  # noqa: E402

SR = 32000
FPS = 25
CHUNK_S, EDGE_S = 30, 2      # windows of 30 s, the first and last 2 s of each are not used
CLASSES = {"whistle": 402, "whistling": 40}   # AudioSet indices: the device, the person
WEIGHTS = os.path.join(ROOT, "models", "panns", "panns_data", "Cnn14_DecisionLevelMax.pth")


def out_path(source, game=GAME):
    return os.path.join(ROOT, "data", "derived", "{}_{}_sed.npz".format(game, source))


def decode(source, game=GAME):
    path = video_path(game) if source == "tactical" else video_path(game, source=source)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
                         stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def run(source):
    import torch
    from panns_inference import SoundEventDetection
    sed = SoundEventDetection(checkpoint_path=WEIGHTS, device="cpu")
    model = sed.model
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    try:
        model.to(dev)
        with torch.no_grad():
            model(torch.zeros(1, SR * 4, device=dev), None)
    except Exception:
        dev = "cpu"
        model.to(dev)
    model.eval()
    audio = decode(source)
    n = source_frames(source, GAME)
    out = {k: np.zeros(n, dtype=np.float32) for k in CLASSES}
    step = (CHUNK_S - 2 * EDGE_S) * SR
    starts = np.arange(0, len(audio), step)
    for i, s0 in enumerate(starts):
        a = max(0, s0 - EDGE_S * SR)
        x = audio[a:s0 + step + EDGE_S * SR]
        if len(x) < SR:
            break
        with torch.no_grad():
            fw = model(torch.from_numpy(x[None]).to(dev), None)["framewise_output"][0].float().cpu().numpy()
        hop = len(x) / len(fw)                       # samples per output frame (10 ms)
        t = a + (np.arange(len(fw)) + 0.5) * hop     # sample position of each output frame
        keep = (t >= s0) & (t < s0 + step)
        vf = np.clip((t[keep] / SR * FPS).astype(np.int64), 0, n - 1)
        for k, c in CLASSES.items():
            np.maximum.at(out[k], vf, fw[keep, c])
        if i % 40 == 0:
            print("{} / {} windows ({})".format(i, len(starts), dev), flush=True)
    np.savez_compressed(out_path(source), **out)
    print("{}: whistle (device) over 0.3 on {:.2f}% of the frames, whistling (person) on {:.2f}% -> {}".format(
        source, 100 * np.mean(out["whistle"] > 0.3), 100 * np.mean(out["whistling"] > 0.3),
        os.path.relpath(out_path(source), ROOT)))


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "tactical")
