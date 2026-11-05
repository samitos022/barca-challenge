"""Tactical ATD as numpy arrays in metres, parsed once and cached.

Parsing the raw text takes about a minute; every later run loads the cache in a second. The cache is
a derivative of the match data, so it lives in data/derived/ (ignored by git) and is never published.

    players     (N, slots, 2) float32   x, y in metres, NaN where the slot is not tracked
    ball        (N, 2)        float32   x, y in metres, NaN where the ball is not tracked
    ball_speed  (N,)          float32   speed reported by the ATD, m/s, NaN if missing
    teams       (slots,)      int8      0 home, 1 away

Row i is video frame i (ATD frame_id - 1) of that recording. For the tactical recording that is
the same index as a submission row; the broadcast has its own timeline (see sync_broadcast.py).

source="fused" loads the tactical timeline with the broadcast merged in (fusion.py), with extra
player slots for players only the broadcast sees.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, require, source_frames  # noqa: E402
from baseline import team_of_each_track, team_names  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "data", "derived")


def _parse(game, source="tactical"):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, source=source)
    teams = team_of_each_track(meta_path)
    length, width = GAMES[game]["pitch"]
    n = source_frames(source, game)
    players = np.full((n, len(teams), 2), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    ball_speed = np.full(n, np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1
            for slot, group in enumerate(parts[1].split(";")[:len(teams)]):
                if group[0] == "N":
                    continue
                vals = group.split(",")
                players[i, slot, 0] = float(vals[0]) * length
                players[i, slot, 1] = float(vals[1]) * width
            vals = parts[2].strip().split(",")
            if vals[0][0] != "N":
                ball[i, 0] = float(vals[0]) * length
                ball[i, 1] = float(vals[1]) * width
                if len(vals) > 2 and vals[2] and vals[2][0] != "N":
                    ball_speed[i] = float(vals[2])
    return teams, players, ball, ball_speed


def load(game=GAME, source="tactical"):
    """(teams, players, ball, ball_speed, (home_name, away_name)), from the cache when present."""
    path = os.path.join(CACHE_DIR, "{}_{}_tracking.npz".format(game, source))
    meta_path = require(["atd_meta"], game)[0]
    names = team_names(meta_path)
    if os.path.exists(path):
        z = np.load(path)
        return z["teams"], z["players"], z["ball"], z["ball_speed"], names
    if source == "fused":
        raise FileNotFoundError("{} missing: run solution/fusion.py first".format(path))
    print("parsing the {} ATD once (cached in data/derived/) ...".format(source), flush=True)
    teams, players, ball, ball_speed = _parse(game, source)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(path, teams=teams, players=players, ball=ball, ball_speed=ball_speed)
    return teams, players, ball, ball_speed, names
