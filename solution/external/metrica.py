"""Metrica Sports open sample games: tracking in the shape tracking.load() returns, plus H/A/D labels
derived from their event data. This is the calibration set (RULES.md: labelled data from OTHER
games is allowed; the provided game's public five minutes are not a tuning set).

Source: https://github.com/metrica-sports/sample-data (3 anonymised games, 105 x 68 m, 25 fps).
Same provider as the BAR-ATH ATD; game 3 is in the same EPTS FIFA format. Download with

    .venv/bin/python solution/external/metrica.py --download

into data/external/metrica/ (git-ignored, like all of data/).

Labels from events (an algorithm applied to someone else's human annotation, documented here):
  - graded frames: from each period's kick-off to its last event; X elsewhere.
  - dead ball: from a stoppage to the next restart.
      stoppage = ball out (end frame), shot off target / goal (end frame), foul (FAULT RECEIVED,
                 CHALLENGE *-FAULT-*, start frame), offside or hand ball (BALL LOST / PASS
                 subtype, end frame).
      restart  = the next SET PIECE or GOAL KICK, else the next PASS / CARRY / SHOT / RECOVERY.
  - alive frames: the team of the last PASS, CARRY, SHOT, SET PIECE or RECOVERY. A BALL LOST or a
    CHALLENGE does not change hands by itself: the ball stays with the last team until the other
    one recovers it or plays it ("contested or loose, possession stays with the last team").
These labels follow Metrica's event definitions, not the challenge annotators': they are a silver
standard, good for "does this change help on football in general", not for fine timing.
"""

import csv
import json
import os
import sys
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DIR = os.path.join(ROOT, "data", "external", "metrica")
URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/"
FILES = {
    1: ["data/Sample_Game_1/Sample_Game_1_RawEventsData.csv",
        "data/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv",
        "data/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv"],
    2: ["data/Sample_Game_2/Sample_Game_2_RawEventsData.csv",
        "data/Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv",
        "data/Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv"],
    3: ["data/Sample_Game_3/Sample_Game_3_events.json",
        "data/Sample_Game_3/Sample_Game_3_metadata.xml",
        "data/Sample_Game_3/Sample_Game_3_tracking.txt"],
}
GAMES = (1, 2, 3)
PITCH = (105.0, 68.0)
HOME, AWAY = 0, 1


def download():
    for files in FILES.values():
        for rel in files:
            dst = os.path.join(DIR, rel)
            if os.path.exists(dst):
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            print("downloading", rel, flush=True)
            urllib.request.urlretrieve(URL + rel, dst)


def _path(game, i):
    p = os.path.join(DIR, FILES[game][i])
    if not os.path.exists(p):
        raise FileNotFoundError("{} missing: run solution/external/metrica.py --download".format(p))
    return p


def _num(s):
    return np.nan if s in ("", "NaN", "nan") else float(s)


# ---- tracking --------------------------------------------------------------------------------

def _load_csv_tracking(game):
    """Games 1-2: one CSV per team, players as column pairs, ball in the last pair of both."""
    sides = []
    for team, i in ((HOME, 1), (AWAY, 2)):
        with open(_path(game, i)) as h:
            rows = list(csv.reader(h))
        header = rows[2]
        cols = [j for j, name in enumerate(header) if name.startswith("Player")]
        bcol = header.index("Ball")
        data = rows[3:]
        period = np.array([int(r[0]) for r in data])
        pl = np.array([[[_num(r[j]), _num(r[j + 1])] for j in cols] for r in data], dtype=np.float32)
        ball = np.array([[_num(r[bcol]), _num(r[bcol + 1])] for r in data], dtype=np.float32)
        sides.append((period, pl, ball))
    period = sides[0][0]
    players = np.concatenate([sides[0][1], sides[1][1]], axis=1)
    teams = np.array([HOME] * sides[0][1].shape[1] + [AWAY] * sides[1][1].shape[1], dtype=np.int8)
    return period, teams, players, sides[0][2]


def _load_epts_tracking(game):
    """Game 3: EPTS FIFA txt, player order given per frame range by the metadata."""
    root = ET.parse(_path(game, 1)).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    team_of_player = {p.get("id"): team_ids.index(p.get("teamId")) for p in root.iter("Player")}
    channel_player = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    pids = sorted(team_of_player, key=lambda k: (team_of_player[k], k))
    slot_of = {pid: i for i, pid in enumerate(pids)}
    specs = []
    for spec in root.iter("DataFormatSpecification"):
        refs = [r.get("playerChannelId") for r in spec.iter("PlayerChannelRef")]
        order = [slot_of[channel_player[r]] for r in refs if r.endswith("_x")]
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), order))
    halves = {}
    for p in root.iter("ProviderParameter"):
        halves[p.findtext("Name")] = p.findtext("Value")
    h1_end = int(halves["first_half_end"])

    lines = open(_path(game, 2)).read().split("\n")
    lines = [l for l in lines if l]
    n = len(lines)
    players = np.full((n, len(pids), 2), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    frame = np.zeros(n, dtype=np.int64)
    for i, line in enumerate(lines):
        f, pl, b = line.split(":")
        frame[i] = int(f)
        order = next(o for a, z, o in specs if a <= frame[i] <= z)
        for slot, xy in zip(order, pl.split(";")):
            x, y = xy.split(",")
            players[i, slot] = (_num(x), _num(y))
        bx, by = b.split(",")[:2]
        ball[i] = (_num(bx), _num(by))
    period = np.where(frame <= h1_end, 1, 2)
    teams = np.array([team_of_player[k] for k in pids], dtype=np.int8)
    return period, teams, players, ball


def load(game):
    """(teams, players, ball, ball_speed, periods) in metres, row i = Metrica frame i + 1.

    periods: [(period, first row, last row)] -- the whole recorded period, before grading."""
    cache = os.path.join(ROOT, "data", "derived", "metrica_{}.npz".format(game))
    if os.path.exists(cache):
        z = np.load(cache)
        period, teams, players, ball = z["period"], z["teams"], z["players"], z["ball"]
    else:
        loader = _load_epts_tracking if game == 3 else _load_csv_tracking
        period, teams, players, ball = loader(game)
        players = players * np.array(PITCH, dtype=np.float32)
        ball = ball * np.array(PITCH, dtype=np.float32)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, period=period, teams=teams, players=players, ball=ball)
    periods = []
    for k in np.unique(period):
        idx = np.flatnonzero(period == k)
        periods.append((int(k), int(idx[0]), int(idx[-1])))
    ball_speed = np.full(len(ball), np.nan, dtype=np.float32)
    return teams, players, ball, ball_speed, periods


# ---- events -> labels -------------------------------------------------------------------------

def events(game):
    """[(team, type, subtypes set, start row, end row, period)], rows 0-based, in time order."""
    out = []
    if game == 3:
        meta = ET.parse(_path(3, 1)).getroot()
        home_name = next(meta.iter("Team")).findtext("Name")
        for e in json.load(open(_path(3, 0)))["data"]:
            sub = e.get("subtypes") or []
            sub = sub if isinstance(sub, list) else [sub]
            team = HOME if e["team"]["name"] == home_name else AWAY
            out.append((team, e["type"]["name"], {s["name"] for s in sub},
                        e["start"]["frame"] - 1, e["end"]["frame"] - 1, e["period"]))
    else:
        with open(_path(game, 0)) as h:
            for r in csv.DictReader(h):
                team = HOME if r["Team"] == "Home" else AWAY
                sub = set(s.strip() for s in r["Subtype"].split("-")) if r["Subtype"] else set()
                # keep the joined forms too ("OFF TARGET", "ON TARGET") for the shot rules
                sub |= {r["Subtype"]} if r["Subtype"] else set()
                start = int(r["Start Frame"]) - 1
                end = max(start, int(r["End Frame"]) - 1)
                out.append((team, r["Type"], sub, start, end, int(r["Period"])))
    out.sort(key=lambda e: e[3])
    return out


OWNING = {"PASS", "CARRY", "SHOT", "SET PIECE", "RECOVERY"}
RESTART_ANY = {"PASS", "CARRY", "SHOT", "RECOVERY"}


def _stoppage(typ, sub):
    """Frame-choice of a stoppage: 'start', 'end' or None if the event does not stop play."""
    if typ == "BALL OUT":
        return "end"
    if typ == "SHOT" and ({"OUT", "GOAL", "OFF TARGET-OUT", "HEAD-OFF TARGET-OUT"} & sub
                         or any("OUT" in s or "GOAL" in s for s in sub)):
        return "end"
    if typ == "FAULT RECEIVED":
        return "start"
    if typ == "CHALLENGE" and any("FAULT" in s for s in sub) and not any("ADVANTAGE" in s for s in sub):
        return "start"
    if typ in ("BALL LOST", "PASS") and ({"OFFSIDE"} & sub):
        return "end"
    if typ == "BALL LOST" and ({"HAND BALL"} & sub):
        return "start"
    return None


def labels(game, n, periods):
    """H/A/D/X per row, from the events (see the module docstring)."""
    ev = events(game)
    lab = np.full(n, "X", dtype="<U1")
    for period, first, last in periods:
        mine = [e for e in ev if e[5] == period and first <= e[3] <= last]
        kick = [e for e in mine if e[1] == "SET PIECE" and "KICK OFF" in e[2]]
        if not kick:
            continue
        start = kick[0][3]
        stop = min(last, max(e[4] for e in mine))
        owner = kick[0][0]
        lab[start:stop + 1] = "H" if owner == HOME else "A"
        # alive possession, event by event. CHALLENGE.md: a change needs CONTROLLED possession,
        # "a simple touch of the ball without control" does not count. A recovery therefore hands
        # the ball over only if that team then plays it (CONTROL event) before the other team
        # does; it is dated to the recovery.
        cur, t0 = owner, start
        pending = None  # (team, frame) of a recovery not yet confirmed
        for e in mine:
            if e[3] < start or e[1] not in OWNING:
                continue
            if e[1] == "RECOVERY":
                if e[0] != cur and (pending is None or pending[0] != e[0]):
                    pending = (e[0], e[3])
                elif e[0] == cur:
                    pending = None
                continue
            if e[0] == cur:
                pending = None
                continue
            at = pending[1] if pending is not None and pending[0] == e[0] else e[3]
            lab[t0:at] = "H" if cur == HOME else "A"
            cur, t0, pending = e[0], at, None
        lab[t0:stop + 1] = "H" if cur == HOME else "A"
        # dead balls on top
        for i, e in enumerate(mine):
            kind = _stoppage(e[1], e[2])
            if kind is None or e[3] < start:
                continue
            d0 = e[3] if kind == "start" else e[4]
            restart = None
            for f in mine[i + 1:]:
                if f[3] <= d0:
                    continue
                if f[1] == "SET PIECE" or "GOAL KICK" in f[2] or f[1] in RESTART_ANY:
                    restart = f[3]
                    break
            if restart is None:
                restart = stop + 1
            lab[d0:min(restart, stop + 1)] = "D"
    return lab


def blocks_from_labels(lab):
    """Graded blocks [(k, first, last)]: the non-X runs, one per period."""
    idx = np.flatnonzero(np.diff(np.concatenate(([0], (lab != "X").astype(np.int8), [0]))))
    return [(k + 1, int(a), int(b) - 1) for k, (a, b) in enumerate(zip(idx[::2], idx[1::2]))]


if __name__ == "__main__":
    if "--download" in sys.argv:
        download()
    for g in GAMES:
        teams, players, ball, _, periods = load(g)
        lab = labels(g, len(ball), periods)
        graded = lab != "X"
        seg = lab[graded]
        cut = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        starts = np.concatenate(([0], cut))
        kinds = seg[starts]
        lens = np.diff(np.concatenate((starts, [len(seg)])))
        print("game {}: {} frames graded ({:.0f} min), ball visible {:.1f}%, H {:.0f}% A {:.0f}% D {:.0f}%,"
              " {} possessions (mean {:.1f} s), {} dead balls (mean {:.1f} s)".format(
                  g, graded.sum(), graded.sum() / 1500, 100 * np.mean(~np.isnan(ball[graded, 0])),
                  100 * np.mean(seg == "H"), 100 * np.mean(seg == "A"), 100 * np.mean(seg == "D"),
                  np.sum(kinds != "D"), lens[kinds != "D"].mean() / 25, np.sum(kinds == "D"),
                  lens[kinds == "D"].mean() / 25))
