"""SkillCorner open data (A-League 2024/25): broadcast tracking in the shape idsse.load() returns,
plus H/A/D labels from SkillCorner's phases of play and player-possession events. A third
calibration set next to Metrica and IDSSE (RULES.md: labelled data from OTHER games is allowed).

Source: https://github.com/SkillCorner/opendata (MIT licence, (c) SkillCorner; credit SkillCorner
when used), pinned here to commit REPO_SHA (September 2026: 20 full matches). Download (about
1.9 GB) with

    .venv/bin/python solution/external/skillcorner.py --download

into data/external/skillcorner/ (git-ignored). The first load parses the JSON lines (a few seconds
per match) and caches the 25 fps arrays in data/external/skillcorner/cache/<id>.npz.

What this tracking is, and why it is different from Metrica and IDSSE:
  - It is made from the BROADCAST video (computer vision), at 10 fps. Rows here are 25 fps: row r
    is the source time r / 25 s (source frame r * 0.4), positions linearly interpolated between
    consecutive source frames, never across a hole longer than MAX_GAP_S, NaN where not tracked.
  - Whenever the director shows a replay or a close-up there is NO tracking at all (no player, no
    ball): about a quarter of every match, almost all of it during stoppages.
  - When there is tracking, all 22 players are always given: the ones outside the picture are
    EXTRAPOLATED by the provider (is_detected false; about 14 of 22 are really seen in play, 9 in
    stoppages). load(game, extrapolated=False) blanks them; raw(game)["pdet"] is the mask.
  - The ball is tracked while it is carried, thrown back or rolled to the restart spot during a
    stoppage (unlike TRACAB, whose ball freezes, and Metrica, whose ball disappears). Samples
    flagged is_detected false are the provider's own interpolation (raw(game)["bdet"]: 1 seen,
    2 interpolated, 0 missing).
  - SkillCorner says about 97% of the player identities are right, and that speeds need smoothing.

Labels (an algorithm applied to SkillCorner's derived data, documented here):
  - ball in play = inside a "phase of play" (README: "Phases of play are only defined when the
    ball is in play"). The phases are NOT a measured flag: SkillCorner builds them from its
    player-possession events and from set piece / goal events (Wyscout events synchronised to the
    tracking). Measured on these matches, a phase that is followed by a stoppage ends exactly 1.0 s
    after the last touch in 4 cases out of 5 (a nominal end), while the tracked ball really crosses
    the line 1.2-1.5 s after the last touch on median, anywhere between 0 and 3 s. A phase starts
    at the restart itself (the throw / the kick): the ball starts moving within 0.5 s of it in
    about two cases out of three where the ball is seen.
  - variant "refined" (default): the same, with the START of every stoppage moved to the row where
    the tracked ball first leaves the pitch, searched from the last touch to OUT_SEARCH_S after it
    (and not beyond the restart); when the ball is already out at the last touch (carried over
    the line), back to where it crossed. Used for about 6 stoppages out of 10 (5 in 6 of the
    throw-ins, goal kicks, corners and goals: the ones with the ball in the picture); fouls and
    the others keep the nominal end.
  - variant "phases": D = graded rows outside every phase, as they are.
  - in both, a hole between two phases is NOT a stoppage when it is shorter than FLIGHT_MAX_S and
    the tracked ball flies through it inside the pitch (bridge_flights: match 1959846 only).
  - team (team="events", default): the team of SkillCorner's player possessions ("the player needs
    to be in control of the ball, i.e. able to do a pass, shot or carry"; blocks and uncontrolled
    touches are not possessions). A possession that ends with a pass, a shot or a foul suffered, or
    lasts FIRM_S or more, gives the ball to its team; a shorter one (clearance, immediate loss)
    only PROPOSES a change, confirmed, and dated to it, if that team has the next firm possession
    before the other one does ("contested or loose, possession stays with the last team"). Each
    alive run starts with the restart taker: the team of its first possession, or the other one
    when SkillCorner says that possession intercepted the restart ("throw_in_against").
    team="phases": SkillCorner's own team_in_possession of the phase. team="frame": the tracking's
    per-frame possession.group, carried forward.
  - graded rows: from the first to the last in-play row of each half (X elsewhere).

These are silver labels. The team is good (the possessions are SkillCorner's product); the dead
ball is the weak part: its start is exact only where the ball is seen crossing the line, a foul's
whistle is "last touch + 1 s", and the restart is as good as the Wyscout synchronisation.
"""

import csv
import json
import os
import sys
import urllib.request
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DIR = os.path.join(ROOT, "data", "external", "skillcorner")
FPS = 25
SRC_FPS = 10
HOME, AWAY = 0, 1
REPO_SHA = "4340d274572876239c154c90bc507a9b3250a656"
RAW = "https://raw.githubusercontent.com/SkillCorner/opendata/{}/data/".format(REPO_SHA)
LFS = "https://media.githubusercontent.com/media/SkillCorner/opendata/{}/data/".format(REPO_SHA)
GAMES = ("1874553", "1886347", "1899585", "1925299", "1927964", "1953632", "1959846", "1986691",
         "1996435", "1996436", "2006229", "2006363", "2007448", "2007721", "2010085", "2011166",
         "2013725", "2015213", "2016236", "2017461")
KINDS = ("match.json", "tracking_extrapolated.jsonl", "dynamic_events.csv", "phases_of_play.csv")
MAX_GAP_S = 0.5          # never interpolate a position across a hole longer than this
OUT_SEARCH_S = 5.0       # "refined": look for the ball leaving the pitch up to this long after the last touch
OUT_MIN_ROWS = 3         # ...and it must stay out this many rows (or until the tracking stops)
FIRM_S = 1.0             # a possession this long is controlled whatever its end
FLIGHT_MAX_S = 6.0       # a hole between phases shorter than this, with the ball flying inside the pitch...
FLIGHT_SPEED = 3.0       # ...faster than this (m/s, median), is play
EDGE_MIN_S = 0.5         # a half does not end on an in-play run shorter than this
OWN_ENDS = ("pass", "shot", "foul_suffered")
OUT_KINDS = ("throw_in", "goal_kick", "corner", "goal")


def _name(game, k):
    return os.path.join(DIR, game, "{}_{}".format(game, KINDS[k]))


def download():
    for game in GAMES:
        os.makedirs(os.path.join(DIR, game), exist_ok=True)
        for k, kind in enumerate(KINDS):
            dst = _name(game, k)
            if os.path.exists(dst):
                continue
            base = LFS if k == 1 else RAW      # the tracking is a git-lfs object
            print("downloading", os.path.basename(dst), flush=True)
            urllib.request.urlretrieve("{}matches/{}/{}_{}".format(base, game, game, kind), dst + ".part")
            os.replace(dst + ".part", dst)


def _path(game, k):
    p = _name(game, k)
    if not os.path.exists(p):
        raise FileNotFoundError("{} missing: run solution/external/skillcorner.py --download".format(p))
    return p


def match_info(game):
    """(home team id, away team id, (length, width), {player id: 0 home / 1 away})."""
    m = json.load(open(_path(game, 0)))
    home, away = m["home_team"]["id"], m["away_team"]["id"]
    side = {p["id"]: (HOME if p["team_id"] == home else AWAY) for p in m["players"]}
    return home, away, (float(m["pitch_length"]), float(m["pitch_width"])), side


def pitch(game):
    return match_info(game)[2]


def row_of(frame):
    """25 fps row of a 10 fps source frame (nearest)."""
    return (5 * np.asarray(frame, dtype=np.int64) + 1) // 2


# ---- tracking --------------------------------------------------------------------------------

def _to25(v, n25):
    """One track (n10, k) with NaN holes -> (n25, k): linear between the two nearest tracked source
    frames when they are at most MAX_GAP_S apart, NaN elsewhere."""
    out = np.full((n25, v.shape[1]), np.nan, dtype=np.float32)
    idx = np.flatnonzero(~np.isnan(v[:, 0]))
    if not len(idx):
        return out
    pos = 2 * np.arange(n25, dtype=np.int64)        # row r sits at source frame 2r / 5
    at = 5 * idx
    j = np.searchsorted(at, pos, side="right")
    lo = np.clip(j - 1, 0, len(idx) - 1)
    hi = np.clip(j, 0, len(idx) - 1)
    exact = (j > 0) & (at[lo] == pos)
    span = at[hi] - at[lo]
    inside = (j > 0) & (j < len(idx)) & (span <= MAX_GAP_S * SRC_FPS * 5) & ~exact
    out[exact] = v[idx[lo[exact]]]
    w = ((pos[inside] - at[lo[inside]]) / span[inside]).astype(np.float32)[:, None]
    out[inside] = (1 - w) * v[idx[lo[inside]]] + w * v[idx[hi[inside]]]
    return out


def _build(game):
    home, away, pitch_lw, side = match_info(game)
    with open(_path(game, 1)) as h:
        rows = [json.loads(line) for line in h]
    n10 = len(rows)
    seen = sorted({p["player_id"] for r in rows for p in r["player_data"]}, key=lambda k: (side[k], k))
    slot = {pid: i for i, pid in enumerate(seen)}
    players = np.full((n10, len(seen), 2), np.nan, dtype=np.float32)
    pdet = np.zeros((n10, len(seen)), dtype=bool)
    ball = np.full((n10, 3), np.nan, dtype=np.float32)
    bdet = np.zeros(n10, dtype=np.int8)
    owner = np.zeros(n10, dtype=np.int8)
    period = np.zeros(n10, dtype=np.int8)
    for i, r in enumerate(rows):
        if r["frame"] != i:
            raise ValueError("{}: frame {} at line {}".format(game, r["frame"], i))
        b = r["ball_data"]
        if b["x"] is not None:
            ball[i] = (b["x"], b["y"], b["z"] if b["z"] is not None else np.nan)
            bdet[i] = 1 if b["is_detected"] else 2
        owner[i] = {"home team": 1, "away team": 2}.get(r["possession"]["group"], 0)
        period[i] = r["period"] or 0
        for p in r["player_data"]:
            if p["x"] is None:
                continue
            s = slot[p["player_id"]]
            players[i, s] = (p["x"], p["y"])
            pdet[i, s] = bool(p["is_detected"])
    off = np.array([pitch_lw[0] / 2, pitch_lw[1] / 2], dtype=np.float32)
    players += off
    ball[:, :2] += off
    n25 = (5 * (n10 - 1)) // 2 + 1
    near = np.minimum((2 * np.arange(n25) + 2) // 5, n10 - 1)     # nearest source frame of each row
    p25 = np.stack([_to25(players[:, s], n25) for s in range(len(seen))], axis=1)
    b25 = _to25(ball, n25)
    return dict(period=period[near], teams=np.array([side[k] for k in seen], dtype=np.int8),
                players=p25, pdet=pdet[near] & ~np.isnan(p25[:, :, 0]), ball=b25[:, :2], ball_z=b25[:, 2],
                bdet=np.where(np.isnan(b25[:, 0]), 0, bdet[near]).astype(np.int8), owner=owner[near],
                pids=np.array(seen, dtype=np.int64), pitch=np.array(pitch_lw))


def _cache(game):
    return os.path.join(DIR, "cache", "{}.npz".format(game))


def build_cache(game):
    if not os.path.exists(_cache(game)):
        z = _build(game)
        os.makedirs(os.path.dirname(_cache(game)), exist_ok=True)
        np.savez_compressed(_cache(game) + ".part.npz", **z)
        os.replace(_cache(game) + ".part.npz", _cache(game))
    return game


def raw(game):
    """Every cached array at 25 fps: period (0 outside the halves), teams, players, pdet (player
    really seen, not extrapolated), ball, ball_z, bdet (1 seen, 2 interpolated by the provider,
    0 missing), owner (the tracking's possession.group: 1 home, 2 away, 0 none), pids, pitch."""
    build_cache(game)
    return dict(np.load(_cache(game)))


def load(game, extrapolated=True):
    """(teams, players, ball, ball_speed, periods) in metres, origin at a pitch corner.

    extrapolated=False keeps only what the cameras saw: players outside the picture and ball
    samples interpolated by the provider become NaN."""
    z = raw(game)
    players, ball = z["players"], z["ball"]
    if not extrapolated:
        players = np.where(z["pdet"][:, :, None], players, np.nan).astype(np.float32)
        ball = np.where((z["bdet"] == 1)[:, None], ball, np.nan).astype(np.float32)
    periods = []
    for k in np.unique(z["period"]):
        if k == 0:
            continue
        idx = np.flatnonzero(z["period"] == k)
        periods.append((int(k), int(idx[0]), int(idx[-1])))
    ball_speed = np.full(len(ball), np.nan, dtype=np.float32)
    return z["teams"], players, ball, ball_speed, periods


# ---- events -----------------------------------------------------------------------------------

def phases(game):
    """[(first row, end row (exclusive), period, team)] of SkillCorner's phases of play."""
    home = match_info(game)[0]
    out = []
    with open(_path(game, 3)) as h:
        for r in csv.DictReader(h):
            out.append((int(row_of(int(r["frame_start"]))), int(row_of(int(r["frame_end"]))), int(r["period"]),
                        HOME if int(r["team_in_possession_id"]) == home else AWAY))
    out.sort()
    return out


def possessions(game):
    """[(first row, last row, team, end type, interruption before, interruption after, taker)] of
    the player possessions, in time order. Interruptions without their _for / _against suffix;
    taker: the team that took the restart before this possession (the other team when the
    restart was intercepted, "_against"), None when no restart came before."""
    home = match_info(game)[0]
    cut = lambda s: s.replace("_for", "").replace("_against", "")
    out = []
    with open(_path(game, 2)) as h:
        for r in csv.DictReader(h):
            if r["event_type"] != "player_possession":
                continue
            team = HOME if int(float(r["team_id"])) == home else AWAY
            before = r["game_interruption_before"]
            taker = None if not before else (1 - team if before.endswith("_against") else team)
            out.append((int(row_of(int(r["frame_start"]))), int(row_of(int(r["frame_end"]))), team,
                        r["end_type"], cut(before), cut(r["game_interruption_after"]), taker))
    out.sort(key=lambda p: p[:2])
    return out


def _runs(mask):
    idx = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(idx[::2].tolist(), idx[1::2].tolist()))


def ball_out(ball, pitch_lw):
    """Rows where the tracked ball is outside the pitch."""
    with np.errstate(invalid="ignore"):
        return (ball[:, 0] < 0) | (ball[:, 0] > pitch_lw[0]) | (ball[:, 1] < 0) | (ball[:, 1] > pitch_lw[1])


def bridge_flights(alive, ball, pitch_lw):
    """Close the holes between phases that are not stoppages: shorter than FLIGHT_MAX_S, with the
    ball tracked on 80% of the rows, never outside the pitch and travelling (median speed above
    FLIGHT_SPEED). Match 1959846 has about 45 of them (a restart or a long pass is a phase of its
    own there, and the ball's flight to the next player is left out); the others 0 to 2."""
    out = ball_out(ball, pitch_lw)
    d = ball[4:] - ball[:-4]
    v = np.full(len(ball), np.nan, dtype=np.float32)
    v[2:-2] = np.hypot(d[:, 0], d[:, 1]) * FPS / 4
    new = alive.copy()
    for a, b in _runs(~alive)[1:-1]:
        if b - a >= FLIGHT_MAX_S * FPS or out[a:b].any() or np.mean(~np.isnan(ball[a:b, 0])) < 0.8:
            continue
        w = v[a:b][~np.isnan(v[a:b])]
        if len(w) and np.median(w) > FLIGHT_SPEED:
            new[a:b] = True
    return new


def refine_stoppages(alive, ball, pitch_lw, pp, stats=None):
    """Move the start of each stoppage to the row where the tracked ball leaves the pitch.

    alive: in-play mask from the phases. For every alive run followed by a stoppage, the ball is
    looked for outside the pitch from the last touch (end of the last player possession of the
    run) to OUT_SEARCH_S after it, never beyond the next restart. Only stoppages that SkillCorner
    names as a throw-in, goal kick, corner or goal are touched. Returns a new mask."""
    out = ball_out(ball, pitch_lw)
    seen = ~np.isnan(ball[:, 0])
    starts = np.array([p[0] for p in pp])
    order = np.argsort([p[1] for p in pp], kind="stable")
    ends = np.array([pp[k][1] for k in order])
    new = alive.copy()
    runs = _runs(alive)
    n_out = n_ref = 0
    shifts = []
    for (a, b), nxt in zip(runs, runs[1:] + [None]):
        if nxt is None:
            continue
        i = np.searchsorted(ends, b, side="right") - 1          # last possession ended before the phase did
        if i < 0 or ends[i] < a:
            continue
        last = pp[order[i]]
        j = np.searchsorted(starts, nxt[0] - FPS)               # first possession of the next run
        kind = last[5] or (pp[j][4] if j < len(pp) and pp[j][0] < nxt[1] else "")
        if kind not in OUT_KINDS:
            continue
        n_out += 1
        lo, hi = int(ends[i]), int(min(nxt[0], ends[i] + OUT_SEARCH_S * FPS))
        at = None
        for r in np.flatnonzero(out[lo:hi]) + lo:
            w = slice(r, min(r + OUT_MIN_ROWS, hi))
            if np.all(out[w] | ~seen[w]):
                at = int(r)
                break
        if at == lo:        # already out at the last touch (carried over the line): walk back to the crossing
            floor = max(a, last[0], lo - int(OUT_SEARCH_S * FPS))
            while at > floor and out[at - 1]:
                at -= 1
        if at is None:
            continue
        n_ref += 1
        shifts.append(at - b)
        new[b:at] = True          # the ball was still in when the phase nominally ended
        new[at:b] = False         # ...or already out before
    if stats is not None:
        stats.update(stoppages=len(runs) - 1, ball_out_kind=n_out, refined=n_ref,
                     shift_s=np.array(shifts, dtype=float) / FPS)
    return new


def _team_runs(pp, alive, lab):
    """Fill lab on every alive run with the team of the player possessions (module docstring)."""
    starts = np.array([p[0] for p in pp])
    name = ("H", "A")
    cur = HOME
    for a, b in _runs(alive):
        lo, hi = np.searchsorted(starts, a - FPS), np.searchsorted(starts, b)
        mine = [p for p in pp[lo:hi] if p[1] >= a]
        if mine:        # the restart taker, also when the other team intercepts the restart
            cur = mine[0][6] if mine[0][6] is not None else mine[0][2]
        t0, pending = a, None
        for s, e, team, end, _, _, _ in mine:
            firm = end in OWN_ENDS or (e - s) >= FIRM_S * FPS
            if team == cur:
                pending = None
                continue
            if not firm:
                if pending is None or pending[0] != team:
                    pending = (team, s)
                continue
            at = pending[1] if pending is not None and pending[0] == team else s
            at = min(max(at, a), b)
            lab[t0:at] = name[cur]
            cur, t0, pending = team, at, None
        lab[t0:b] = name[cur]


def labels(game, variant="refined", team="events", stats=None):
    """H/A/D/X per 25 fps row. variant: 'refined' (default) or 'phases'; team: 'events' (default),
    'phases' or 'frame' (see the module docstring). stats: a dict filled with what 'refined' did."""
    z = raw(game)
    n = len(z["period"])
    ph = phases(game)
    pp = possessions(game)
    alive = np.zeros(n, dtype=bool)
    # the last "phase" of the first half is stretched by SkillCorner from the whistle to the second
    # half's kick-off, across the interval: a phase that leaves its own half is not play
    last = {int(k): int(np.flatnonzero(z["period"] == k)[-1]) for k in np.unique(z["period"]) if k}
    ph = [p for p in ph if p[1] <= last.get(p[2], -1) + 1]
    for a, b, k, _ in ph:
        alive[a:b] |= z["period"][a:b] == k
    g = np.zeros(n, dtype=bool)
    for k in sorted({p[2] for p in ph}):
        runs = _runs(alive & (z["period"] == k))
        while len(runs) > 1 and runs[-1][1] - runs[-1][0] < EDGE_MIN_S * FPS:
            alive[runs[-1][0]:runs[-1][1]] = False      # a one-frame "phase" at the whistle
            runs.pop()
        if runs:
            g[runs[0][0]:runs[-1][1]] = True
    alive = bridge_flights(alive, z["ball"], z["pitch"]) & g
    if variant == "refined":
        alive = refine_stoppages(alive, z["ball"], z["pitch"], pp, stats) & g
    elif variant != "phases":
        raise ValueError("variant: refined or phases")
    lab = np.full(n, "X", dtype="<U1")
    lab[g] = "D"
    if team == "events":
        _team_runs(pp, alive, lab)
    elif team == "phases":
        own = np.full(n, -1, dtype=np.int8)
        for a, b, k, t in ph:
            own[a:b] = np.where(z["period"][a:b] == k, t, own[a:b])
        for a, b in _runs(alive):
            seg = own[a:b].copy()
            idx = np.flatnonzero(seg >= 0)
            if not len(idx):
                seg[:] = own[max(a - 1, 0)] if own[max(a - 1, 0)] >= 0 else HOME
            else:
                fill = np.maximum.accumulate(np.where(seg >= 0, np.arange(len(seg)), -1))
                seg = seg[np.where(fill >= 0, fill, idx[0])]
            lab[a:b] = np.where(seg == HOME, "H", "A")
    elif team == "frame":
        own = z["owner"]
        for a, b in _runs(alive):
            seg = own[a:b]
            idx = np.flatnonzero(seg > 0)
            if not len(idx):
                lab[a:b] = lab[a - 1] if a and lab[a - 1] in ("H", "A") else "H"
                continue
            fill = np.maximum.accumulate(np.where(seg > 0, np.arange(len(seg)), -1))
            seg = seg[np.where(fill >= 0, fill, idx[0])]
            lab[a:b] = np.where(seg == 1, "H", "A")
    else:
        raise ValueError("team: events, phases or frame")
    lab[~g] = "X"
    return lab


def blocks_from_labels(lab):
    """Graded blocks [(k, first, last)]: the non-X runs, one per period."""
    idx = np.flatnonzero(np.diff(np.concatenate(([0], (lab != "X").astype(np.int8), [0]))))
    return [(k + 1, int(a), int(b) - 1) for k, (a, b) in enumerate(zip(idx[::2], idx[1::2]))]


if __name__ == "__main__":
    import sanity
    if "--download" in sys.argv:
        download()
    games = [g for g in GAMES if os.path.exists(_name(g, 1))]
    with Pool(min(8, os.cpu_count())) as pool:
        pool.map(build_cache, games)
    rows = []
    for g in games:
        z = raw(g)
        st = {}
        lab = labels(g, stats=st)
        lp = labels(g, "phases")
        gm = lab != "X"
        alive = gm & (lab != "D")
        s = sanity.aggregates(lab, z["players"], z["ball"])
        seen = sanity.aggregates(lab, np.where(z["pdet"][:, :, None], z["players"], np.nan),
                                 np.where((z["bdet"] == 1)[:, None], z["ball"], np.nan))
        rows.append(s)
        sh = st["shift_s"]
        print(sanity.line(g, s), flush=True)
        print("    really seen by the cameras: ball {:.1f}% of graded, {:.1f}% of dead (moving {:.1f}%), players {:.1f} "
              "per frame; no tracking at all on {:.1f}% of dead and {:.1f}% of alive rows".format(
                  seen["ball_pct"], seen["dead_ball_pct"], seen["dead_ball_moving_pct"], seen["players"],
                  100 * np.mean(np.isnan(z["players"][gm & ~alive, :, 0]).all(axis=1)),
                  100 * np.mean(np.isnan(z["players"][alive, :, 0]).all(axis=1))))
        print("    refined {} of {} stoppages ({} named throw-in / goal kick / corner / goal): start moved by "
              "median {:+.2f} s, quartiles {:+.2f} / {:+.2f}, within 0.5 s {:.0f}%; D phases {:.1f}% -> refined {:.1f}%; "
              "team events vs phases {:.1f}%, vs frame {:.1f}%".format(
                  st["refined"], st["stoppages"], st["ball_out_kind"], np.median(sh), *np.percentile(sh, [25, 75]),
                  100 * np.mean(np.abs(sh) <= 0.5), 100 * np.mean(lp[gm] == "D"), 100 * np.mean(lab[gm] == "D"),
                  100 * np.mean(lab[alive] == labels(g, team="phases")[alive]),
                  100 * np.mean(lab[alive] == labels(g, team="frame")[alive])), flush=True)
    print(sanity.line("mean of {}".format(len(rows)), sanity.mean(rows)))
