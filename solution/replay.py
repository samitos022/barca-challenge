"""Replays in the broadcast: frames that show a moment of the match already played.

A broadcast frame that is not live (its players do not match the tactical tracking at the synced
time) is a replay when they match the tactical tracking of an EARLIER moment: the same test as
sync_broadcast.py (median distance from each broadcast player to the closest tactical player of
his team under LIVE_COST), searched MIN_BACK_S..MAX_BACK_S back. Directors show replays while the
game is stopped, so a replay is a dead-ball signal that does not depend on the ball tracking.

No labels are used. The false-alarm rate is measured by running the same search FORWARD in time
(a replay cannot show the future): a forward match is a chance match.

A sample counts only inside a run of at least MIN_RUN consecutive matching samples whose shown
moment moves forward at most as fast as real time (normal speed or slow motion), and the whole
run is kept: chance matches are isolated and their shown moment jumps around.

Writes data/derived/<game>_replays.npz:
    replay_t   (n_tactical,) bool   the broadcast is showing a replay at this tactical frame
    runs       (k, 4) int           tactical start, end, shown start, shown end of each replay

    .venv/bin/python solution/replay.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, asset_paths, half_video_blocks, source_frames  # noqa: E402
import tracking  # noqa: E402
import sync_broadcast  # noqa: E402

FPS = 25
OUT = os.path.join(ROOT, "data", "derived", "{}_replays.npz".format(GAME))
P = {
    "EVERY": 5,             # broadcast frames between checks (as sync_broadcast TRACK_EVERY)
    "MIN_PLAYERS": 6,       # broadcast players needed for a check (more than sync's 4: the search
                            # here is 2,000 times wider, so a chance match must be harder)
    "LIVE_COST": sync_broadcast.P["LIVE_COST"],  # m, the same test as the synchronisation
    "MIN_BACK_S": 2.0,      # s: closer than this it is a delay, not a replay
    "MAX_BACK_S": 90.0,     # s: goal replays come up to a minute or more after the goal
    "MIN_RUN": 3,           # consecutive matching checks (0.6 s) for a replay
    "MAX_SPEED": 1.25,      # shown time may advance at most this much per broadcast frame (real
                            # speed is 1, slow motion less; margin for the +-1 frame search)
}


def team_cost(bp, bt, cand, t_pl, t_teams):
    """sync_broadcast._cost for one broadcast frame against many tactical frames at once."""
    meds = []
    for team in (0, 1):
        a = bp[(bt == team) & ~np.isnan(bp[:, 0])]
        if not len(a):
            continue
        tp = t_pl[cand][:, t_teams == team]                     # (K, s, 2)
        d = np.linalg.norm(tp[:, None] - a[None, :, None], axis=3)  # (K, m, s)
        d = np.where(np.isnan(d), np.inf, d).min(axis=2)        # (K, m)
        meds.append(d)
    if not meds:
        return np.full(len(cand), np.inf)
    return np.median(np.concatenate(meds, axis=1), axis=1)


def search(samples, t_of_b, b_pl, b_teams, t_pl, t_teams, sign, p):
    """Best match per sample, searching back (sign -1) or forward (+1): (cost, shown frame)."""
    n_t = len(t_pl)
    lo, hi = int(p["MIN_BACK_S"] * FPS), int(p["MAX_BACK_S"] * FPS)
    best = np.full(len(samples), np.inf)
    shown = np.full(len(samples), -1, np.int64)
    for i, b in enumerate(samples):
        t0 = t_of_b[b]
        cand = t0 + sign * np.arange(lo, hi + 1)
        cand = cand[(cand >= 0) & (cand < n_t)]
        if not len(cand):
            continue
        c = team_cost(b_pl[b], b_teams, cand, t_pl, t_teams)
        k = int(np.argmin(c))
        best[i], shown[i] = c[k], cand[k]
    return best, shown


def runs_of(samples, ok, shown, p):
    """Runs of consecutive matching samples whose shown moment moves forward plausibly."""
    out, cur = [], []
    for i in range(len(samples)):
        good = ok[i]
        if good and cur:
            j = cur[-1]
            db, ds = samples[i] - samples[j], shown[i] - shown[j]
            good_step = db <= p["EVERY"] and 0 <= ds <= p["MAX_SPEED"] * db + 2
            if not good_step:
                if len(cur) >= p["MIN_RUN"]:
                    out.append(cur)
                cur = []
        if good:
            cur.append(i)
        else:
            if len(cur) >= p["MIN_RUN"]:
                out.append(cur)
            cur = []
    if len(cur) >= p["MIN_RUN"]:
        out.append(cur)
    return out


def main(p=P):
    s = np.load(sync_broadcast.OUT)
    t_of_b, live = s["t_of_b"], s["live"]
    t_teams, t_pl, *_ = tracking.load(GAME)
    b_teams, b_pl, *_ = tracking.load(GAME, "broadcast")
    hcsv = asset_paths(GAME, source="broadcast")["homography"]
    valid_h = np.loadtxt(hcsv, delimiter=",", skiprows=1, usecols=(11,)) == 1
    n_t, n_b = source_frames("tactical"), len(b_pl)
    n_pl = np.sum(~np.isnan(b_pl[:, :, 0]), axis=1)
    samples = np.array([b for b in range(0, n_b, p["EVERY"])
                        if valid_h[b] and not live[b] and t_of_b[b] >= 0 and n_pl[b] >= p["MIN_PLAYERS"]])
    print("non-live broadcast checks with a usable homography and >= {} players: {}".format(
        p["MIN_PLAYERS"], len(samples)), flush=True)

    res = {}
    for name, sign in (("back", -1), ("forward", +1)):
        cost, shown = search(samples, t_of_b, b_pl, b_teams, t_pl, t_teams, sign, p)
        ok = cost <= p["LIVE_COST"]
        runs = runs_of(samples, ok, shown, p)
        res[name] = (cost, shown, ok, runs)
        print("{:8s}: single matches {:5d} ({:.1f}% of checks), runs >= {} checks: {}, checks in runs {}".format(
            name, int(ok.sum()), 100 * ok.mean(), p["MIN_RUN"], len(runs), sum(len(r) for r in runs)))

    cost, shown, ok, runs = res["back"]
    n_fwd = sum(len(r) for r in res["forward"][3])
    n_back = sum(len(r) for r in runs)
    print("chance matches (forward) / replay matches (back), in run checks: {:.1%}".format(
        n_fwd / max(n_back, 1)))

    replay_t = np.zeros(n_t, bool)
    rows = []
    for r in runs:
        b0, b1 = samples[r[0]], samples[r[-1]] + p["EVERY"] - 1
        t0, t1 = t_of_b[b0], t_of_b[min(b1, n_b - 1)]
        if t1 < t0:
            continue
        replay_t[t0:t1 + 1] = True
        rows.append((t0, t1, shown[r[0]], shown[r[-1]]))
    rows = np.array(rows, np.int64).reshape(-1, 4)
    # merge runs of the same replay (a cut between two angles) for the summary
    g = np.zeros(n_t, bool)
    for _, a, z in half_video_blocks(GAME):
        g[a:z + 1] = True
    e = np.flatnonzero(np.diff(np.concatenate(([0], replay_t.astype(np.int8), [0]))))
    L = (e[1::2] - e[::2]) / FPS
    lag = (rows[:, 0] - rows[:, 2]) / FPS if len(rows) else np.zeros(0)
    print("replay stretches: {} ({} runs), {:.1f}% of graded frames, length median {:.1f} s, p90 {:.1f} s; "
          "shown moment {:.0f} s back (median)".format(len(L), len(rows), 100 * replay_t[g].mean(),
                                                     np.median(L) if len(L) else 0,
                                                     np.percentile(L, 90) if len(L) else 0,
                                                     np.median(lag) if len(lag) else 0))
    sp = []
    for r in runs:
        db = samples[r[-1]] - samples[r[0]]
        if db > 0:
            sp.append((shown[r[-1]] - shown[r[0]]) / db)
    if sp:
        print("replay speed (shown / broadcast time): median {:.2f}, share slow motion (< 0.8) {:.0f}%".format(
            np.median(sp), 100 * np.mean(np.array(sp) < 0.8)))
    np.savez(OUT, replay_t=replay_t, runs=rows)
    print("-> {}".format(os.path.relpath(OUT, ROOT)))


if __name__ == "__main__":
    main()
