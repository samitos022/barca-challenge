"""Force the team of every tracked player from what he does and from the frames before and after.

The tracker gives each track slot a team. It is sometimes wrong for a whole stretch: a goalkeeper
carried as an opposing forward, a player re-labelled in the middle of a run (on BAR-ATH a track
ends and, one frame later and at the same spot, a track of the OTHER team starts: 50-70 times).
This module decides the team again, per stretch of track, from three sources of evidence:

1. Time. Tracks are cut into pieces (a slot tracked without interruption and without a jump),
   and pieces that continue each other (next frames, same spot) are linked in a chain: one
   physical player, whatever the labels say. A player does not change team, so the pieces of a
   chain must agree unless the evidence says the link is wrong.
2. Context. A model looks at one player on one frame WITHOUT his label and says which team he
   plays for, from where he stands and moves with respect to the two teams as labelled around
   him: depth from each goal, rank inside each team's block, distance to each team's last man and
   defensive line, nearest players, pace with respect to each team, how many players each team
   already has, the ball. The same features averaged over the seconds before and after are
   added. Trained only on other matches (IDSSE, Metrica) dirtied like BAR-ATH, where the true
   team is known from the provider; symmetric in the two teams (it cannot learn a club).
3. The tracker's own label, as a prior: it is right most of the time, so changing a piece costs.

Decision: for each chain, the cheapest sequence of teams (dynamic programming over 5 s segments):
context evidence, plus a cost for contradicting the tracker on a piece, plus a cost for changing
team along the chain. Then the context is recomputed with the corrected teams and the decision
repeated (a keeper moved to his team changes what the others look like).

No hand labels, no frame-specific constants; costs chosen on the external matches only (cv).

    .venv/bin/python solution/team_reassign.py cv        # out-of-fold on the external matches
    .venv/bin/python solution/team_reassign.py train     # final model -> data/derived/team_reassign.pkl
    .venv/bin/python solution/team_reassign.py apply     # BAR-ATH: aggregate report, cache of the result
"""

import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE, os.path.join(HERE, "external"), os.path.join(ROOT, "starter"), os.path.join(ROOT, "evaluation")):
    if p not in sys.path:
        sys.path.insert(0, p)
from clean_tracking import MAX_PER_TEAM  # noqa: E402
from track_fixes import own_goals  # noqa: E402

FPS = 25
MODEL = os.path.join(ROOT, "data", "derived", "team_reassign.pkl")
CACHE = os.path.join(ROOT, "data", "derived", "{}_team_reassign.npz")

STEP = 5             # frames between evaluated frames (5 Hz)
TELEPORT = 1.0       # m in one frame (25 m/s): the slot jumped onto something else, the piece ends
LINK_GAP = 12        # frames: a piece continues one that ended at most this long before
LINK_D0 = 0.5        # m: distance allowed between the end of one and the start of the next ...
LINK_V = 0.1         # ... plus this per frame of gap
SEG_S = 5.0          # s: length of a decision segment
WIN_S = 5.0          # s: half-width of the "before and after" averages
LCLIP = 3.0          # log-odds: no single frame counts more than this
TRAIN_EVERY = 10     # of the evaluated frames kept for training (0.5 Hz)

# Costs in log-odds, chosen by cv() on the external matches.
PARAMS = {
    "TAU": 4.0,       # s of context worth one independent look
    "CAP_S": 10.0,    # s: a piece's context evidence stops growing after this (a role can look wrong
                      # for long: beyond it the decision rests on the mean, not on the length)
    "L_LABEL": 6.0,   # contradicting the tracker on a whole piece
    "L_LINK": 6.0,    # changing team between two linked pieces
    "L_IN": 18.0,     # changing team inside a piece
    "ITERS": 1,       # a second pass with the corrected teams lost recall out of fold
}
# cv (5 folds, 10 matches, the model never sees the match it corrects), positions on the wrong team:
#   standard 0.851% -> 0.036% (changes right 96.9%), heavy 2.951% -> 0.197% (96.7%),
#   keeper 1.985% -> 0.037% (98.7%); the old count-only fix leaves 0.849% / 2.904% / 2.213%.
#   First costs tried (TAU 2, CAP 60, 2 passes): 0.354% / 1.362% / 0.438%.

BASE = ("n", "u", "rank", "du_min", "du_2", "du_max", "du_mean", "dy", "yfrac", "dn1", "dn2", "dn3",
        "dv", "dvn", "bu")                    # one per team
SOLO = ("speed", "yc", "ball_d")              # no team
WINDOWED = ("n", "rank", "du_min", "du_2", "dn1", "u")


def feature_names():
    names = list(SOLO) + ["{}_{}".format(b, k) for k in (0, 1) for b in BASE]
    names += ["speed_w"] + ["{}_{}_w".format(b, k) for k in (0, 1) for b in WINDOWED]
    return names


def swap_index(names):
    """Column order of the same features with the two teams exchanged."""
    pos = {n: i for i, n in enumerate(names)}
    out = []
    for n in names:
        parts = n.split("_")
        for i, tok in enumerate(parts):
            if tok in ("0", "1") and i > 0:
                parts[i] = "1" if tok == "0" else "0"
                break
        out.append(pos["_".join(parts)])
    return np.array(out)


# ---- pieces and chains ---------------------------------------------------------------------------
def pieces(players, blocks, teleport=TELEPORT):
    """int array (m, 4): slot, first, stop, block index. Maximal runs of a slot inside a graded
    block, cut where the slot jumps more than `teleport` in one frame."""
    ok = ~np.isnan(players[..., 0])
    out = []
    for bi, (_, a0, z0) in enumerate(blocks):
        o = ok[a0:z0 + 1]
        P = players[a0:z0 + 1]
        for s in range(o.shape[1]):
            e = np.flatnonzero(np.diff(np.concatenate(([0], o[:, s].astype(np.int8), [0]))))
            for a, b in zip(e[::2], e[1::2]):
                if b - a > 1:
                    st = np.linalg.norm(np.diff(P[a:b, s], axis=0), axis=1)
                    cuts = a + 1 + np.flatnonzero(st > teleport)
                else:
                    cuts = np.array([], dtype=np.int64)
                edges = np.concatenate(([a], cuts, [b]))
                out += [(s, a0 + int(x), a0 + int(y), bi) for x, y in zip(edges[:-1], edges[1:])]
    return np.array(out, dtype=np.int64).reshape(-1, 4)


def link(pc, players, max_gap=LINK_GAP, d0=LINK_D0, dv=LINK_V):
    """(nxt, prv) piece indices, -1 if none: piece j ends, piece i starts within max_gap frames at
    the spot j was heading to (or was at). One to one, closest first."""
    m = len(pc)
    order = np.argsort(pc[:, 2], kind="stable")
    ends = pc[order, 2]
    cand = []
    for i in range(m):
        s, a, _, bi = pc[i]
        lo, hi = np.searchsorted(ends, [a - max_gap, a + 1])
        for j in order[lo:hi]:
            s0, a0, b0, bi0 = pc[j]
            if j == i or bi0 != bi:
                continue
            gap = a - b0
            p1 = players[b0 - 1, s0]
            k = max(a0, b0 - 6)
            v = (p1 - players[k, s0]) / max(b0 - 1 - k, 1)
            d = min(np.hypot(*(p1 + v * (gap + 1) - players[a, s])), np.hypot(*(p1 - players[a, s])))
            if d <= d0 + dv * gap:
                cand.append((float(d), int(j), i))
    cand.sort()
    nxt = np.full(m, -1, dtype=np.int64)
    prv = np.full(m, -1, dtype=np.int64)
    for _, j, i in cand:
        if nxt[j] < 0 and prv[i] < 0:
            nxt[j], prv[i] = i, j
    return nxt, prv


# ---- context features ----------------------------------------------------------------------------
def _fin(a):
    return np.where(np.isfinite(a), a, np.nan).astype(np.float32)


def _chunk(x, y, vx, vy, bx, by, T, ok, gx, W):
    n, S = x.shape
    eye = np.eye(S, dtype=bool)[None]
    f = {"speed": np.hypot(vx, vy), "yc": np.abs(y - W / 2), "ball_d": np.hypot(x - bx[:, None], y - by[:, None])}
    with np.errstate(invalid="ignore", divide="ignore"):
        dist = np.hypot(x[:, :, None] - x[:, None, :], y[:, :, None] - y[:, None, :])
        for k in (0, 1):
            dep = np.abs(x - gx[:, k, None])
            M = (ok & (T == k))[:, None, :] & ~eye           # (n, target, other)
            cnt = M.sum(2).astype(np.float32)
            c1 = np.where(cnt > 0, cnt, np.nan)
            depj = np.broadcast_to(dep[:, None, :], M.shape)
            lo2 = np.partition(np.where(M, depj, np.inf), 1, axis=2)[..., :2]
            f["n_%d" % k] = cnt
            f["u_%d" % k] = dep
            f["rank_%d" % k] = (M & (depj < dep[:, :, None])).sum(2) / c1
            f["du_min_%d" % k] = dep - lo2[..., 0]
            f["du_2_%d" % k] = dep - lo2[..., 1]
            f["du_max_%d" % k] = dep - np.where(M, depj, -np.inf).max(2)
            f["du_mean_%d" % k] = dep - np.where(M, depj, 0).sum(2) / c1
            yj = np.broadcast_to(y[:, None, :], M.shape)
            f["dy_%d" % k] = np.abs(y - np.where(M, yj, 0).sum(2) / c1)
            ymin, ymax = np.where(M, yj, np.inf).min(2), np.where(M, yj, -np.inf).max(2)
            f["yfrac_%d" % k] = (y - ymin) / (ymax - ymin)
            dd = np.where(M, dist, np.inf)
            near = np.sort(np.partition(dd, 2, axis=2)[..., :3], axis=2)
            for q in range(3):
                f["dn%d_%d" % (q + 1, k)] = near[..., q]
            vbx = np.where(M, vx[:, None, :], 0).sum(2) / c1
            vby = np.where(M, vy[:, None, :], 0).sum(2) / c1
            f["dv_%d" % k] = np.hypot(vx - vbx, vy - vby)
            j1 = np.argmin(dd, axis=2)
            f["dvn_%d" % k] = np.where(cnt > 0, np.hypot(vx - np.take_along_axis(vx, j1, 1),
                                                         vy - np.take_along_axis(vy, j1, 1)), np.nan)
            f["bu_%d" % k] = dep - np.abs(bx - gx[:, k])[:, None]
    return f


def context(team_of, players, ball, blocks, goals, pitch, pc, step=STEP, chunk=1500):
    """(fr, F, names): fr (n,) evaluated frames, F (n, slots, features) float32 (NaN where the slot
    is not tracked). team_of (N, slots): the team each slot is taken to have on each frame."""
    names = feature_names()
    col = {n: i for i, n in enumerate(names)}
    fr = np.concatenate([np.arange(a, z + 1, step) for _, a, z in blocks])
    gx = np.concatenate([np.repeat(np.array(goals[i], dtype=np.float32)[None], len(np.arange(a, z + 1, step)), axis=0)
                         for i, (_, a, z) in enumerate(blocks)])
    n, S = len(fr), players.shape[1]
    N = len(players)
    lo, hi = np.clip(fr - 2, 0, N - 1), np.clip(fr + 2, 0, N - 1)
    vel = np.nan_to_num((players[hi] - players[lo]) * (FPS / np.maximum(hi - lo, 1))[:, None, None], nan=0.0)
    F = np.full((n, S, len(names)), np.nan, dtype=np.float32)
    for c in range(0, n, chunk):
        sl = slice(c, min(n, c + chunk))
        P = players[fr[sl]]
        ok = ~np.isnan(P[..., 0])
        f = _chunk(P[..., 0], P[..., 1], vel[sl, :, 0], vel[sl, :, 1], ball[fr[sl], 0], ball[fr[sl], 1],
                   team_of[fr[sl]], ok, gx[sl], pitch[1])
        for k, v in f.items():
            F[sl, :, col[k]] = np.where(ok, _fin(v), np.nan)
    w = int(round(WIN_S * FPS / step))
    src = [col["speed"]] + [col["%s_%d" % (b, k)] for k in (0, 1) for b in WINDOWED]
    dst = [col["speed_w"]] + [col["%s_%d_w" % (b, k)] for k in (0, 1) for b in WINDOWED]
    for s, a, b, _ in pc:
        i0, i1 = np.searchsorted(fr, [a, b])
        if i1 <= i0:
            continue
        v = F[i0:i1, s][:, src]
        good = ~np.isnan(v)
        cs = np.concatenate((np.zeros((1, len(src))), np.cumsum(np.where(good, v, 0), axis=0)))
        cn = np.concatenate((np.zeros((1, len(src))), np.cumsum(good, axis=0)))
        idx = np.arange(i1 - i0)
        p, q = np.maximum(idx - w, 0), np.minimum(idx + w + 1, i1 - i0)
        with np.errstate(invalid="ignore", divide="ignore"):
            F[i0:i1, s][:, dst] = (cs[q] - cs[p]) / (cn[q] - cn[p])
    return fr, F, names


# ---- model ---------------------------------------------------------------------------------------
def fit(X, y, names, seed=0):
    """Team-0 classifier, trained on the rows and on their mirror (teams exchanged)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    sw = swap_index(names)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=63,
                                         min_samples_leaf=200, l2_regularization=1.0, random_state=seed)
    clf.fit(np.concatenate([X, X[:, sw]]), np.concatenate([y, 1 - y]))
    return clf


def logit_team0(clf, X, names):
    """log-odds that the player is of team 0, averaged over the two orderings of the teams."""
    sw = swap_index(names)
    p = 0.5 * (clf.predict_proba(X)[:, 1] + 1 - clf.predict_proba(X[:, sw])[:, 1])
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p)).astype(np.float32)


def frame_logits(clf, F, names):
    """(n, slots) log-odds of team 0, NaN where the slot is not tracked."""
    ok = ~np.isnan(F[..., names.index("u_0")])
    out = np.full(F.shape[:2], np.nan, dtype=np.float32)
    if ok.any():
        out[ok] = logit_team0(clf, F[ok], names)
    return out


# ---- decision ------------------------------------------------------------------------------------
def segments(pc, fr, lg, step=STEP):
    """Per piece: [(first, stop, mean clipped log-odds, evaluated frames)] over ~SEG_S segments."""
    per = max(1, int(round(SEG_S * FPS / step)))
    out = []
    for s, a, b, _ in pc:
        i0, i1 = np.searchsorted(fr, [a, b])
        if i1 <= i0:
            out.append([(int(a), int(b), 0.0, 0)])
            continue
        v = np.clip(np.nan_to_num(lg[i0:i1, s], nan=0.0), -LCLIP, LCLIP)
        cuts = list(range(0, i1 - i0, per))
        if len(cuts) > 1 and (i1 - i0) - cuts[-1] < per // 2:
            cuts.pop()                                     # a short tail joins the previous segment
        segs = []
        for q, c in enumerate(cuts):
            c2 = cuts[q + 1] if q + 1 < len(cuts) else i1 - i0
            f0 = int(a) if q == 0 else int(fr[i0 + c])
            f1 = int(b) if q + 1 == len(cuts) else int(fr[i0 + c2])
            segs.append((f0, f1, float(v[c:c2].mean()), c2 - c))
        out.append(segs)
    return out


def decide(pc, nxt, prv, label, segs, params=PARAMS, step=STEP):
    """Team of every segment: [(slot, first, stop, team)] and the per-piece summary
    (piece, frames changed). label (pieces,) the tracker's team of each piece."""
    tau, cap = params["TAU"], params["CAP_S"]
    out, changed = [], np.zeros(len(pc), dtype=np.int64)
    for head in np.flatnonzero(prv < 0):
        chain = []
        i = head
        while i >= 0:
            chain.append(i)
            i = nxt[i]
        cost, trans, meta = [], [], []
        for qi, i in enumerate(chain):
            dur = (pc[i, 2] - pc[i, 1]) / FPS
            scale = min(1.0, cap / max(dur, 1e-9)) * (step / FPS) / tau
            for qs, (f0, f1, m, nf) in enumerate(segs[i]):
                e = m * nf * scale                         # > 0: team 0
                lab = params["L_LABEL"] * (f1 - f0) / max(pc[i, 2] - pc[i, 1], 1)
                cost.append((-e / 2 + (lab if label[i] != 0 else 0.0), e / 2 + (lab if label[i] != 1 else 0.0)))
                if qs > 0:
                    trans.append(params["L_IN"])
                elif qi > 0:
                    trans.append(params["L_LINK"])
                meta.append((i, f0, f1))
        c = np.array(cost)
        best = np.zeros_like(c)
        back = np.zeros(c.shape, dtype=np.int8)
        best[0] = c[0]
        for t in range(1, len(c)):
            for k in (0, 1):
                stay, move = best[t - 1, k], best[t - 1, 1 - k] + trans[t - 1]
                back[t, k] = k if stay <= move else 1 - k
                best[t, k] = c[t, k] + min(stay, move)
        k = int(np.argmin(best[-1]))
        for t in range(len(c) - 1, -1, -1):
            i, f0, f1 = meta[t]
            out.append((int(pc[i, 0]), f0, f1, k))
            if k != label[i]:
                changed[i] += f1 - f0
            k = int(back[t, k])
    return out, changed


def materialise(teams, players, team_of):
    """fix_team_swaps layout: twice the slots, the frames taken to the other team in the copy."""
    n, slots = players.shape[:2]
    players2 = np.full((n, 2 * slots, 2), np.nan, dtype=players.dtype)
    moved = team_of != teams[None]
    players2[:, :slots][~moved] = players[~moved]
    players2[:, slots:][moved] = players[moved]
    return np.concatenate([teams, 1 - teams]).astype(teams.dtype), players2


def reassign(teams, players, ball, blocks, pitch, clf, params=PARAMS, return_info=False):
    """(teams2, players2) with the team forced where the evidence says the tracker is wrong, in
    the fix_team_swaps layout. With return_info also {team_of, pieces, changed, logit, fr}."""
    pc = pieces(players, blocks)
    nxt, prv = link(pc, players)
    label = teams[pc[:, 0]].astype(np.int64)
    goals = [own_goals(teams, players, a, z, pitch) for _, a, z in blocks]
    team_of = np.repeat(teams[None].astype(np.int8), len(players), axis=0)
    info = {}
    for _ in range(params["ITERS"]):
        fr, F, names = context(team_of, players, ball, blocks, goals, pitch, pc)
        lg = frame_logits(clf, F, names)
        del F
        out, changed = decide(pc, nxt, prv, label, segments(pc, fr, lg), params)
        team_of = np.repeat(teams[None].astype(np.int8), len(players), axis=0)
        for s, f0, f1, k in out:
            team_of[f0:f1, s] = k
        info = {"team_of": team_of, "pieces": pc, "changed": changed, "logit": lg, "fr": fr,
                "links": int((nxt >= 0).sum()),
                "cross_links": int(sum(label[i] != label[nxt[i]] for i in np.flatnonzero(nxt >= 0)))}
    teams2, players2 = materialise(teams, players, team_of)
    return (teams2, players2, info) if return_info else (teams2, players2)


# ---- external matches: regimes, cv, training ------------------------------------------------------
REGIMES = {
    "standard": {"swap": 0.009, "keeper": 0.0},     # degrade.py defaults (BAR-ATH's count histogram)
    "heavy": {"swap": 0.03, "keeper": 0.0},
    "keeper": {"swap": 0.009, "keeper": 0.12},      # + keepers carried by the other team for long stretches
    "train": {"swap": 0.02, "keeper": 0.12},
}
KEEPER_S = 90.0     # s, mean length of a keeper stretch (BAR-ATH: 3 stretches, 266 s, fix_keepers)


def keeper_slots(teams, players, blocks, pitch):
    """Per block, per team: the slot that stays deepest (tracked on more than half of the block)."""
    out = []
    for _, a, z in blocks:
        gx = own_goals(teams, players, a, z, pitch)
        row = []
        for k in (0, 1):
            cols = np.flatnonzero(teams == k)
            dep = np.abs(players[a:z + 1][:, cols, 0] - gx[k])
            share = (~np.isnan(dep)).mean(0)
            med = np.where(share > 0.5, np.nanmedian(np.where(np.isnan(dep), np.inf, dep), axis=0), np.inf)
            row.append(int(cols[int(np.argmin(med))]))
        out.append(row)
    return out


def dirty(teams, players, blocks, pitch, regime, seed):
    """(teams2, players2, truth): the clean tracking dirtied under a regime; truth (slots2,) is the
    real team of each slot of the result."""
    import degrade
    r = REGIMES[regime]
    base = len(teams)
    teams2, players2 = degrade.degrade(teams, players, seed=seed, swap=r["swap"])
    if r["keeper"]:
        rng = np.random.default_rng(seed + 7919)
        for (_, a, z), gk in zip(blocks, keeper_slots(teams, players, blocks, pitch)):
            for s in gk:
                t = a
                while t <= z:
                    length = max(1, int(rng.exponential(KEEPER_S * FPS)))
                    seg = slice(t, min(z + 1, t + length))
                    if rng.random() < r["keeper"]:
                        tmp = players2[seg, s].copy()
                        players2[seg, s] = players2[seg, s + base]
                        players2[seg, s + base] = tmp
                    t += length
    return teams2, players2, np.concatenate([teams, teams]).astype(np.int8)


def external_games():
    import idsse
    import metrica
    return [(g, "tracab") for g in idsse.GAMES] + [(g, None) for g in metrica.GAMES]


def clean_case(game, labels):
    """Clean players, BAR-ATH-like ball (gaps and shape), graded blocks, pitch."""
    import calibrate
    teams, players = calibrate.case((game, 1), "clean", labels)[:2]
    _, _, ball, _, blocks, _ = calibrate.case((game, 1), "realistic", labels)
    return teams, players, ball, blocks, calibrate.pitch_of((game, 1))


def _seed(game, labels):
    return 1000 + 17 * [g for g, _ in external_games()].index(game)


def _train_job(args):
    game, labels = args
    teams, players, ball, blocks, pitch = clean_case(game, labels)
    t2, p2, truth = dirty(teams, players, blocks, pitch, "train", _seed(game, labels))
    pc = pieces(p2, blocks)
    goals = [own_goals(t2, p2, a, z, pitch) for _, a, z in blocks]
    team_of = np.repeat(t2[None].astype(np.int8), len(p2), axis=0)
    fr, F, names = context(team_of, p2, ball, blocks, goals, pitch, pc)
    rng = np.random.default_rng(_seed(game, labels))
    keep = np.arange(int(rng.integers(TRAIN_EVERY)), len(fr), TRAIN_EVERY)
    F = F[keep]
    ok = ~np.isnan(F[..., names.index("u_0")])
    y = np.broadcast_to((truth == 0)[None], ok.shape)[ok].astype(np.int8)
    return F[ok], y


def measure(teams2, truth, players2, team_of, blocks):
    """Aggregates on the graded frames: wrong-team share before / after, precision and recall of
    the changes (in positions), share of team-frames over 11 before / after."""
    g = np.zeros(len(players2), bool)
    for _, a, z in blocks:
        g[a:z + 1] = True
    ok = ~np.isnan(players2[..., 0]) & g[:, None]
    lab = np.broadcast_to(teams2[None], ok.shape)
    tru = np.broadcast_to(truth[None], ok.shape)
    wrong0 = ok & (lab != tru)
    wrong1 = ok & (team_of != tru)
    ch = ok & (team_of != lab)
    over = []
    for T in (lab, team_of):
        c = np.stack([(ok & (T == k)).sum(1) for k in (0, 1)])[:, g]
        over.append(float(np.mean(c > MAX_PER_TEAM)))
    return {"positions": int(ok.sum()), "wrong_before": int(wrong0.sum()), "wrong_after": int(wrong1.sum()),
            "changed": int(ch.sum()), "changed_right": int((ch & ~wrong1).sum()),
            "fixed": int((wrong0 & ~wrong1).sum()), "over_before": over[0], "over_after": over[1]}


def _eval_job(args):
    game, labels, regime, model_path, grid = args
    with open(model_path, "rb") as fh:
        clf = pickle.load(fh)
    teams, players, ball, blocks, pitch = clean_case(game, labels)
    t2, p2, truth = dirty(teams, players, blocks, pitch, regime, _seed(game, labels) + 1)
    pc = pieces(p2, blocks)
    nxt, prv = link(pc, p2)
    label = t2[pc[:, 0]].astype(np.int64)
    goals = [own_goals(t2, p2, a, z, pitch) for _, a, z in blocks]
    base_of = np.repeat(t2[None].astype(np.int8), len(p2), axis=0)
    out = {}
    lg_first = None
    for name, params in grid:
        team_of = base_of
        for it in range(params["ITERS"]):
            if it == 0 and lg_first is not None:
                fr, lg = lg_first
            else:
                fr, F, names = context(team_of, p2, ball, blocks, goals, pitch, pc)
                lg = frame_logits(clf, F, names)
                del F
                if it == 0:
                    lg_first = (fr, lg)
            dec, _ = decide(pc, nxt, prv, label, segments(pc, fr, lg), params)
            team_of = base_of.copy()
            for s, f0, f1, k in dec:
                team_of[f0:f1, s] = k
        out[name] = measure(t2, truth, p2, team_of, blocks)
    # per-frame accuracy of the context alone, and the two older fixes for reference
    fr, lg = lg_first
    okf = ~np.isnan(lg)
    acc = float(np.mean((lg[okf] > 0) == np.broadcast_to((truth == 0)[None], lg.shape)[okf]))
    import clean_tracking
    ref = {}
    tt, pp, _ = clean_tracking.fix_team_swaps(t2, p2)
    S = len(t2)
    tof = np.where(~np.isnan(pp[:, S:, 0]), 1 - t2[None], t2[None]).astype(np.int8)
    ref["fix_team_swaps"] = measure(t2, truth, p2, tof, blocks)
    return game, regime, out, acc, ref


def _sum(rows):
    keys = ("positions", "wrong_before", "wrong_after", "changed", "changed_right", "fixed")
    t = {k: sum(r[k] for r in rows) for k in keys}
    t["over_before"] = float(np.mean([r["over_before"] for r in rows]))
    t["over_after"] = float(np.mean([r["over_after"] for r in rows]))
    return t


def _line(name, t):
    return ("{:28s} wrong team {:.3f}% -> {:.3f}%   changes right {:5.1f}%   wrong ones fixed {:5.1f}%   "
            "over 11: {:.2f}% -> {:.2f}%").format(
        name, 100 * t["wrong_before"] / t["positions"], 100 * t["wrong_after"] / t["positions"],
        100 * t["changed_right"] / max(t["changed"], 1), 100 * t["fixed"] / max(t["wrong_before"], 1),
        100 * t["over_before"], 100 * t["over_after"])


def grid_default():
    g = [("default", dict(PARAMS))]
    for k, vals in (("L_LABEL", (3.0, 10.0)), ("L_LINK", (3.0, 10.0)), ("TAU", (1.0, 4.0)),
                    ("CAP_S", (20.0, 180.0)), ("L_IN", (9.0, 1e9)), ("ITERS", (1,))):
        for v in vals:
            g.append(("{}={:g}".format(k, v), dict(PARAMS, **{k: v})))
    return g


def cv(folds=5, workers=6, grid=None):
    """Out-of-fold: the model never sees the match it corrects. Prints the aggregates per regime."""
    from multiprocessing import Pool
    games = external_games()
    grid = grid or grid_default()
    with Pool(workers) as pool:
        data = pool.map(_train_job, games)
    names = feature_names()
    results = []
    tmp = os.path.join(ROOT, "data", "derived", "team_reassign_cv")
    os.makedirs(tmp, exist_ok=True)
    for f in range(folds):
        test = [i for i in range(len(games)) if i % folds == f]
        X = np.concatenate([data[i][0] for i in range(len(games)) if i not in test])
        y = np.concatenate([data[i][1] for i in range(len(games)) if i not in test])
        clf = fit(X, y, names)
        path = os.path.join(tmp, "fold{}.pkl".format(f))
        with open(path, "wb") as fh:
            pickle.dump(clf, fh)
        jobs = [(games[i][0], games[i][1], r, path, grid) for i in test for r in ("standard", "heavy", "keeper")]
        with Pool(min(workers, len(jobs))) as pool:
            results += pool.map(_eval_job, jobs)
        print("fold {} done: test games {}, {} training rows".format(f, [games[i][0] for i in test], len(y)), flush=True)
    print("\ncontext alone, per evaluated frame: right team on {:.1f}% of the positions".format(
        100 * np.mean([r[3] for r in results])))
    for regime in ("standard", "heavy", "keeper"):
        rr = [r for r in results if r[1] == regime]
        print("\n== regime {} ({} matches)".format(regime, len(rr)))
        print(_line("fix_team_swaps (old)", _sum([r[4]["fix_team_swaps"] for r in rr])))
        for name, _ in grid:
            print(_line(name, _sum([r[2][name] for r in rr])))
    return results


GRID = ((4.0,), (5.0, 10.0, 15.0), (4.0, 6.0, 8.0), (3.0, 6.0), (18.0,))   # TAU, CAP_S, L_LABEL, L_LINK, L_IN


def cv_grid(grid, folds=5, workers=6):
    """Same out-of-fold report as cv() for other costs, with the fold models cv() saved."""
    from multiprocessing import Pool
    games = external_games()
    tmp = os.path.join(ROOT, "data", "derived", "team_reassign_cv")
    jobs = [(g, lab, r, os.path.join(tmp, "fold{}.pkl".format(i % folds)), grid)
            for i, (g, lab) in enumerate(games) for r in ("standard", "heavy", "keeper")]
    with Pool(workers) as pool:
        results = pool.map(_eval_job, jobs)
    tot = {}
    for regime in ("standard", "heavy", "keeper"):
        rr = [r for r in results if r[1] == regime]
        for name, _ in grid:
            tot.setdefault(name, []).append(_sum([r[2][name] for r in rr]))
    order = sorted(tot, key=lambda k: sum(t["wrong_after"] / t["positions"] for t in tot[k]))
    for name in order:
        print("{:44s} ".format(name) + "   ".join(
            "{}: {:.3f}% -> {:.3f}%, right {:4.1f}%, over 11 {:.2f}%".format(
                reg[:4], 100 * t["wrong_before"] / t["positions"], 100 * t["wrong_after"] / t["positions"],
                100 * t["changed_right"] / max(t["changed"], 1), 100 * t["over_after"])
            for reg, t in zip(("standard", "heavy", "keeper"), tot[name])), flush=True)
    return results


def train(workers=6):
    from multiprocessing import Pool
    with Pool(workers) as pool:
        data = pool.map(_train_job, external_games())
    X = np.concatenate([d[0] for d in data])
    y = np.concatenate([d[1] for d in data])
    names = feature_names()
    clf = fit(X, y, names)
    with open(MODEL, "wb") as fh:
        pickle.dump({"clf": clf, "names": names, "params": PARAMS}, fh)
    print("trained on {} rows of {} external matches -> {}".format(len(y), len(data), MODEL))


def load_model():
    if not os.path.exists(MODEL):
        raise FileNotFoundError("{} missing: run solution/team_reassign.py train".format(MODEL))
    with open(MODEL, "rb") as fh:
        return pickle.load(fh)["clf"]


def apply(game=None, base="tactical_nofrozen"):
    """BAR-ATH: re-assign, print aggregates only, cache the per-frame team of each slot.

    Decided on the tracking without the positions frozen out of view (track_fixes.drop_frozen):
    the external matches the model learnt from have no such ghosts, and a ghost moved to another
    team only makes that team 12. Slots are the same as the plain tactical tracking's."""
    import tracking
    from games import GAME, GAMES, half_video_blocks
    import track_fixes as tf
    game = game or GAME
    teams, players, ball, _, _ = tracking.load(game, base)
    blocks = half_video_blocks(game)
    pitch = GAMES[game]["pitch"]
    t2, p2, info = reassign(teams, players, ball, blocks, pitch, load_model(), return_info=True)
    np.savez_compressed(CACHE.format(game), team_of=info["team_of"], base=np.array(base))
    pc, ch = info["pieces"], info["changed"]
    g = np.zeros(len(players), bool)
    for _, a, z in blocks:
        g[a:z + 1] = True
    ok = ~np.isnan(players[..., 0]) & g[:, None]
    moved = ok & (info["team_of"] != teams[None])
    dur = (pc[:, 2] - pc[:, 1]) / FPS
    full = ch == (pc[:, 2] - pc[:, 1])
    print("{} ({}): {} pieces, {} links between pieces ({} between slots of different teams)".format(
        game, base, len(pc), info["links"], info["cross_links"]))
    print("  positions moved to the other team: {:.2f}% ({:.0f} s of player time)".format(
        100 * moved.sum() / ok.sum(), moved.sum() / FPS))
    print("  pieces changed: {} whole, {} in part; whole ones last {:.0f} s (median), {:.0f} s (total)".format(
        int(full.sum()), int(((ch > 0) & ~full).sum()), float(np.median(dur[full])) if full.any() else 0.0,
        float(dur[full].sum())))
    for name, (tt, pp) in (("before", (teams, players)), ("after", (t2, p2))):
        st = tf.team_count_stats(tt, pp, g)
        print("  {:6s} team with 12+: {:.1f}% / {:.1f}%   exactly 11: {:.1f}% / {:.1f}%".format(
            name, 100 * st[0][0], 100 * st[1][0], 100 * st[0][2], 100 * st[1][2]))
    return t2, p2, info


def cached(game, teams, players):
    """(teams2, players2) from the cache written by apply()."""
    path = CACHE.format(game)
    if not os.path.exists(path):
        raise FileNotFoundError("{} missing: run solution/team_reassign.py apply".format(path))
    return materialise(teams, players, np.load(path)["team_of"])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "cv"
    if cmd == "cv":
        cv()
    elif cmd == "grid":
        import itertools
        cv_grid([("tau {:g} cap {:g} label {:g} link {:g} in {:g}".format(*v),
                  dict(PARAMS, TAU=v[0], CAP_S=v[1], L_LABEL=v[2], L_LINK=v[3], L_IN=v[4], ITERS=1))
                 for v in itertools.product(*GRID)])
    elif cmd == "train":
        train()
    elif cmd == "apply":
        apply()
    else:
        raise SystemExit(__doc__)
