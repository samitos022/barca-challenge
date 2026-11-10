"""Full-match benchmark of possession_v2 on external games (IDSSE Bundesliga, Metrica samples).

Never touches BAR-ATH. For every run (game, seed) and tracking condition it runs v2 with the
current PARAMS (no whistle: there is no audio) and the challenge's scorer, and prints:
  1. per match: spellF1, macroF1, predicted / true possessions, predicted / true D share;
  2. the reference baseline (starter/baseline.py: nearest player + majority vote +-50 frames);
  3. where v2's dead balls come from: predicted D frames by the rule that entered them, with the
     share that is really D (precision), and the true D v2 misses;
  4. ablations of the dead-ball rules: D share and spellF1 with each rule switched off.

    .venv/bin/python analisi/bench_esterno.py                    # IDSSE, all of it
    .venv/bin/python analisi/bench_esterno.py --games all        # + Metrica
    .venv/bin/python analisi/bench_esterno.py --only main,ball   # sections: main, base, why, ablate, ball

Conditions: clean (as delivered), players (external/degrade.degrade), degraded (players and, on
IDSSE, the ball: degrade.degrade_ball). Metrica has no "players" condition distinct from its
"degraded" one, because its ball is already BAR-ATH-like.

--kickoff-fix: TRACAB's halves start at the kick-off or just after it (no tracking before it,
unlike BAR-ATH), often with the ball already more than KICKOFF_R from the centre spot; v2 then
waits KICKOFF_MAX_S (300 s) in D. The option sets KICKOFF_MAX_S = 0 on IDSSE runs only, a harness
correction for that artifact, not a proposal for PARAMS.
"""

import argparse
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "solution"), os.path.join(ROOT, "solution", "external"),
          os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "starter")):
    sys.path.insert(0, p)
import calibrate  # noqa: E402
import degrade  # noqa: E402
import idsse  # noqa: E402
import possession_metrics as pm  # noqa: E402
import possession_v2 as v2  # noqa: E402
from baseline import smooth  # noqa: E402

FPS = 25
CODE_D = 2  # pm.LABELS = (H, A, D, X)
ABLATE = {
    "rule 8 off (GAP_RULE 0)": {"GAP_RULE": 0},
    "rule 8 without support (GAP_SUPPORT 0)": {"GAP_SUPPORT": 0},
    "rule 9 off (RESTART_MOVE 0)": {"RESTART_MOVE": 0},
    "rule 10 off (LIGHTNING 0)": {"LIGHTNING": 0},
    "corner rule on (CORNER_RULE 1)": {"CORNER_RULE": 1},
    "dead score x0.5 (RATE_DEAD)": {"RATE_DEAD": v2.PARAMS["RATE_DEAD"] * 0.5},
    "dead threshold x2 (THRESHOLD_DEAD)": {"THRESHOLD_DEAD": v2.PARAMS["THRESHOLD_DEAD"] * 2},
    "no-cause penalty x4 (NO_CAUSE_PENALTY)": {"NO_CAUSE_PENALTY": v2.PARAMS["NO_CAUSE_PENALTY"] * 4},
    "braking only (W_RHYTHM 1)": {"W_RHYTHM": 1.0},
    "distance only (W_RHYTHM 0)": {"W_RHYTHM": 0.0},
    "rules 2-4 off (RATE_DEAD 0)": {"RATE_DEAD": 0.0},
    "min dead duration x2 (D_MIN_S, D_MIN_AFTER_S)": {"D_MIN_S": v2.PARAMS["D_MIN_S"] * 2,
                                                       "D_MIN_AFTER_S": v2.PARAMS["D_MIN_AFTER_S"] * 2},
    "alive rate x2 (ALIVE_RATE)": {"ALIVE_RATE": v2.PARAMS["ALIVE_RATE"] * 2},
}


def _reasons(labels, changes):
    """Per predicted D run, the reason of the change that entered it."""
    dead = [(start, reason) for _, start, new, reason in changes if new == v2.DEAD]
    starts = np.array([s for s, _ in dead]) if dead else np.zeros(0, int)
    d = labels == "D"
    idx = np.flatnonzero(np.diff(np.concatenate(([0], d.astype(np.int8), [0]))))
    out = []
    for a, b in zip(idx[::2], idx[1::2]):
        inside = np.flatnonzero((starts >= a) & (starts < b))
        if len(inside):
            reason = dead[inside[0]][1]
        else:
            before = np.flatnonzero(starts < a)
            reason = dead[before[-1]][1] if len(before) else "half start"
        if a in [blk[1] for blk in BLOCKS_CUR]:
            reason = "half start (before kick-off)"
        out.append((a, b, reason))
    return out


BLOCKS_CUR = []


def job(args):
    run, cond, name, change, extra = args
    global BLOCKS_CUR
    teams, players, ball, ball_speed, blocks, gt = calibrate.case(run, cond)
    BLOCKS_CUR = blocks
    graded = gt != 3
    res = {"run": run, "cond": cond, "name": name}
    if name == "baseline":
        d = np.linalg.norm(players - ball[:, None, :], axis=2)
        d = np.where(np.isnan(d), np.inf, d)
        best = np.argmin(d, axis=1)
        codes = np.where(np.isfinite(d.min(axis=1)), teams[best], -1).astype(np.int8)
        lab = smooth(codes, 50)
        lab[~graded] = "X"
    else:
        p = {**v2.PARAMS, **extra, **change}
        lab, f, r = v2.predict(teams, players, ball, ball_speed, blocks, calibrate.pitch_of(run), p)
        if name == "main":
            gd = gt == CODE_D
            why = defaultdict(lambda: [0, 0])
            for a, b, reason in _reasons(lab, r["changes"]):
                why[reason][0] += b - a
                why[reason][1] += int(gd[a:b].sum())
            res["why"] = dict(why)
            res["missed_d"] = int(np.sum(gd & graded & (lab != "D")))
            # where the excess lives: inside / outside the ball's gaps
            seen = ~np.isnan(ball[:, 0])
            pdead = (lab == "D") & graded
            res["pd_gap"] = int(np.sum(pdead & ~seen)), int(np.sum(pdead & ~seen & gd))
            res["pd_seen"] = int(np.sum(pdead & seen)), int(np.sum(pdead & seen & gd))
            res["gd_gap"] = int(np.sum(gd & ~seen))
    pc = pm.label_codes(lab)
    e = pm.evaluate(gt, pc, bootstrap_n=0)
    res.update(spell=e["spell_f1_mean"], macro=e["macro_f1"], n_gt=e["n_gt_spells"],
               n_pred=e["n_pred_spells"], d_pred=float(np.mean(pc[graded] == CODE_D)),
               d_gt=float(np.mean(gt[graded] == CODE_D)), graded=int(graded.sum()),
               d_prec=float(np.mean(gt[graded & (pc == CODE_D)] == CODE_D)) if np.any(graded & (pc == CODE_D)) else np.nan,
               d_rec=float(np.mean(pc[graded & (gt == CODE_D)] == CODE_D)))
    return res


def ball_job(run):
    """Ball statistics of the degraded IDSSE ball against BAR-ATH's (degrade.BALL_TARGETS)."""
    game, seed = run
    teams, players, ball, _, blocks, gt = calibrate.case(run, "degraded")
    _, _, clean_ball, _, _ = idsse.load(game)
    graded = gt != 3
    st = degrade.ball_stats(ball, players, graded, idsse.pitch(game))
    st0 = degrade.ball_stats(clean_ball, players, graded, idsse.pitch(game))
    st["seen_clean"] = 1 - st0["missing"]
    st["pace_all"] = float(np.nanmean(degrade.pace(players)[graded]))
    # not a target, only a diagnostic: how much of the removed ball falls on true dead balls
    gap = np.isnan(ball[:, 0]) & ~np.isnan(clean_ball[:, 0]) & graded
    st["gap_on_d"] = float(np.mean(gt[gap] == CODE_D))
    st["d_in_gap"] = float(np.mean(gap[graded & (gt == CODE_D)]))
    return run, st


def runs_of(games):
    return calibrate.GAME_SETS[games]


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(xs)) if xs else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", choices=("idsse", "metrica", "all"), default="idsse")
    ap.add_argument("--only", default="ball,main,base,why,ablate")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count()))
    ap.add_argument("--kickoff-fix", action="store_true", help="KICKOFF_MAX_S = 0 on IDSSE (see above)")
    a = ap.parse_args()

    def extra(run):
        return {"KICKOFF_MAX_S": 0.0} if a.kickoff_fix and run[0] in idsse.GAMES else {}
    only = set(a.only.split(","))
    runs = runs_of(a.games)
    conds = {}
    for run in runs:
        if run[0] in idsse.GAMES:
            conds[run] = ("clean", "players", "degraded") if run[1] == calibrate.SEEDS[0] else ("players", "degraded")
        else:
            conds[run] = ("clean", "degraded") if run[1] == calibrate.SEEDS[0] else ("degraded",)

    with Pool(a.workers) as pool:
        if "ball" in only and any(r[0] in idsse.GAMES for r in runs):
            ir = [r for r in runs if r[0] in idsse.GAMES]
            print("\n== ball degradation (IDSSE, players + ball degraded) vs BAR-ATH targets")
            keys = ("missing", "median_gap_s", "share_30s", "share_5s", "gaps_per_106min", "pace_gap",
                    "pace_seen", "last_line", "last_box", "last_else")
            print("{:<12}".format("run") + "".join("{:>10}".format(k[:10]) for k in keys) +
                  "{:>9}{:>9}{:>9}".format("seen0", "gap_on_D", "D_in_gap"))
            allst = []
            for run, st in pool.imap(ball_job, ir):
                allst.append(st)
                print("{:<12}".format("{} s{}".format(*run)) + "".join("{:>10.2f}".format(st[k]) for k in keys) +
                      "{:>9.3f}{:>9.2f}{:>9.2f}".format(st["seen_clean"], st["gap_on_d"], st["d_in_gap"]), flush=True)
            print("{:<12}".format("mean") + "".join("{:>10.2f}".format(mean([s[k] for s in allst])) for k in keys) +
                  "{:>9.3f}{:>9.2f}{:>9.2f}".format(*(mean([s[k] for s in allst]) for k in ("seen_clean", "gap_on_d", "d_in_gap"))))
            t = degrade.BALL_TARGETS
            print("{:<12}".format("BAR-ATH") + "".join("{:>10.2f}".format(t[k]) if k in t else "{:>10}".format("1.00")
                                                     for k in keys))

        jobs = []
        if "main" in only or "why" in only:
            jobs += [(r, c, "main", {}, extra(r)) for r in runs for c in conds[r]]
        if "base" in only:
            jobs += [(r, c, "baseline", {}, {}) for r in runs for c in conds[r]]
        if "ablate" in only:
            jobs += [(r, c, name, ch, extra(r)) for r in runs for c in conds[r] if c != "players"
                     for name, ch in ABLATE.items()]
        jobs.sort(key=lambda j: (str(j[0]), j[1]))
        res = list(pool.imap(job, jobs, chunksize=1))

    by = defaultdict(list)
    for x in res:
        by[(x["name"], x["cond"])].append(x)
    games = sorted({r[0] for r in runs}, key=str)
    all_conds = ("clean", "players", "degraded")

    def table(name, title):
        print("\n== " + title)
        print("{:<9}{:<10}{:>8}{:>8}{:>12}{:>16}".format("cond", "game", "spellF1", "macroF1", "poss p/t", "D share p/t"))
        for c in all_conds:
            rows = by.get((name, c), [])
            if not rows:
                continue
            for g in games:
                rr = [x for x in rows if x["run"][0] == g]
                if not rr:
                    continue
                print("{:<9}{:<10}{:>8.3f}{:>8.3f}{:>7.0f}/{:<4.0f}{:>9.1f}/{:<5.1f}".format(
                    c, str(g), mean([x["spell"] for x in rr]), mean([x["macro"] for x in rr]),
                    mean([x["n_pred"] for x in rr]), mean([x["n_gt"] for x in rr]),
                    100 * mean([x["d_pred"] for x in rr]), 100 * mean([x["d_gt"] for x in rr])))
            for label, sel in (("IDSSE", lambda x: x["run"][0] in idsse.GAMES),
                               ("Metrica", lambda x: x["run"][0] not in idsse.GAMES)):
                rr = [x for x in rows if sel(x)]
                if not rr:
                    continue
                print("{:<9}{:<10}{:>8.3f}{:>8.3f}{:>7.0f}/{:<4.0f}{:>9.1f}/{:<5.1f}   D prec {:.2f} rec {:.2f}  ({} runs)".format(
                    c, "MEAN " + label, mean([x["spell"] for x in rr]), mean([x["macro"] for x in rr]),
                    mean([x["n_pred"] for x in rr]), mean([x["n_gt"] for x in rr]),
                    100 * mean([x["d_pred"] for x in rr]), 100 * mean([x["d_gt"] for x in rr]),
                    mean([x["d_prec"] for x in rr]), mean([x["d_rec"] for x in rr]), len(rr)))

    if "main" in only:
        table("main", "possession_v2, current PARAMS, no whistle")
    if "base" in only:
        table("baseline", "reference baseline (nearest player, majority vote +-50 frames)")
    if "why" in only:
        for label, sel in (("IDSSE", lambda x: x["run"][0] in idsse.GAMES),
                           ("Metrica", lambda x: x["run"][0] not in idsse.GAMES)):
            for c in all_conds:
                rr = [x for x in by.get(("main", c), []) if sel(x)]
                if not rr:
                    continue
                tot = sum(x["graded"] for x in rr)
                print("\n== where v2's D comes from: {} {} (share of graded frames; precision = really D)".format(label, c))
                why = defaultdict(lambda: [0, 0])
                for x in rr:
                    for k, (fr, ok) in x["why"].items():
                        why[k][0] += fr
                        why[k][1] += ok
                for k, (fr, ok) in sorted(why.items(), key=lambda kv: -kv[1][0]):
                    print("   {:<34} D {:5.1f}%  precision {:.2f}  false D {:5.1f}%".format(
                        k, 100 * fr / tot, ok / max(fr, 1), 100 * (fr - ok) / tot))
                pg = sum(x["pd_gap"][0] for x in rr), sum(x["pd_gap"][1] for x in rr)
                ps = sum(x["pd_seen"][0] for x in rr), sum(x["pd_seen"][1] for x in rr)
                print("   predicted D in ball gaps {:.1f}% (precision {:.2f}), with the ball seen {:.1f}% (precision {:.2f});"
                      " true D missed {:.1f}%".format(100 * pg[0] / tot, pg[1] / max(pg[0], 1), 100 * ps[0] / tot,
                                                       ps[1] / max(ps[0], 1), 100 * sum(x["missed_d"] for x in rr) / tot))
    if "ablate" in only:
        for label, sel in (("IDSSE", lambda x: x["run"][0] in idsse.GAMES),
                           ("Metrica", lambda x: x["run"][0] not in idsse.GAMES)):
            for c in ("clean", "degraded"):
                ref = [x for x in by.get(("main", c), []) if sel(x)]
                if not ref:
                    continue
                print("\n== ablations, {} {}: D share pred (true {:.1f}%), D precision / recall, spellF1".format(
                    label, c, 100 * mean([x["d_gt"] for x in ref])))
                rs = {x["run"]: x["spell"] for x in ref}
                print("   {:<46} D {:5.1f}%  prec {:.2f} rec {:.2f}  spell {:.3f}".format(
                    "current PARAMS", 100 * mean([x["d_pred"] for x in ref]), mean([x["d_prec"] for x in ref]),
                    mean([x["d_rec"] for x in ref]), mean([x["spell"] for x in ref])))
                for name in ABLATE:
                    rr = [x for x in by.get((name, c), []) if sel(x)]
                    if not rr:
                        continue
                    d = [x["spell"] - rs[x["run"]] for x in rr]
                    print("   {:<46} D {:5.1f}%  prec {:.2f} rec {:.2f}  spell {:.3f}  d {:+.3f} ({}/{} up)".format(
                        name, 100 * mean([x["d_pred"] for x in rr]), mean([x["d_prec"] for x in rr]),
                        mean([x["d_rec"] for x in rr]), mean([x["spell"] for x in rr]), np.mean(d),
                        sum(v > 0.005 for v in d), len(d)))


if __name__ == "__main__":
    main()
