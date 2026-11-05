"""Calibrate possession_v2 on external games, never on the BAR-ATH public five minutes.

The calibration set is the three Metrica Sports sample games (solution/external/metrica.py): same
tracking provider as BAR-ATH, labels derived from their event data. Every score here is the
challenge's own scorer (evaluation/possession_metrics.evaluate) run on those games.

    .venv/bin/python solution/calibrate.py baseline                 # current PARAMS, per game
    .venv/bin/python solution/calibrate.py sweep                    # one parameter at a time
    .venv/bin/python solution/calibrate.py sweep --only V_SLOW,R_FAR
    .venv/bin/python solution/calibrate.py ablate                   # each rule switched off
    .venv/bin/python solution/calibrate.py tracking                 # clean / degraded / fixed players

Scores are on DEGRADED tracking by default (external/degrade.py): the Metrica players are perfect,
BAR-ATH's are not, and parameters must suit the tracking we are actually scored on.

How to read a sweep: a change is worth keeping only if it helps on every game (or on two and is
flat on the third) and its neighbours help too -- a plateau, not a lone peak. The games are three
and their labels are a silver standard, so differences under ~0.02 spellF1 are noise.
"""

import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "external"))
import clean_tracking  # noqa: E402
import degrade  # noqa: E402
import metrica  # noqa: E402
import possession_metrics as pm  # noqa: E402
import possession_v2 as v2  # noqa: E402

KEYS = ("spell_f1_mean", "macro_f1", "alive_dead_f1", "team_acc_gt_alive", "spell_count_error")
_CASES = {}
TRACKING = "degraded"   # clean | degraded | fixed (degraded, then clean_tracking.fix_team_swaps)


SEEDS = (1, 2)          # two degraded copies of every game: 6 runs per variant
RUNS = tuple((g, s) for g in metrica.GAMES for s in SEEDS)


def case(run, variant=None):
    """(tracking arrays, graded blocks, GT codes) for one (game, seed) run, cached per process.

    degraded: players made BAR-ATH-like by external/degrade.py, the default, because BAR-ATH's
    tracking is the one we are scored on. fixed: the same, then cleaned. clean ignores the seed."""
    variant = variant or TRACKING
    game, seed = run
    if (run, variant) not in _CASES:
        teams, players, ball, ball_speed, periods = metrica.load(game)
        lab = metrica.labels(game, len(ball), periods)
        if variant in ("degraded", "fixed"):
            teams, players = degrade.degrade(teams, players, seed=10 * game + seed)
        if variant == "fixed":
            teams, players, _ = clean_tracking.fix_team_swaps(teams, players)
        _CASES[(run, variant)] = (teams, players, ball, ball_speed,
                                  metrica.blocks_from_labels(lab), pm.label_codes(lab))
    return _CASES[(run, variant)]


def score(run, params):
    teams, players, ball, ball_speed, blocks, gt = case(run, params.get("_tracking"))
    labels, _, _ = v2.predict(teams, players, ball, ball_speed, blocks, metrica.PITCH, params)
    r = pm.evaluate(gt, pm.label_codes(labels), bootstrap_n=0)
    return {k: r[k] for k in KEYS}


def _job(args):
    game, params = args
    return score(game, params)


def run_all(variants, games=RUNS, workers=None):
    """{name: {(game, seed): scores}} for every (name, params) variant, in parallel."""
    jobs = [(name, g, p) for name, p in variants for g in games]
    with Pool(workers or os.cpu_count()) as pool:
        res = pool.map(_job, [(g, p) for _, g, p in jobs])
    out = {}
    for (name, g, _), r in zip(jobs, res):
        out.setdefault(name, {})[g] = r
    return out


def row(name, per_game, ref=None):
    s = [per_game[g]["spell_f1_mean"] for g in sorted(per_game)]
    m = np.mean([per_game[g]["macro_f1"] for g in per_game])
    ad = np.mean([per_game[g]["alive_dead_f1"] for g in per_game])
    ta = np.mean([per_game[g]["team_acc_gt_alive"] for g in per_game])
    ce = np.mean([per_game[g]["spell_count_error"] for g in per_game])
    txt = "{:<36} spell {:.3f} [{}]  macro {:.3f}  alive/dead {:.3f}  team {:.3f}  count {:+.0f}".format(
        name, np.mean(s), " ".join("{:.3f}".format(x) for x in s), m, ad, ta, ce)
    if ref is not None:
        d = [a - b for a, b in zip(s, ref)]
        wins = sum(x > 0.005 for x in d)
        txt += "   d {:+.3f} ({}/{} up)".format(np.mean(d), wins, len(d))
    return txt, s


SWEEP_SCALES = (0.5, 0.75, 1.33, 2.0)
SKIP = {"GAP_RULE", "CORNER_RULE", "R_CORNER", "CORNER_STAND_S", "PRESS_R", "CORNER_AREA",
        "CORNER_BACK_S", "GAP_MIN", "OUT_MIN_FRAMES", "VEL_HALF", "DV_LAG", "TOUCH_NMS", "SPEED_LAG", "MIN_PLAYERS",
        "K_NEAR", "SPEED_MAX", "KICKOFF_MAX_S", "W_WHISTLE", "WHISTLE_LOOK_S", "WHISTLE_LEAD_S"}


def sweep_variants(only=None):
    base = dict(v2.PARAMS)
    names = only or [k for k in base if k not in SKIP and isinstance(base[k], (int, float))]
    out = []
    for k in names:
        for s in SWEEP_SCALES:
            p = dict(base)
            v = base[k] * s
            p[k] = int(round(v)) if isinstance(base[k], int) else v
            if p[k] == base[k]:
                continue
            out.append(("{} x{:g} ({:g})".format(k, s, p[k]), p))
    return out


ABLATIONS = {
    "no long-gap rule (rule 8)": {"GAP_RULE": 0},
    "no look-ahead (rule 4 v1)": {"W_LOOK": 0.0},
    "no touch (rule 2 v1)": {"W_TOUCH": 0.0, "W_TOUCH_OWN": 0.0},
    "no cause needed (rule 7)": {"NO_CAUSE_PENALTY": 0.0, "CAUSE_S": 0.0},
    "with corner kicks (rule 5)": {"CORNER_RULE": 1},
    "no braking (rule 3, distance only)": {"W_RHYTHM": 0.0},
    "no distance (rule 3 only)": {"W_RHYTHM": 1.0},
    "no min dead-ball duration": {"D_MIN_S": 0.0, "D_MIN_AFTER_S": 0.0},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("baseline", "sweep", "ablate", "tracking"))
    ap.add_argument("--only", help="comma-separated parameter names for sweep")
    a = ap.parse_args()

    base = run_all([("current PARAMS", dict(v2.PARAMS))])["current PARAMS"]
    txt, ref = row("current PARAMS", base)
    print(txt)
    if a.mode == "baseline":
        return
    if a.mode == "tracking":
        variants = [(v, {**v2.PARAMS, "_tracking": v}) for v in ("clean", "degraded", "fixed")]
        res = run_all(variants)
        for name, _ in variants:
            print(row("tracking " + name, res[name], ref)[0])
        return
    if a.mode == "sweep":
        variants = sweep_variants(a.only.split(",") if a.only else None)
    else:
        variants = [(name, {**v2.PARAMS, **ch}) for name, ch in ABLATIONS.items()]
    res = run_all(variants)
    lines = [row(name, res[name], ref) for name, _ in variants]
    for txt, _ in lines:
        print(txt, flush=True)


if __name__ == "__main__":
    main()
