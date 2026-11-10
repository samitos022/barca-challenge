"""What team re-assignment is worth for possession, on the external matches (never BAR-ATH).

The same possession_v2 on the same dirtied tracking, with the teams: as the tracker gives them,
after the old count-only fix (clean_tracking.fix_team_swaps), after team_reassign (model of the
fold that never saw the match: run `solution/team_reassign.py cv` first), and with the true teams
(the most any team fix could give). Two regimes: standard (degrade.py) and keeper (keepers carried
by the other team for long stretches, as on BAR-ATH).

    .venv/bin/python analisi/bench_riassegna.py
"""

import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "solution"), os.path.join(ROOT, "solution", "external"),
          os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "starter")):
    sys.path.insert(0, p)
import calibrate  # noqa: E402
import clean_tracking  # noqa: E402
import idsse  # noqa: E402
import possession_metrics as pm  # noqa: E402
import possession_v2 as v2  # noqa: E402
import team_reassign as tr  # noqa: E402

FOLDS = 5
VARIANTS = ("tracker", "fix_team_swaps", "reassigned", "true teams")


def job(args):
    i, game, labels, regime = args
    teams, players, ball, blocks, pitch = tr.clean_case(game, labels)
    gt = calibrate.case((game, 1), "realistic", labels)[5]
    t2, p2, truth = tr.dirty(teams, players, blocks, pitch, regime, tr._seed(game, labels) + 1)
    with open(os.path.join(ROOT, "data", "derived", "team_reassign_cv", "fold{}.pkl".format(i % FOLDS)), "rb") as fh:
        clf = pickle.load(fh)
    p = dict(v2.PARAMS)
    if game in idsse.GAMES:
        p["KICKOFF_MAX_S"] = 0
    bs = np.full(len(ball), np.nan, np.float32)
    out = {}
    for name in VARIANTS:
        if name == "tracker":
            tt, pp = t2, p2
        elif name == "fix_team_swaps":
            tt, pp, _ = clean_tracking.fix_team_swaps(t2, p2)
        elif name == "reassigned":
            tt, pp = tr.reassign(t2, p2, ball, blocks, pitch, clf)
        else:
            tt, pp = tr.materialise(t2, p2, np.repeat(truth[None], len(p2), axis=0))
        lab, _, _ = v2.predict(tt, pp, ball, bs, blocks, pitch, p)
        out[name] = pm.evaluate(gt, pm.label_codes(lab), bootstrap_n=0)["spell_f1_mean"]
    return game, regime, out


def main():
    games = tr.external_games()
    jobs = [(i, g, lab, r) for r in ("standard", "keeper") for i, (g, lab) in enumerate(games)]
    with Pool(6) as pool:
        R = pool.map(job, jobs)
    for regime in ("standard", "keeper"):
        for sname, pick in (("DFL", lambda g: g in idsse.GAMES), ("Metrica", lambda g: g not in idsse.GAMES)):
            rr = [r[2] for r in R if r[1] == regime and pick(r[0])]
            print("{:9s} {:8s} ({} matches) spellF1:  ".format(regime, sname, len(rr)) + "   ".join(
                "{} {:.3f}".format(v, np.mean([x[v] for x in rr])) for v in VARIANTS), flush=True)


if __name__ == "__main__":
    main()
