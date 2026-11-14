"""The older versions on the matches that did not exist when they were made: SkillCorner (20) and
the 10 PFF World Cup matches held out in cv3. None of these models saw those matches.

    .venv/bin/python analisi/classifica_versioni.py
"""

import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "starter", "solution", "solution/external", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, d))
import calibrate  # noqa: E402
import pff  # noqa: E402
import possession_metrics as pm  # noqa: E402
import possession_ml as ml  # noqa: E402
import possession_v2 as v2  # noqa: E402
import skillcorner  # noqa: E402

MORNING = {"KEEPER_RULE": 0, "LAYOUT_RULE": 0, "DIST_ENTRY": 1}     # v2 as it was before 10/11


def job(game):
    run = (game, 1)
    teams, players, ball, bs, blocks, gt = calibrate.case(run, "realistic")
    pitch = calibrate.pitch_of(run)
    p = dict(v2.PARAMS, KICKOFF_MAX_S=0)
    lab, f, r = v2.predict(teams, players, ball, bs, blocks, pitch, p)
    codes = pm.label_codes(lab)
    X3, names = ml.features3(teams, players, ball, bs, blocks, pitch, codes, f, r)
    old = pm.label_codes(v2.predict(teams, players, ball, bs, blocks, pitch, dict(p, **MORNING))[0])
    return game, X3, names, gt, codes, old, blocks


def main():
    games = list(skillcorner.GAMES) + list(pff.GAMES[:10])
    with Pool(5) as pool:
        data = pool.map(job, games)
    names = data[0][2]
    D = os.path.join(ROOT, "data", "derived")
    team = pickle.load(open(os.path.join(D, "possession_ml.pkl"), "rb"))
    first = pickle.load(open(os.path.join(D, "possession_ml3_primo.pkl"), "rb"))
    moved = pickle.load(open(os.path.join(D, "possession_ml3_senza_skillcorner.pkl"), "rb"))
    nt = len(team["names"])
    km = len(moved["names"])     # features3 grew the keeper block (12/11) after this model
    assert names[:nt] == team["names"] and names[:len(first["names"])] == first["names"] and names[:km] == moved["names"]
    res = {}
    for game, X3, _, gt, codes, old, blocks in data:
        out = {"v2 of this morning": old, "v2 now": codes}
        ph = ml.prob_home(team["clf"], X3[:, :nt], team["names"])
        out["v2 + learnt team"] = ml.decode(ph, codes, team["switch_cost"])
        k = len(first["names"])
        out["three states, first"] = ml.decode3(ml.prob3(first["clf"], X3[:, :k], first["names"]), blocks, first["cost_team"], first["cost_dead"])
        out["three states + moved ball"] = ml.decode3(ml.prob3(moved["clf"], X3[:, :km], moved["names"]), blocks, moved["cost_team"], moved["cost_dead"])
        for k_, c in out.items():
            e = pm.evaluate(gt, c, bootstrap_n=0)
            res.setdefault(k_, []).append((game, e["spell_f1_mean"], e["macro_f1"], float(np.mean(c[gt != 3] == gt[gt != 3]))))
        print("done", game, flush=True)
    for sname, sel in (("SkillCorner (20)", set(skillcorner.GAMES)), ("PFF held out (10)", set(pff.GAMES[:10]))):
        print("\n== " + sname)
        for k_, rows in res.items():
            rr = [r for r in rows if r[0] in sel]
            print("  {:28s} spellF1 {:.3f}   macroF1 {:.3f}   frames right {:.1f}%".format(
                k_, np.mean([r[1] for r in rr]), np.mean([r[2] for r in rr]), 100 * np.mean([r[3] for r in rr])))


if __name__ == "__main__":
    main()
