"""Option 2 of the plan: keep the model as it is and correct its probabilities with a small module on
the collective signals, with weights chosen on half of the matches and scored on the other half.

The module is a logistic regression (few parameters, easy to read) trained out of fold on the
collective signals alone: dead_score(t) = log-odds that play is stopped. It enters the decoder in
two places, each with its own weight:
  - emissions: log P(dead) += w_e * dead_score  (log P(home), log P(away) unchanged)
  - transition costs: entering D is cheaper where the dead score is rising and leaving D cheaper
    where it is falling (w_in, w_out scale the cost by exp(-w * trend)).
w = 0 everywhere is the model alone, which is always among the candidates, so the correction can
only be chosen where it helps on the matches used to choose.

    .venv/bin/python analisi/spell_correttore.py [tag]
"""

import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE] + [os.path.join(os.path.dirname(HERE), d) for d in ("solution", "evaluation", "starter", "solution/external")]
import spell_exp as E  # noqa: E402
import spell_lab as L  # noqa: E402
import viterbi  # noqa: E402

D, X = 2, 3
SIGNALS = ("col_speed_med_home", "col_speed_med_away", "col_still_home", "col_still_away", "col_running_home",
           "col_running_away", "col_brake_share_home", "col_brake_share_away", "col_pace_vs_30s", "col_pace_vs_120s",
           "col_press_d_home", "col_press_d_away", "col_own_half_home", "col_own_half_away", "col_stand_group_home",
           "col_stand_group_away", "col_corner_d", "col_touch_d", "col_ball_goal_area", "col_centre_circle", "col_react",
           "col_polar_home", "col_polar_away", "col_to_ball_home", "col_to_ball_away", "col_n_all",
           "col_speed_med_c5s_home", "col_speed_med_c5s_away", "col_speed_med_trend_home", "col_speed_med_trend_away",
           "col_still_trend_home", "col_still_trend_away", "col_press_d_trend_home", "col_press_d_trend_away",
           "col_pace_vs_30s_c2s", "col_pace_vs_30s_trend", "col_corner_d_c2s", "col_touch_d_c2s")


def signals(r):
    z = np.load(L.col_path(r), mmap_mode="r")
    names = [str(x) for x in z["names"]]
    return np.asarray(z["X"])[:, [names.index(s) for s in SIGNALS]].astype(np.float64)


def prep(S, med):
    miss = np.isnan(S)
    return np.concatenate([np.where(miss, med, S), miss.astype(np.float64)], axis=1)


def dead_scores(every=10):
    """{test key: (n,) out-of-fold dead log-odds from the collective signals}."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    rs = L.runs()
    fo = L.fold_of()
    out = {}
    for f in range(L.FOLDS):
        Ss, ys = [], []
        for r in rs:
            if fo[r[0]] == f:
                continue
            z = L.load(r)
            m = np.asarray(z["usable"]).copy()
            step = L.PFF_EVERY if L.set_of(r[0]) == "PFF" else every
            m[np.arange(len(m)) % step != 0] = False
            Ss.append(signals(r)[m])
            ys.append((np.asarray(z["gt"])[m] == D).astype(np.int8))
        S, y = np.concatenate(Ss), np.concatenate(ys)
        med = np.nanmedian(S, axis=0)
        sc = StandardScaler().fit(prep(S, med))
        clf = LogisticRegression(C=0.1, max_iter=500).fit(sc.transform(prep(S, med)), y)
        for r in rs:
            if fo[r[0]] == f and r[4] == "test":
                out[L.key(r)] = clf.decision_function(sc.transform(prep(signals(r), med))).astype(np.float32)
        print("dead score fold {} done".format(f), flush=True)
    return out


def trend(v, half=25):
    """next-1 s mean minus past-1 s mean."""
    c = np.concatenate(([0.0], np.cumsum(v)))
    n = len(v)
    i = np.arange(n)
    lo, hi = np.maximum(i - half, 0), np.minimum(i + half + 1, n)
    past = (c[i] - c[lo]) / np.maximum(i - lo, 1)
    nxt = (c[hi] - c[np.minimum(i + 1, n)]) / np.maximum(hi - i - 1, 1)
    return nxt - past


def _dec(p3, blocks, cfg, extra, z):
    ct, cd, we, wi, wo = cfg
    s = extra
    lp = np.log(np.clip(p3, 1e-4, 1.0))
    if we:
        lp = lp.copy()
        lp[:, 2] += we * np.clip(s, -6, 6)
    tr = np.clip(trend(s), -3, 3)
    ci = cd * np.exp(-wi * np.maximum(tr, 0)) if wi else cd
    co = cd * np.exp(-wo * np.maximum(-tr, 0)) if wo else cd
    return viterbi.decode(p3, blocks, ct, ci, co, emit=lp)


def main(tag="base"):
    o = L.load_oof(tag)
    ds = dead_scores()
    np.savez(os.path.join(L.LAB, "deadscore_col.npz"), **ds)
    keys = E.test_keys(o)
    cfgs = [(16, 32, we, wi, wo) for we in (0.0, 0.1, 0.25, 0.5) for wi in (0.0, 0.5, 1.0) for wo in (0.0, 0.5, 1.0)]
    res = E.grid(o, _dec, cfgs, keys, extras={k: ds[k] for k in keys})
    base = {k: v[0] for k, v in res.items()}
    only_e = [i for i, c in enumerate(cfgs) if c[3] == 0 and c[4] == 0]
    s_e, ch_e = E.choose_and_score({k: v[only_e] for k, v in res.items()}, [cfgs[i] for i in only_e])
    s_all, ch_all = E.choose_and_score(res, cfgs)
    E.summary([("model '{}', decoder 16 / 32".format(tag), base), ("+ collective dead score in the emissions, chosen", s_e),
               ("+ dead score in emissions and costs, chosen", s_all)], "option 2: correction module on the collective signals")
    print("  chosen (team, dead, w_emission, w_in, w_out) half 0 / 1: emissions {} | all {}".format(ch_e, ch_all))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "base")
