"""Keeper / team fix and frozen-out-of-view fix on BAR-ATH, before and after, aggregates only.

    .venv/bin/python analisi/audit_portieri.py            # BAR-ATH

On BAR-ATH there is no truth: the numbers are the structural checks of analisi/audit_tracking.py
(team counts, an opponent in a goal area with the ball far, keeper presence when the own goal is
in view). Precision on external games with truth: analisi/sweep_portieri_idsse.py.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "starter", "solution", os.path.join("solution", "external")):
    sys.path.insert(0, os.path.join(ROOT, d))

FPS = 25


def keeper_checks(teams, pl, ball, blocks, pitch, H=None):
    """Per half and team: share of ball-far frames with no own player within 20 m of the own
    goal; with an opponent within 6 m of the own goal centre; keeper presence when the own goal
    is in view (needs H)."""
    import track_fixes as tf
    L, W = pitch
    rows = []
    for half, a, z in blocks:
        seg = slice(a, z + 1)
        gx = tf.own_goals(teams, pl, a, z, pitch)
        for k in (0, 1):
            xs, ys = pl[seg][:, teams == k, 0], pl[seg][:, teams == k, 1]
            deep = np.nanmin(np.where(np.isnan(xs), np.inf, np.abs(xs - gx[k])), axis=1)
            far = np.nan_to_num(np.abs(ball[seg, 0] - gx[k]), nan=0.0) > 40
            miss = far & (deep > 20)
            xo, yo = pl[seg][:, teams == 1 - k, 0], pl[seg][:, teams == 1 - k, 1]
            dgo = np.hypot(xo - gx[k], yo - W / 2)
            opp = np.nanmin(np.where(np.isnan(dgo), np.inf, dgo), axis=1) < 6
            r = {"half": half, "team": k, "miss": miss[far].mean(), "opp": (far & opp)[far].mean()}
            if H is not None:
                gv = tf.in_view(H[a:z + 1], np.full((z + 1 - a, 1), gx[k]), np.full((z + 1 - a, 1), W / 2), pitch)[:, 0]
                r["present_in_view"] = 1 - miss[far & gv].mean() if (far & gv).any() else np.nan
                r["miss_goal_out"] = 1 - gv[miss].mean() if miss.any() else np.nan
            rows.append(r)
    return rows


def counts(teams, pl, blocks):
    out = {}
    for half, a, z in blocks:
        ok = ~np.isnan(pl[a:z + 1, :, 0])
        for k in (0, 1):
            c = ok[:, teams == k].sum(1)
            out[(half, k)] = (np.mean(c >= 12), np.mean(c <= 10), np.mean(c == 11))
    return out


def bar_ath():
    from games import GAME, GAMES, asset_paths, half_video_blocks
    import clean_tracking as ct
    import track_fixes as tf
    import tracking
    teams, pl, ball, _, names = tracking.load(GAME)
    blocks = half_video_blocks(GAME)
    pitch = GAMES[GAME]["pitch"]
    H = np.loadtxt(asset_paths(GAME)["homography"], delimiter=",", skiprows=1, usecols=range(1, 10)).reshape(-1, 3, 3)
    S = pl.shape[1]
    t_k, p_k, fl_k, info = tf.fix_keepers(teams, pl, ball, blocks, pitch, return_info=True)
    print("fix_keepers: {} candidate tracklets (>= 2 s ball-far), {} flipped ({:.0f} s); shares (near & deepest) of the "
          "candidates: {}".format(len(info), len(fl_k), sum(b - a for _, a, b in fl_k) / FPS,
                                  np.histogram([r[6] for r in info], bins=[0, .2, .4, .6, .8, 1.01])[0].tolist()))
    t_s, p_s, fl_s = ct.fix_team_swaps(teams, pl)
    t_ks, p_ks, fl_ks = ct.fix_team_swaps(t_k, p_k)
    t_ks, p_ks = tf.compact(t_ks, p_ks, S)
    print("fix_team_swaps alone: {} flipped ({:.0f} s); after fix_keepers: {} more ({:.0f} s)".format(
        len(fl_s), sum(b - a for _, a, b in fl_s) / FPS, len(fl_ks), sum(b - a for _, a, b in fl_ks) / FPS))
    p_f, frz = tf.drop_frozen(pl, H, pitch)
    p_kf, _ = tf.drop_frozen(p_ks, H, pitch)
    p_kfh, frh = tf.drop_frozen(p_ks, H, pitch, held_in_view=True)
    g = np.zeros(len(pl), bool)
    for _, a, z in blocks:
        g[a:z + 1] = True
    print("frozen out of view masked: {:.1f}% of tracked graded positions, {:.2f} per frame".format(
        100 * frz[g].sum() / (~np.isnan(pl[g][..., 0])).sum(), frz[g].sum(1).mean()))
    variants = [("ATD", teams, pl), ("keepers", t_k, p_k), ("team swaps", t_s, p_s),
                ("keepers + swaps", t_ks, p_ks), ("frozen removed", teams, p_f), ("all three", t_ks, p_kf),
                ("all + held in view", t_ks, p_kfh)]
    print("\nteam counts per half: 12+ / 10- / exactly 11 (% of graded frames)")
    for name, t, p in variants:
        c = counts(t, p, blocks)
        print("  {:16s} ".format(name) + "  ".join(
            "h{} {}: {:4.1f}/{:4.1f}/{:4.1f}".format(h, names[k][:4], *(100 * np.array(c[(h, k)])))
            for h, k in sorted(c)))
    print("\nkeepers, ball far (> 40 m from the own goal line): no own player within 20 m / opponent within 6 m "
          "of own goal / own player within 20 m when the own goal is in view / goal out of view when missing")
    for name, t, p in variants:
        rows = keeper_checks(t, p, ball, blocks, pitch, H)
        print("  {:16s} ".format(name) + "  ".join(
            "h{} {}: {:4.1f}/{:4.1f}/{:3.0f}/{:3.0f}".format(r["half"], names[r["team"]][:4], 100 * r["miss"], 100 * r["opp"],
                                                       100 * r["present_in_view"], 100 * r["miss_goal_out"]) for r in rows))
    pace_report(pl, p_f, ball, g)


def pace_report(pl, p_f, ball, g):
    """Mean player pace (as v2 / audit_tracking measure it), before and after the frozen fix."""
    def pace(p):
        vel = np.linalg.norm(np.gradient(p, axis=0), axis=2) * FPS
        with np.errstate(invalid="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                return np.nanmean(vel, axis=1)
    p0, p1 = pace(pl), pace(p_f)
    seen = ~np.isnan(ball[:, 0])
    e = np.flatnonzero(np.diff(np.concatenate(([0], (~seen & g).astype(np.int8), [0]))))
    long_gap = np.zeros(len(ball), bool)
    gp0, gp1 = [], []
    for a, b in zip(e[::2], e[1::2]):
        if b - a >= 5 * FPS:
            long_gap[a:b] = True
            gp0.append(np.nanmean(p0[a:b]))
            gp1.append(np.nanmean(p1[a:b]))
    gp0, gp1 = np.array(gp0), np.array(gp1)
    print("\nmean player pace (m/s): ball seen mean {:.2f} -> {:.2f}, median {:.2f} -> {:.2f}; inside long gaps "
          "(>= 5 s) frame mean {:.2f} -> {:.2f}; per gap median {:.2f} -> {:.2f}".format(
              np.nanmean(p0[g & seen]), np.nanmean(p1[g & seen]), np.nanmedian(p0[g & seen]), np.nanmedian(p1[g & seen]),
              np.nanmean(p0[long_gap]), np.nanmean(p1[long_gap]), np.median(gp0), np.median(gp1)))
    walk = (gp0 + 0.0) < 1.5
    print("long gaps {}: players 'walking' (< 1.5 m/s, (SPD_PLAY+SPD_DEAD)/2 of v2) {} -> {}; gap frames with "
          "pace > 1.8 m/s {:.0f}% -> {:.0f}%".format(len(gp0), int(walk.sum()), int(np.sum(gp1 < 1.5)),
                                                    100 * np.mean(p0[long_gap] > 1.8), 100 * np.mean(p1[long_gap] > 1.8)))
    print("ratio pace(after)/pace(before), ball seen median {:.3f}, gaps median {:.3f}".format(
        np.nanmedian(p1[g & seen] / p0[g & seen]), np.nanmedian(p1[long_gap] / p0[long_gap])))


if __name__ == "__main__":
    bar_ath()
