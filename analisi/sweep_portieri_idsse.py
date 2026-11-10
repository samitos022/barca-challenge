"""Choose fix_keepers' zone radius and share on the degraded IDSSE games (never on BAR-ATH).

Truth for team membership: in the degraded layout (external/degrade.py) a slot j >= S is a copy
holding a tracklet moved to the wrong team. Keeper slots of the clean game: per half, the player
of each team most often closest to its own goal. Aggregates only.

    .venv/bin/python analisi/sweep_portieri_idsse.py [--chain-only]
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "starter", "solution", os.path.join("solution", "external")):
    sys.path.insert(0, os.path.join(ROOT, d))
import calibrate  # noqa: E402
import clean_tracking as ct  # noqa: E402
import idsse  # noqa: E402
import track_fixes as tf  # noqa: E402

FPS = 25
RADII = (11.0, 16.5, 20.0, 25.0, 30.0)
SHARES = (0.6, 0.8, 0.9)


def keeper_slots(teams, pl, blocks, pitch):
    W = pitch[1]
    gk = set()
    for _, a, z in blocks:
        gx = tf.own_goals(teams, pl, a, z, pitch)
        for k in (0, 1):
            cols = np.flatnonzero(teams == k)
            d = np.hypot(pl[a:z + 1, cols, 0] - gx[k], pl[a:z + 1, cols, 1] - W / 2)
            d = np.where(np.isnan(d), np.inf, d)
            best = np.bincount(np.argmin(d, axis=1)[np.isfinite(d.min(axis=1))], minlength=len(cols))
            gk.add(int(cols[np.argmax(best)]))
    return gk


def membership(players_final, n_doublings, S, graded):
    """Share of tracked graded slot-frames on the TRUE team. Degraded layout: slot j < S right,
    j >= S a wrong-team copy; each later doubling (fix) puts flipped data in the upper copy."""
    m = players_final.shape[1]
    B = 2 * S
    j = np.arange(m)
    flips = np.zeros(m, int)
    for d in range(n_doublings):
        flips += (j // (B * 2 ** d)) % 2
    wrong0 = (j % B) >= S
    right = wrong0 == (flips % 2 == 1)
    ok = ~np.isnan(players_final[graded][..., 0])
    return float(ok[:, right].sum() / ok.sum())


def chain_report():
    """Team membership accuracy and team counts: degraded, keepers, swaps, keepers + swaps."""
    acc = {k: [] for k in ("degraded", "keepers", "swaps", "keepers+swaps")}
    over = {k: [] for k in acc}
    for run in calibrate.IDSSE_RUNS:
        teams, players, ball, _, blocks, _ = calibrate.case(run, "realistic", "tracab")
        pitch = calibrate.pitch_of(run)
        S = len(teams) // 2
        g = np.zeros(len(ball), bool)
        for _, a, z in blocks:
            g[a:z + 1] = True
        tk, pk, _ = tf.fix_keepers(teams, players, ball, blocks, pitch)
        ts, ps, _ = ct.fix_team_swaps(teams, players)
        tks, pks, _ = ct.fix_team_swaps(tk, pk)
        for name, t, p, nd in (("degraded", teams, players, 0), ("keepers", tk, pk, 1), ("swaps", ts, ps, 1),
                               ("keepers+swaps", tks, pks, 2)):
            acc[name].append(membership(p, nd, S, g))
            ok = ~np.isnan(p[g][..., 0])
            over[name].append(np.mean([np.mean(ok[:, t == k].sum(1) >= 12) for k in (0, 1)]))
        print("run", run, "chain done", flush=True)
    for name in acc:
        print("{:14s} tracked slot-frames on the wrong team {:.3f}%; team-frames with 12+ {:.2f}%".format(
            name, 100 * (1 - np.mean(acc[name])), 100 * np.mean(over[name])))


def main():
    res = {(r, s): [0, 0, 0.0, 0.0, 0] for r in RADII for s in SHARES}   # flips, right, flip s, right s, found
    n_gk = 0
    for run in calibrate.IDSSE_RUNS:
        teams, players, ball, _, blocks, _ = calibrate.case(run, "realistic", "tracab")
        pitch = calibrate.pitch_of(run)
        S = len(teams) // 2
        c_teams, c_pl, _, _, _ = idsse.load(run[0])
        gk = keeper_slots(c_teams, c_pl, blocks, pitch)
        ok = ~np.isnan(players[..., 0])
        swapped = set()
        for _, a0, z0 in blocks:
            for s, a, b in ct.tracklets(ok[a0:z0 + 1]):
                if s >= S and (s - S) in gk and b - a >= tf.MIN_FAR_S * FPS:
                    swapped.add((s, a + a0, b + a0))
        n_gk += len(swapped)
        for r in RADII:
            info = tf.keeper_candidates(teams, players, ball, blocks, pitch, r_gk=r)
            for sh in SHARES:
                acc = res[(r, sh)]
                for s, a, b, nf, sn, sd, sb in info:
                    if sb >= sh:
                        acc[0] += 1
                        acc[2] += (b - a) / FPS
                        if s >= S:
                            acc[1] += 1
                            acc[3] += (b - a) / FPS
                        if (s, a, b) in swapped:
                            acc[4] += 1
        print("run", run, "done", flush=True)
    print("{} swapped keeper tracklets (>= 2 s) in {} degraded IDSSE runs".format(n_gk, len(calibrate.IDSSE_RUNS)))
    for (r, sh), (f, ok_, fs, rs, found) in res.items():
        print("R_GK {:4.1f} SHARE {:.1f}: flips {:3d}, precision {:3.0f}% ({:3.0f}% of time), keeper recall {}/{}".format(
            r, sh, f, 100 * ok_ / max(f, 1), 100 * rs / max(fs, 1e-9), found, n_gk))


if __name__ == "__main__":
    if "--chain-only" not in sys.argv:
        main()
    chain_report()
