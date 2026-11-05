"""Fix tracklets assigned to the wrong team, from the "at most 11 per team" constraint alone.

Measured on BAR-ATH: a team has more than 11 tracked players on 6.4% of graded team-frames, and in
almost all of them the other team has fewer at the same time (12-10: 7967 frames, 10-12: 3593,
12-11 only 1979). The total is still 22: a player of one team is carried by a track slot of the
other (typically a goalkeeper, or a player after an identity switch in a crowd). The extra player
is usually not a double of a teammate (closest teammate 3.8 m away, median), so removing
duplicates would not help; changing the team of the right tracklet does.

A tracklet is a maximal run of frames in which one slot is tracked. Greedy: flip the tracklet
whose change of team removes the most frames over 11 while adding the fewest new ones, and repeat
while that gain is positive. The cost uses only the tracking (no labels, no frame-specific rule),
so it runs unchanged on any game.

Output keeps the shape the rest of the code expects: twice the slots, the second half being each
slot's copy for the other team; a flipped tracklet's frames move to its copy.
"""

import numpy as np

MAX_PER_TEAM = 11
MIN_GAIN = 25          # frames: a flip must remove at least one second of violations


def tracklets(ok):
    """[(slot, first, stop)] maximal runs of True per slot."""
    out = []
    for s in range(ok.shape[1]):
        idx = np.flatnonzero(np.diff(np.concatenate(([0], ok[:, s].astype(np.int8), [0]))))
        out += [(s, int(a), int(b)) for a, b in zip(idx[::2], idx[1::2])]
    return out


def fix_team_swaps(teams, players, max_iter=500):
    """(teams2, players2, flipped tracklets [(slot, first, stop)])."""
    n, slots = players.shape[:2]
    ok = ~np.isnan(players[..., 0])
    team_of = np.repeat(teams[None].astype(np.int8), n, axis=0)  # per frame, per slot
    cnt = np.stack([np.sum(ok & (team_of == k), axis=1) for k in (0, 1)])
    tl = tracklets(ok)
    flipped = set()

    def csum(x):
        return np.concatenate(([0], np.cumsum(x)))

    for _ in range(max_iter):
        # Objective: sum over frames of |c0 - 11| + |c1 - 11|. Moving one player from team k to
        # the other changes it by -1 per frame where k has more than 11 (+1 where it has not), and
        # by -1 per frame where the other team has fewer than 11 (+1 where it has not). Counted
        # over the WHOLE tracklet: a correct player who happens to be on the pitch during a
        # 12-10 stretch also lives outside it, where moving him costs; the misassigned tracklet
        # starts and ends with the stretch.
        over = [csum(cnt[k] > MAX_PER_TEAM) for k in (0, 1)]
        short = [csum(cnt[k] < MAX_PER_TEAM) for k in (0, 1)]
        best, best_gain = None, MIN_GAIN - 1
        for i, (s, a, b) in enumerate(tl):
            if i in flipped:
                continue
            k = int(team_of[a, s])
            removed = over[k][b] - over[k][a]          # frames where its team has too many
            if removed < MIN_GAIN:
                continue
            life = b - a
            filled = short[1 - k][b] - short[1 - k][a]  # frames where the other team lacks one
            gain = (2 * removed - life) + (2 * filled - life)
            if gain > best_gain:
                best, best_gain = i, gain
        if best is None:
            break
        s, a, b = tl[best]
        k = int(team_of[a, s])
        team_of[a:b, s] = 1 - k
        cnt[k, a:b] -= 1
        cnt[1 - k, a:b] += 1
        flipped.add(best)

    players2 = np.full((n, 2 * slots, 2), np.nan, dtype=players.dtype)
    moved = team_of != teams[None]
    players2[:, :slots][~moved] = players[~moved]
    players2[:, slots:][moved] = players[moved]
    teams2 = np.concatenate([teams, 1 - teams]).astype(teams.dtype)
    return teams2, players2, [tl[i] for i in sorted(flipped)]


def team_counts(teams, players, mask=None):
    """(share of team-frames over 11, share under 9) inside mask."""
    ok = ~np.isnan(players[..., 0])
    c = np.stack([ok[:, teams == k].sum(1) for k in (0, 1)])
    if mask is not None:
        c = c[:, mask]
    return float(np.mean(c > MAX_PER_TEAM)), float(np.mean(c < 9))
