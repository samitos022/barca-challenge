"""Synthetic behaviour tests, never competition ground-truth fixtures."""
import itertools
from pathlib import Path
import tempfile
import unittest

import numpy as np
from starter.temporal_possession import (
    Config, decode, emissions, read_tracking, tag_evidence, viterbi,
)


class TemporalTests(unittest.TestCase):
    def test_viterbi_matches_exhaustive_global_optimum(self):
        scores = np.random.default_rng(9).normal(size=(6, 3))
        cost = 0.7
        def objective(path):
            return sum(scores[i, state] for i, state in enumerate(path)) - cost * sum(
                a != b for a, b in zip(path, path[1:]))
        actual = viterbi(scores, cost)
        expected = max(objective(p) for p in itertools.product(range(3), repeat=6))
        self.assertAlmostEqual(objective(actual), expected)

    def test_gap_and_isolated_false_opponent_do_not_split_possession(self):
        scores = np.zeros((100, 3))
        scores[:, 0] = 0.08
        scores[40:60] = 0
        scores[70] = [0, 0.08, 0]
        np.testing.assert_array_equal(viterbi(scores, 1.5), np.zeros(100))

    def test_real_change_and_stoppage_survive(self):
        scores = np.zeros((150, 3))
        scores[:50, 0] = 0.1
        scores[50:100, 2] = 0.1
        scores[100:, 1] = 0.1
        np.testing.assert_array_equal(viterbi(scores, 1.5), [0]*50 + [2]*50 + [1]*50)

    def test_halves_reset_and_outside_abstains(self):
        scores = np.zeros((20, 3))
        scores[2:6, 0] = 1
        scores[12:16, 1] = 1
        labels = decode(scores, [(1, 2, 5), (2, 12, 15)], Config())
        self.assertEqual(''.join(labels), 'XXHHHHXXXXXXAAAAXXXX')
        self.assertTrue(np.all(decode(np.zeros((20, 3)), [(1, 0, 19)], Config()) == 'X'))

    def test_defence_inverts_team_and_end_is_exclusive(self):
        event = dict(code='DEFENSIVE TRANSITION', start_frame=2, end_frame=4,
                     tags={'Team': 'Home'})
        scores, _, _ = tag_evidence([event, event], ('Home', 'Away'), 6, Config())
        np.testing.assert_array_equal(scores[:, 1], [0, 0, .45, .45, 0, 0])
        self.assertFalse(scores[:, 0].any())
        with self.assertRaises(ValueError):
            tag_evidence([event], ('Different', 'Away'), 6, Config())

    def test_missing_tag_team_is_neutral(self):
        for team in (None, '', 'N/A'):
            event = dict(code='BUILD UP', start_frame=0, end_frame=6,
                         tags={'Team': team})
            scores, set_piece, _ = tag_evidence([event], ('Home', 'Away'), 6, Config())
            self.assertFalse(scores.any())
            self.assertFalse(set_piece.any())

    def test_missing_ball_does_not_imply_dead_and_set_piece_needs_stillness(self):
        n = 50
        event = dict(code='SET PIECES', start_frame=0, end_frame=n, tags={})
        ball = np.full((n, 2), np.nan)
        distance = np.full((n, 2), np.inf)
        speed = np.full(n, np.nan)
        scores, _ = emissions(ball, distance, speed, speed, [event], ('H', 'A'),
                              (105, 68), 25, Config())
        self.assertFalse(scores.any())
        ball[:] = [50, 30]
        speed[:] = 0
        scores, _ = emissions(ball, distance, speed, speed, [event], ('H', 'A'),
                              (105, 68), 25, Config())
        self.assertTrue(np.all(scores[:, 2] > 0))

    def test_atd_alignment_units_missing_and_bad_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'tracking.txt'
            path.write_text('1:0.5,0.5,1;0.7,0.5,2:0.51,0.5,1\n'
                            '2:0.5,0.5,1;0.7,0.5,2:NaN,NaN,NaN\n')
            ball, distance, speed, _ = read_tracking(
                path, np.array([0, 1]), 2, np.array([100, 60]), 25, Config())
            self.assertAlmostEqual(distance[0, 0], 1)
            self.assertTrue(np.isnan(ball[1]).all())
            self.assertTrue(np.isinf(distance[1]).all())
            self.assertTrue(np.isnan(speed).all())
            path.write_text(path.read_text().replace('2:', '1:'))
            with self.assertRaises(ValueError):
                read_tracking(path, np.array([0, 1]), 2, np.array([100, 60]), 25, Config())


if __name__ == '__main__':
    unittest.main()
