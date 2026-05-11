import math
import unittest

from krope_core import (
    DEFAULT_THETA,
    DEFAULT_PHI,
    MODEL_ADDITIVE,
    MODEL_DUAL,
    MODEL_ROPE,
    SENTENCE_PAIRS,
    aggregate_model_means,
    compute_pair,
    dot,
    rope_frequencies,
    rotate_pair,
    softmax_rows,
)


class KropeMathTests(unittest.TestCase):
    def assertAlmostList(self, actual, expected, places=9):
        self.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            self.assertAlmostEqual(a, e, places=places)

    def test_rotate_pair_preserves_vector_norm(self):
        original = [1.2, -0.7]
        original_norm_sq = dot(original, original)
        for angle in [-3.0, -1.2, 0.0, 0.7, 2.4, 6.0]:
            rotated = rotate_pair(original, angle)
            self.assertAlmostEqual(dot(rotated, rotated), original_norm_sq, places=12)

    def test_relative_rotation_identity_for_dot_product(self):
        q = [0.8, -0.3]
        k = [0.2, 1.1]
        a = 0.75
        b = 2.10
        left = dot(rotate_pair(q, a), rotate_pair(k, b))
        right = dot(q, rotate_pair(k, b - a))
        self.assertAlmostEqual(left, right, places=12)

    def test_softmax_rows_are_stable_and_sum_to_one(self):
        rows = softmax_rows([[1000.0, 1001.0, 1002.0], [-1000.0, -999.0, -998.0]])
        for row in rows:
            self.assertAlmostEqual(sum(row), 1.0, places=12)
            self.assertTrue(all(0.0 <= value <= 1.0 for value in row))
        self.assertAlmostList(rows[0], rows[1], places=12)

    def test_rope_frequencies_follow_geometric_sequence(self):
        self.assertAlmostList(rope_frequencies(8), [1.0, 0.1, 0.01, 0.001], places=12)
        freqs = rope_frequencies(16)
        ratio = freqs[1] / freqs[0]
        for prev, current in zip(freqs, freqs[1:]):
            self.assertAlmostEqual(current / prev, ratio, places=12)

    def test_pair_result_contains_valid_attention_and_deltas(self):
        result = compute_pair(SENTENCE_PAIRS[0], theta=DEFAULT_THETA, phi=DEFAULT_PHI)
        for model_name in [MODEL_ROPE, MODEL_ADDITIVE, MODEL_DUAL]:
            model = result.models[model_name]
            for matrix in [model.attention_a, model.attention_b]:
                for row in matrix:
                    self.assertAlmostEqual(sum(row), 1.0, places=12)
                    self.assertTrue(all(0.0 <= value <= 1.0 for value in row))
            for relation in model.relations:
                self.assertGreaterEqual(relation.abs_diff, 0.0)
                self.assertLessEqual(relation.abs_diff, 1.0)

    def test_default_dual_axis_krope_is_most_stable_on_curated_pairs(self):
        means = aggregate_model_means(SENTENCE_PAIRS, theta=DEFAULT_THETA, phi=DEFAULT_PHI)
        self.assertAlmostEqual(means[MODEL_ROPE], 0.131298, places=6)
        self.assertAlmostEqual(means[MODEL_ADDITIVE], 0.114263, places=6)
        self.assertAlmostEqual(means[MODEL_DUAL], 0.070550, places=6)
        self.assertLess(means[MODEL_DUAL], means[MODEL_ADDITIVE])
        self.assertLess(means[MODEL_ADDITIVE], means[MODEL_ROPE])


if __name__ == "__main__":
    unittest.main()
