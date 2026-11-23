import unittest
from kad_dispatch import choose, load_policy

POLICY = load_policy(__file__.replace("test_kad_dispatch.py", "../configs/kad-policy.json"))


class KADTests(unittest.TestCase):
    def test_qualified_shape_selects_candidate(self):
        shape = {"b": 1, "sq": 2048, "sk": 2048, "hq": 16, "hk": 4, "d": 128, "dtype": "bfloat16", "causal": True}
        decision = choose(POLICY, shape)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.config["tile_m"], 64)
        self.assertGreaterEqual(decision.lower_bound, 1.10)

    def test_unknown_shape_falls_back(self):
        shape = {"b": 1, "sq": 123, "sk": 123, "hq": 16, "hk": 4, "d": 128, "dtype": "bfloat16", "causal": True}
        self.assertIsNone(choose(POLICY, shape))

    def test_regression_or_failed_gate_falls_back(self):
        shape = {"b": 1, "sq": 2048, "sk": 2048, "hq": 16, "hk": 4, "d": 128, "dtype": "bfloat16", "causal": True}
        bad = {**POLICY, "shapes": {next(iter(POLICY["shapes"])): {"lower_bound": 1.2, "regressions": 1, "correctness_pass": True, "config": {}}}}
        self.assertIsNone(choose(bad, shape))

    def test_explicit_pattern_can_cover_pr_shape_without_case_id_lookup(self):
        shape = {"b": 8, "sq": 2048, "sk": 2048, "hq": 16, "hk": 4, "d": 128, "dtype": "bfloat16", "causal": True}
        policy = {"schema_version": 1, "minimum_lower_bound": 1.1, "shapes": {}, "patterns": [{
            "id": "pr-gqa-d128-causal", "match": {"d": 128, "hq": 16, "hk": 4, "dtype": "bfloat16", "causal": True},
            "lower_bound": 1.2, "regressions": 0, "correctness_pass": True,
            "config": {"tile_m": 64, "tile_n": 112, "num_threads": 128}
        }]}
        decision = choose(policy, shape)
        self.assertEqual(decision.shape_id, "pr-gqa-d128-causal")


if __name__ == "__main__":
    unittest.main()
