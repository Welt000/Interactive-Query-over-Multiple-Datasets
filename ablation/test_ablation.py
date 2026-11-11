"""Correctness checks for ablation switches and exhaustive selection."""
import sys
import unittest
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from SharedQ.algorithm import SharedQuestions, DetectDivideSharedQuestions, _DatasetState
from structure.common import UtilityRange
from experiment_config import canonical_method
from baseline.experiment_final import build_parser


class AblationTests(unittest.TestCase):
    def test_naming_and_recursion(self):
        self.assertEqual(canonical_method(' SharedQ '), 'sharedq')
        args = build_parser().parse_args(['dummy.txt', '--sharedq-detect-workers', '1',
                                         '--summary-csv', 'summary.csv',
                                         '--detail-csv', 'detail.csv'])
        self.assertEqual(args.sharedq_detect_workers, 1)

    def test_exhaustive_matches_scalar_bruteforce_without_pair_cap(self):
        rng = np.random.default_rng(7)
        datasets = [rng.random((9, 3)), rng.random((8, 3))]
        algorithm = SharedQuestions(hyperplane_reduction=False, max_pairs_per_dataset=1)
        states = [_DatasetState(i, data, list(range(len(data)))) for i, data in enumerate(datasets)]
        region = UtilityRange(3)
        caches = {s.dataset_id: algorithm._state_region_cache(s, region) for s in states}
        total = sum(len(s.candidates) for s in states)
        scored = []
        for state in states:
            for left, right in combinations(state.candidates, 2):
                normal = state.points[left] - state.points[right]
                values = region.vertices() @ normal
                if not (np.any(values > region.tol) and np.any(values < -region.tol)):
                    continue
                score = algorithm._shared_pruning_score(normal, states, caches, total, region)
                scored.append((score, state.dataset_id, left, right))
        self.assertGreater(len(scored), 2)
        expected = max(scored, key=lambda x: x[0])
        actual = algorithm._select_question(states, region, set())
        self.assertAlmostEqual(actual.score, expected[0])
        self.assertEqual((actual.dataset_id, actual.left, actual.right), expected[1:])
        # Selection must not call the PCA helper in exhaustive mode.
        algorithm._ideal_hyperplane = lambda *a: self.fail('PCA called while disabled')
        self.assertIsNotNone(algorithm._select_question(states, region, set()))

    def test_all_four_variants_return_exact_top1(self):
        rng = np.random.default_rng(91)
        datasets = [rng.random((15, 3)) for _ in range(3)]
        for pca in (False, True):
            for local in (False, True):
                with self.subTest(pca=pca, local=local):
                    kwargs = dict(hyperplane_reduction=pca, random_state=10,
                                  detect_workers=1, max_questions=500)
                    if local:
                        kwargs.update(radius=0.3, random_initial_center=True)
                    algorithm = (DetectDivideSharedQuestions if local else SharedQuestions)(**kwargs)
                    utility = np.array([0.23, 0.51, 0.26])
                    result = algorithm.search(datasets, utility)
                    self.assertEqual(len(result.dataset_results), 3)
                    for item in result.dataset_results:
                        scores = datasets[item.dataset_id] @ utility
                        self.assertAlmostEqual(scores[item.point_index], scores.max())
                        self.assertEqual(item.candidate_count, 1)

    def test_neighbor_region_walk_matches_full_scan(self):
        rng = np.random.default_rng(2026)
        algorithm = SharedQuestions()
        for _ in range(4):
            points = rng.random((35, 3))
            candidates, adjacency = algorithm._convex_top1_candidates_and_adjacency(points)
            state = _DatasetState(0, points, candidates, adjacency=adjacency)
            for narrowed in (False, True):
                region = UtilityRange(3)
                if narrowed:
                    region.add_preference(points[0], points[1])
                brute = {}
                expected = {
                    candidate for candidate in candidates
                    if algorithm._partition_region(
                        points, candidate, candidates, region, brute, adjacency
                    ) is not None
                }
                state.local_regions = {}
                state.local_region_cache_version = None
                actual = set(algorithm._state_region_cache(state, region))
                self.assertEqual(actual, expected)

    def test_hdpi_partition_builder_incremental_clipping(self):
        rng = np.random.default_rng(518)
        points = rng.random((42, 3))
        helper = SharedQuestions()
        candidates, adjacency = helper._convex_top1_candidates_and_adjacency(points)
        hdpi = SharedQuestions(partition_construction='hdpi', hyperplane_reduction=True)
        state = _DatasetState(0, points, list(candidates), adjacency=adjacency)
        full_range = UtilityRange(3)
        initial = hdpi._state_region_cache(state, full_range)
        self.assertEqual(set(initial), set(candidates))

        preferred, rejected = 3, 7
        normal = points[preferred] - points[rejected]
        narrowed = full_range.copy()
        narrowed.add_preference(points[preferred], points[rejected])
        state.local_regions = initial
        self.assertTrue(hdpi._clip_hdpi_partition_cache(state, narrowed, normal))
        state.local_region_cache_version = hdpi._state_cache_version(state, narrowed)
        incrementally_feasible = {
            idx for idx, region in state.local_regions.items()
            if region.vertices is not None and len(region.vertices) > 0
        }

        rebuilt_state = _DatasetState(0, points, list(candidates), adjacency=adjacency)
        rebuilt = hdpi._state_region_cache(rebuilt_state, narrowed)
        rebuilt_feasible = {
            idx for idx, region in rebuilt.items()
            if region.vertices is not None and len(region.vertices) > 0
        }
        self.assertEqual(incrementally_feasible, rebuilt_feasible)

        datasets = [rng.random((18, 3)) for _ in range(3)]
        utility = np.array([0.31, 0.47, 0.22])
        pca_calls = []
        original_ideal_hyperplane = hdpi._ideal_hyperplane
        def count_pca(*args, **kwargs):
            pca_calls.append(1)
            return original_ideal_hyperplane(*args, **kwargs)
        hdpi._ideal_hyperplane = count_pca
        result = hdpi.search(datasets, utility)
        self.assertGreater(len(pca_calls), 0)
        for item in result.dataset_results:
            scores = datasets[item.dataset_id] @ utility
            self.assertAlmostEqual(scores[item.point_index], scores.max())
            self.assertEqual(item.candidate_count, 1)


if __name__ == '__main__':
    unittest.main()
