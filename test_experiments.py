"""Regression checks for the portable evaluator and experimental protocol."""

import json
import math
import tempfile
import unittest
from pathlib import Path

from ga_models import Config, Person

ROOT = Path(__file__).resolve().parent


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.people = [
            Person(**row)
            for row in json.loads((ROOT / "data" / "people.json").read_text(encoding="utf-8"))
        ]

    def test_numpy_cpu_matches_independent_scalar_nutrition(self):
        from meal_ga import MealGA, load_foods
        from test_all import scalar_reference

        foods, _ = load_foods(ROOT / "dataset")
        ga = MealGA(foods, self.people, Config(population_size=4), 42, backend="numpy")
        population = [ga.initialize() for _ in range(4)]
        self.assertEqual(ga.population_weights(population).shape, (4, 3, 30))
        for plan, batch in zip(population, ga.evaluate_population(population)):
            expected = scalar_reference(ga, plan)
            self.assertEqual(batch.ul_status, "not_fully_assessed")
            self.assertAlmostEqual(batch.fitness, expected.fitness, places=10)
            for actual, reference in zip(batch.people, expected.people):
                self.assertEqual(actual["totals"].keys(), reference["totals"].keys())
                for key, total in actual["totals"].items():
                    target = reference["totals"][key]
                    if target is None:
                        self.assertIsNone(total)
                    else:
                        self.assertTrue(math.isclose(total, target, rel_tol=1e-12))

    def test_random_search_scores_same_initial_batch_and_keeps_incumbent(self):
        from experiments.run_experiments import run_random_search
        from meal_ga import MealGA, load_foods

        foods, _ = load_foods(ROOT / "dataset")
        cfg = Config(population_size=4, generations=2)
        ga = MealGA(foods, self.people, cfg, 7, backend="numpy")
        reference = MealGA(foods, self.people, cfg, 7, backend="numpy")
        independently_scored = [
            reference.evaluate(reference.initialize()) for _ in range(12)
        ]
        individual, winner, history = run_random_search(ga)
        ga.validate(individual)
        self.assertEqual([row["evaluations"] for row in history], [4, 8, 12])
        for row in history:
            expected = min(independently_scored[:row["evaluations"]], key=lambda e: e.key)
            self.assertAlmostEqual(row["incumbent_fitness"], expected.fitness)
        self.assertAlmostEqual(winner.fitness, history[-1]["incumbent_fitness"])

    def test_quick_suite_exports_paired_inputs_and_all_conditions(self):
        import csv

        from experiments.run_experiments import run_suite

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            run_suite(output, seeds=(42,), population=8, generations=2)
            with (output / "runs.csv").open(encoding="utf-8", newline="") as stream:
                rows = {row["condition"]: row for row in csv.DictReader(stream)}
            self.assertEqual(set(rows), {
                "default_ea", "random_search", "no_crossover", "no_portion_mutation",
                "no_salmon_ea", "no_salmon_random",
            })
            self.assertEqual({row["evaluations"] for row in rows.values()}, {"24"})
            self.assertEqual(
                rows["default_ea"]["initial_best_fitness"], rows["random_search"]["initial_best_fitness"]
            )
            self.assertEqual(
                rows["no_salmon_ea"]["initial_best_fitness"], rows["no_salmon_random"]["initial_best_fitness"]
            )
            self.assertEqual(rows["no_salmon_ea"]["candidate_count"], "29")
            self.assertEqual(rows["no_salmon_ea"]["salmon_g_total"], "0.0")
            for filename in ("history.csv", "summary.csv", "nutrients.csv", "run_manifest.json",
                             "example_plan_seed42.json", "convergence.png", "final_comparison.png"):
                self.assertGreater((output / filename).stat().st_size, 0)
