"""统一测试入口：CNF数据、遗传算子、三餐约束及PyTorch评分。"""

import copy
import csv
import json
import math
import random
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock

try:
    import torch
except ImportError:
    torch = None

from cnf_data import csv_rows, load_cnf_data
from device_utils import resolve_device
from ga_models import (
    ALL_FIELDS,
    EPS,
    NUTRIENTS,
    UL,
    WEIGHTS,
    Config,
    Dish,
    Evaluation,
    Food,
    Fruit,
    Individual,
    Meal,
    Person,
)
from genetic_operators import crossover as crossover_operator
from genetic_operators import mutation, parent_selection, survivor_selection
from meal_ga import MealGA, bounded_fruit_weights, load_foods

ROOT = Path(__file__).resolve().parent


class CNFDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = ROOT / "dataset"
        cls.foods, cls.metadata = load_foods(cls.dataset)
        cls.candidates = json.loads(
            (cls.dataset / "candidates.json").read_text(encoding="utf-8")
        )
        cls.codes = {f.id for f in cls.foods}
        cls.nutrient_map = {
            field: (record["code"], record["unit"])
            for field, record in cls.metadata["nutrient_mapping"].items()
        }
        needed = {code for code, _ in cls.nutrient_map.values()}
        # Small fixtures are selected from the real files; mutation tests never
        # change the repository's original CNF CSVs.
        cls.tables = {}
        for name in (
            "food_name.csv",
            "food_source.csv",
            "cnf_food_group.csv",
            "nutrient_name.csv",
            "nutrient_amount.csv",
        ):
            rows = csv_rows(cls.dataset / name, set())
            if name == "food_name.csv":
                rows = (r for r in rows if r["Food_Code"] in cls.codes)
            elif name == "nutrient_amount.csv":
                rows = (
                    r
                    for r in rows
                    if r["Food_Code"] in cls.codes and r["Nutrient_Code"] in needed
                )
            cls.tables[name] = list(rows)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        for name, rows in self.tables.items():
            self.write_table(name, rows)
        (self.path / "candidates.json").write_text(
            json.dumps(self.candidates), encoding="utf-8"
        )

    def write_table(self, name, rows, fieldnames=None):
        with (self.path / name).open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=fieldnames if fieldnames is not None else self.tables[name][0].keys()
            )
            writer.writeheader()
            writer.writerows(rows)

    def test_every_selected_nutrient_matches_original_csv(self):
        raw = {
            (r["Food_Code"], r["Nutrient_Code"]): float(r["Nutrient_Amount"])
            for r in self.tables["nutrient_amount.csv"]
            if r["Nutrient_Amount"].strip()
        }
        self.assertEqual(len(self.foods), 30)
        self.assertFalse(self.metadata["demo_only"])
        for food in self.foods:
            for field, (code, _) in self.nutrient_map.items():
                self.assertEqual(food.per100g[field], raw.get((food.id, code)))
            self.assertIsNone(food.per100g["ul_applicable_e_mg"])
            self.assertTrue(self.metadata["food_records"][food.id]["description_en"])
        # Regression guards for the exact staple entries and nutrient forms.
        foods = {f.id: f for f in self.foods}
        self.assertEqual(
            {f.id for f in self.foods if f.role == "staple"}, {"4523", "4066", "4458"}
        )
        self.assertAlmostEqual(foods["4523"].per100g["carbohydrate_g"], 28.17)
        self.assertAlmostEqual(foods["4066"].per100g["carbohydrate_g"], 49.21)
        self.assertEqual(foods["4066"].per100g["synthetic_folic_acid_ug"], 97)
        self.assertEqual(len(self.metadata["input_sha256"]), 6)

    def test_real_data_runs_through_ga(self):
        foods, _ = load_foods(self.path)
        ga = MealGA(
            foods,
            [Person("A", 30, "female", 60, 2000)],
            Config(population_size=8, generations=2),
            seed=7,
        )
        plans, history = ga.run()
        self.assertEqual(len(history), 3)
        for individual, ev in plans:
            ga.validate(individual)
            self.assertIsNone(ev.people[0]["totals"]["ul_applicable_e_mg"])

    def test_missing_nutrient_file_is_explicit_error(self):
        (self.path / "nutrient_amount.csv").unlink()
        with self.assertRaisesRegex(ValueError, "nutrient_amount.csv"):
            load_foods(self.path)

    def test_missing_required_nutrient_not_zero(self):
        rows = [
            r
            for r in self.tables["nutrient_amount.csv"]
            if (r["Food_Code"], r["Nutrient_Code"]) != ("4523", "320")
        ]
        self.write_table("nutrient_amount.csv", rows)
        with self.assertRaisesRegex(ValueError, "vitamin_a_rae_ug"):
            load_cnf_data(self.path)

    def test_missing_optional_ul_remains_unknown(self):
        rows = [
            r
            for r in self.tables["nutrient_amount.csv"]
            if (r["Food_Code"], r["Nutrient_Code"]) != ("4523", "319")
        ]
        self.write_table("nutrient_amount.csv", rows)
        foods, _ = load_cnf_data(self.path)
        rice = next(f for f in foods if f["id"] == "4523")
        self.assertIsNone(rice["per100g"]["preformed_a_ug"])
        self.assertEqual(rice["per100g"]["vitamin_a_rae_ug"], 0)

    def test_mismatched_unit_rejected(self):
        rows = copy.deepcopy(self.tables["nutrient_name.csv"])
        next(r for r in rows if r["Nutrient_Code"] == "328")["Nutrient_Unit"] = (
            "International Unit"
        )
        self.write_table("nutrient_name.csv", rows)
        with self.assertRaisesRegex(ValueError, "code/unit mismatch"):
            load_cnf_data(self.path)

    def test_nutrient_codes_are_resolved_after_renumbering(self):
        for name in ("nutrient_name.csv", "nutrient_amount.csv"):
            rows = copy.deepcopy(self.tables[name])
            for row in rows:
                row["Nutrient_Code"] = str(int(row["Nutrient_Code"]) + 100000)
            self.write_table(name, rows)
        foods, metadata = load_foods(self.path)
        self.assertEqual(foods, self.foods)
        for field, (code, unit) in self.nutrient_map.items():
            mapping = metadata["nutrient_mapping"][field]
            self.assertEqual(mapping["code"], str(int(code) + 100000))
            self.assertEqual(mapping["unit"], unit)

    def test_standard_tag_has_priority_over_english_name(self):
        rows = copy.deepcopy(self.tables["nutrient_name.csv"])
        next(r for r in rows if r["Tagname"] == "PROCNT")["Nutrient_Name_EN"] = "Renamed protein"
        self.write_table("nutrient_name.csv", rows)
        foods, _ = load_foods(self.path)
        self.assertEqual(foods, self.foods)

    def test_english_names_work_without_standard_tags(self):
        for remove_column in (False, True):
            with self.subTest(remove_column=remove_column):
                rows = copy.deepcopy(self.tables["nutrient_name.csv"])
                for row in rows:
                    if remove_column:
                        row.pop("Tagname")
                    else:
                        row["Tagname"] = ""
                    row["Nutrient_Name_EN"] = "  " + row["Nutrient_Name_EN"].upper() + "  "
                self.write_table("nutrient_name.csv", rows, fieldnames=rows[0].keys())
                foods, _ = load_foods(self.path)
                self.assertEqual(foods, self.foods)

    def test_missing_nutrient_identifiers_rejected(self):
        rows = copy.deepcopy(self.tables["nutrient_name.csv"])
        protein = next(r for r in rows if r["Tagname"] == "PROCNT")
        protein["Tagname"], protein["Nutrient_Name_EN"] = "", "Unknown nutrient"
        self.write_table("nutrient_name.csv", rows)
        with self.assertRaisesRegex(ValueError, "nutrient not found: protein_g"):
            load_cnf_data(self.path)

    def test_ambiguous_standard_tags_rejected(self):
        rows = copy.deepcopy(self.tables["nutrient_name.csv"])
        duplicate = dict(next(r for r in rows if r["Tagname"] == "PROCNT"))
        duplicate["Nutrient_Code"] = "duplicate-protein"
        self.write_table("nutrient_name.csv", rows + [duplicate])
        with self.assertRaisesRegex(ValueError, "Ambiguous CNF nutrient: protein_g"):
            load_cnf_data(self.path)

    def test_ambiguous_english_names_rejected(self):
        rows = copy.deepcopy(self.tables["nutrient_name.csv"])
        for row in rows:
            row["Tagname"] = ""
        duplicate = dict(next(r for r in rows if r["Nutrient_Name_EN"] == "Protein"))
        duplicate["Nutrient_Code"] = "duplicate-protein"
        self.write_table("nutrient_name.csv", rows + [duplicate])
        with self.assertRaisesRegex(ValueError, "Ambiguous CNF nutrient: protein_g"):
            load_cnf_data(self.path)

    def test_duplicate_nutrient_record_rejected(self):
        rows = self.tables["nutrient_amount.csv"]
        self.write_table("nutrient_amount.csv", rows + [rows[0]])
        with self.assertRaisesRegex(ValueError, "Duplicate CNF nutrient record"):
            load_cnf_data(self.path)

    def test_invalid_numeric_amount_rejected(self):
        rows = copy.deepcopy(self.tables["nutrient_amount.csv"])
        rows[0]["Nutrient_Amount"] = "NaN"
        self.write_table("nutrient_amount.csv", rows)
        with self.assertRaisesRegex(ValueError, "Invalid CNF nutrient amount"):
            load_cnf_data(self.path)

    def test_changed_food_description_rejected(self):
        rows = copy.deepcopy(self.tables["food_name.csv"])
        next(r for r in rows if r["Food_Code"] == "4523")["Food_Description_EN"] = (
            "Dry rice"
        )
        self.write_table("food_name.csv", rows)
        with self.assertRaisesRegex(ValueError, "description mismatch"):
            load_cnf_data(self.path)

    def test_json_input_still_supported(self):
        path = self.path / "foods.json"
        path.write_text(
            json.dumps(
                {
                    "foods": [asdict(f) for f in self.foods],
                    "metadata": {"source": "CNF JSON test"},
                }
            ),
            encoding="utf-8",
        )
        foods, metadata = load_foods(path)
        self.assertEqual(metadata["source"], "CNF JSON test")
        self.assertEqual(foods, self.foods)
        self.assertEqual(len(foods), 30)
        with self.assertRaisesRegex(ValueError, "only valid"):
            load_foods(path, self.path / "candidates.json")


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.population = [
            Individual(
                [
                    Meal(
                        [Dish([str(i)], [10], 100)],
                        "rice",
                        Fruit(["apple"], [10], [250 / 3]),
                    )
                ]
            )
            for i in range(6)
        ]
        self.evaluations = [
            Evaluation([], 100, 100, 0.5, "exceeded"),
            Evaluation([], 60, 60, 0, "checked_no_excess"),
            Evaluation([], 100, 100, 0, "not_fully_assessed"),
            Evaluation([], 90, 90, 0, "checked_no_excess"),
            Evaluation([], 50, 50, 0.1, "exceeded"),
            Evaluation([], 100, 100, 0.2, "exceeded"),
        ]

    def test_survivors_use_ul_order_and_deep_copy(self):
        self.assertEqual(
            survivor_selection.rank_population(self.evaluations), [3, 1, 2, 4, 5, 0]
        )
        elites = survivor_selection.select_survivors(
            self.population, self.evaluations, 2
        )
        self.assertEqual(elites, [self.population[3], self.population[1]])
        elites[0].meals[0].dishes[0].units[0] = 9
        elites[0].meals[0].fruit.person_weights_g[0] = 50
        self.assertEqual(self.population[3].meals[0].dishes[0].units, [10])
        self.assertEqual(self.population[3].meals[0].fruit.person_weights_g, [250 / 3])

    def test_parent_selection_runs_two_independent_tournaments(self):
        rng = Mock(spec=random.Random)
        # The second draw overlaps the first but has a different best candidate.
        rng.sample.side_effect = [[0, 1, 2, 4], [0, 2, 3, 5]]
        a, b = parent_selection.select_parents(self.population, self.evaluations, rng)
        self.assertIs(a, self.population[1])
        self.assertIs(b, self.population[3])
        self.assertEqual(rng.sample.call_count, 2)
        for call in rng.sample.call_args_list:
            self.assertEqual(call.args, (range(6), 4))

    def test_both_tournaments_may_select_the_same_parent(self):
        rng = Mock(spec=random.Random)
        rng.sample.side_effect = [[0, 1, 2, 3], [2, 3, 4, 5]]
        a, b = parent_selection.select_parents(self.population, self.evaluations, rng)
        self.assertIs(a, self.population[3])
        self.assertIs(b, self.population[3])

    def test_zero_elites_and_invalid_selection_inputs(self):
        self.assertEqual(
            survivor_selection.select_survivors(self.population, self.evaluations, 0),
            [],
        )
        for count in (-1, 7):
            with self.assertRaises(ValueError):
                survivor_selection.select_survivors(
                    self.population, self.evaluations, count
                )
        with self.assertRaises(ValueError):
            survivor_selection.select_survivors(
                self.population, self.evaluations[:2], 2
            )
        with self.assertRaises(ValueError):
            parent_selection.select_parent(
                self.population, self.evaluations[:2], random.Random(1)
            )
        with self.assertRaises(ValueError):
            parent_selection.select_parent(
                self.population[:3], self.evaluations[:3], random.Random(1)
            )


class MealGATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cnf_foods, _ = load_foods(ROOT / "dataset")

    def setUp(self):
        self.foods = copy.deepcopy(self.cnf_foods)
        self.people = [
            Person("A", 30, "male", 75, 2400),
            Person("B", 28, "female", 60, 2000),
            Person("C", 45, "female", 65, 1800),
        ]
        self.ga = MealGA(
            self.foods, self.people, Config(population_size=12, generations=3), seed=17
        )

    def test_targets_and_staple_formula(self):
        p = self.people[1]
        self.assertEqual(p.targets()["carbohydrate_g"], 275)
        self.assertEqual(p.targets()["protein_g"], 48)
        self.assertAlmostEqual(p.targets()["fat_g"], 2000 * 0.275 / 9)
        for m, carbs in enumerate((55, 88, 77)):
            for fid in self.ga.staples:
                grams = self.ga.staple_weights(m, fid)[1]
                self.assertAlmostEqual(
                    grams / 100 * self.ga.foods[fid].per100g["carbohydrate_g"], carbs
                )

    def test_initialization_budget_no_double_counting(self):
        individual = self.ga.initialize()
        self.assertEqual([len(m.dishes) for m in individual.meals], [3, 4, 3])
        for m, meal in enumerate(individual.meals):
            expected = self.ga.config.side_budgets_g[m] * 3.1
            self.assertTrue(
                0.8 * expected <= sum(d.weight_g for d in meal.dishes) <= 1.2 * expected
            )
            self.assertEqual(meal.fruit.person_weights_g, [250 / 3] * 3)
        self.ga.validate(individual)

    def test_olive_oil_dish_rejects_more_than_half_by_weight(self):
        plan = self.ga.initialize()
        dishes = plan.meals[0].dishes
        grams = dishes[0].weight_g
        dishes[0] = Dish(["422", "3404"], [5, 5], grams)
        self.ga.validate(plan)
        for ids, units in ((["422", "3404"], [7, 3]), (["422"], [10])):
            with self.subTest(units=units):
                dishes[0] = Dish(ids, units, grams)
                with self.assertRaises(ValueError):
                    self.ga.validate(plan)

    def test_random_dishes_respect_olive_oil_cap(self):
        oil_units = [
            units
            for dish in (self.ga.new_dish(0) for _ in range(300))
            for fid, units in zip(dish.food_ids, dish.units)
            if fid == "422"
        ]
        self.assertTrue(oil_units)
        self.assertLessEqual(max(oil_units), 5)

    def test_different_household_sizes(self):
        for n in (1, 2, 5):
            people = [Person(str(i), 30, "male", 70, 2000) for i in range(n)]
            ga = MealGA(self.foods, people, seed=5)
            individual = ga.initialize()
            self.assertEqual([len(m.dishes) for m in individual.meals], [n, n + 1, n])
            for mi in range(3):
                self.assertAlmostEqual(
                    ga.baseline(mi) * ga.slot_count(mi),
                    ga.config.side_budgets_g[mi] * n,
                )
            for pi in range(n):
                self.assertAlmostEqual(
                    sum(
                        g
                        for _, module, _, g in ga.portions(individual, pi)
                        if module == "fruit"
                    ),
                    250,
                )
            ga.evaluate(individual)

    def test_nutrition_counts_all_sources(self):
        individual = self.ga.initialize()
        evaluation = self.ga.evaluate(individual)
        for pi, person in enumerate(self.people):
            totals = {k: 0 for k in NUTRIENTS}
            for mi, meal in enumerate(individual.meals):
                contributions = [
                    (meal.staple_id, self.ga.staple_weights(mi, meal.staple_id)[pi])
                ]
                for dish in meal.dishes:
                    for fid, u in zip(dish.food_ids, dish.units):
                        contributions.append(
                            (fid, dish.weight_g * person.factor / 3.1 * u / 10)
                        )
                for fid, u in zip(meal.fruit.food_ids, meal.fruit.units):
                    contributions.append(
                        (fid, meal.fruit.person_weights_g[pi] * u / 10)
                    )
                for fid, grams in contributions:
                    for key in NUTRIENTS:
                        totals[key] += grams / 100 * self.ga.foods[fid].per100g[key]
            for key in NUTRIENTS:
                self.assertAlmostEqual(
                    evaluation.people[pi]["totals"][key], totals[key]
                )
            self.assertGreater(
                evaluation.people[pi]["totals"]["carbohydrate_g"],
                0.8 * person.targets()["carbohydrate_g"],
            )

    def test_unknown_ul_and_reference_forms_are_separate(self):
        evaluation = self.ga.evaluate(self.ga.initialize())
        for person in evaluation.people:
            self.assertEqual(
                person["ul_checks"]["ul_applicable_e_mg"]["status"], "not_assessed"
            )
            self.assertIsNone(person["totals"]["ul_applicable_e_mg"])
        # High total A/DFE from plant sources must not cause preformed-A/folic-acid excess.
        altered = []
        for food in self.foods:
            nutrients = dict(food.per100g)
            nutrients.update(
                vitamin_a_rae_ug=10000,
                folate_dfe_ug=10000,
                preformed_a_ug=0,
                synthetic_folic_acid_ug=0,
                ul_applicable_e_mg=0,
            )
            altered.append(
                Food(
                    food.id,
                    food.name,
                    food.category,
                    food.role,
                    nutrients,
                    food.mix_group,
                )
            )
        ga = MealGA(altered, self.people, seed=10)
        ev = ga.evaluate(ga.initialize())
        for p in ev.people:
            self.assertEqual(p["scores"]["vitamin_a_rae_ug"], 4)
            self.assertEqual(p["ul_checks"]["preformed_a_ug"]["excess_ratio"], 0)
            self.assertEqual(
                p["ul_checks"]["synthetic_folic_acid_ug"]["excess_ratio"], 0
            )

    def test_ul_precedes_score_and_excess_is_normalized(self):
        safe = Evaluation([], 50, 50, 0, "checked_no_excess")
        unknown = Evaluation([], 100, 100, 0, "not_fully_assessed")
        excess = Evaluation([], 100, 99, 0.1, "exceeded")
        worse = Evaluation([], 100, 100, 0.2, "exceeded")
        self.assertLess(safe.key, unknown.key)
        self.assertLess(unknown.key, excess.key)
        self.assertLess(excess.key, worse.key)
        foods = [
            Food(
                f.id,
                f.name,
                f.category,
                f.role,
                {k: (UL[k] * 2 if k in UL else f.per100g.get(k)) for k in ALL_FIELDS},
                f.mix_group,
            )
            for f in self.foods
        ]
        ga = MealGA(foods, self.people, seed=10)
        ev = ga.evaluate(ga.initialize())
        expected = sum(
            max(0, p["totals"][k] / limit - 1)
            for p in ev.people
            for k, limit in UL.items()
        )
        self.assertEqual(ev.ul_status, "exceeded")
        self.assertAlmostEqual(ev.excess, expected)
        self.assertAlmostEqual(ev.fitness, ev.base_score - 10 * expected / 3)

    def test_crossover_and_repeated_mutations_preserve_constraints(self):
        a, b = self.ga.initialize(), self.ga.initialize()
        before = copy.deepcopy(a)
        kind_events, portion_events = set(), set()
        for _ in range(1000):
            child = crossover_operator.crossover(self.ga, a, b)
            kind_events.add(mutation.mutate_kind(self.ga, child))
            portion_events.add(mutation.mutate_portions(self.ga, child))
            self.ga.validate(child)
            self.assertTrue(all(
                units <= 5
                for meal in child.meals for dish in meal.dishes
                for fid, units in zip(dish.food_ids, dish.units) if fid == "422"
            ))
            b = child
        self.assertEqual(a, before)
        self.assertEqual(
            kind_events, {"none", "same_category", "copy_other", "delete", "add"}
        )
        self.assertEqual(
            portion_events,
            {"none", "side_ratio", "side_weight", "fruit_ratio", "fruit_transfer"},
        )

    def test_bounded_fruit_repair(self):
        for weights in ([0, 0, 0], [1, 1, 10000], [150, 150, 50], [50, 50, 150]):
            repaired = bounded_fruit_weights(list(weights))
            self.assertAlmostEqual(sum(repaired), 250)
            self.assertTrue(all(50 - 1e-7 <= w <= 150 + 1e-7 for w in repaired))

    def test_missing_data_does_not_hide_known_ul_excess(self):
        foods = [
            Food(
                f.id,
                f.name,
                f.category,
                f.role,
                {
                    **f.per100g,
                    "ul_applicable_e_mg": 100000 if f.role == "staple" else None,
                },
                f.mix_group,
            )
            for f in self.foods
        ]
        ga = MealGA(foods, self.people, seed=9)
        ev = ga.evaluate(ga.initialize())
        self.assertEqual(ev.ul_status, "exceeded")
        for p in ev.people:
            check = p["ul_checks"]["ul_applicable_e_mg"]
            self.assertIsNone(check["actual"])
            self.assertFalse(check["fully_assessed"])
            self.assertEqual(check["status"], "exceeded")
            self.assertGreater(check["known_lower_bound"], 1000)

    def test_staple_type_mutation_recalculates_weight(self):
        ga = MealGA(
            self.foods, self.people, Config(staple_mutation_probability=1), seed=42
        )
        individual = ga.initialize()
        before = [m.staple_id for m in individual.meals]
        mutation.mutate_staple(ga, individual)
        self.assertEqual(
            sum(a != m.staple_id for a, m in zip(before, individual.meals)), 1
        )
        for mi, meal in enumerate(individual.meals):
            for pi, grams in enumerate(ga.staple_weights(mi, meal.staple_id)):
                self.assertAlmostEqual(
                    grams / 100 * ga.foods[meal.staple_id].per100g["carbohydrate_g"],
                    self.people[pi].targets()["carbohydrate_g"]
                    * 0.8
                    * ga.config.meal_shares[mi],
                )

    def test_run_reproducibility_statistics_and_elitism(self):
        plans, history = self.ga.run()
        other = MealGA(self.foods, self.people, self.ga.config, seed=17)
        plans2, history2 = other.run()
        self.assertEqual(history, history2)
        self.assertEqual(
            [self.ga.signature(i) for i, _ in plans],
            [other.signature(i) for i, _ in plans2],
        )
        self.assertEqual(len(history), 4)
        self.assertEqual([h["generation"] for h in history], [0, 1, 2, 3])
        self.assertTrue(1 <= len(plans) <= 3)
        self.assertEqual(len({self.ga.signature(i) for i, _ in plans}), len(plans))
        self.assertTrue(all(0 <= e.base_score <= 100 + 1e-7 for _, e in plans))
        for individual, ev in plans:
            self.ga.validate(individual)
            recomputed = self.ga.evaluate(individual)
            self.assertEqual(ev.ul_status, recomputed.ul_status)
            for field in ("base_score", "fitness", "excess"):
                self.assertAlmostEqual(
                    getattr(ev, field), getattr(recomputed, field), places=9
                )
            for a, b in zip(ev.people, recomputed.people):
                for key, value in a["totals"].items():
                    if value is None:
                        self.assertIsNone(b["totals"][key])
                    else:
                        self.assertAlmostEqual(value, b["totals"][key], places=9)
        # For fully assessed, no-excess data, elite preservation makes best score monotone.
        zero_ul = [
            Food(
                f.id,
                f.name,
                f.category,
                f.role,
                {**f.per100g, **{k: 0 for k in UL}},
                f.mix_group,
            )
            for f in self.foods
        ]
        _, safe_history = MealGA(zero_ul, self.people, self.ga.config, seed=4).run()
        scores = [h["best_score"] for h in safe_history]
        self.assertEqual(scores, sorted(scores))

    def test_reject_missing_data_and_children(self):
        broken = copy.deepcopy(self.foods)
        broken[0].per100g.pop("protein_g")
        with self.assertRaises(ValueError):
            MealGA(broken, self.people)
        with self.assertRaises(ValueError):
            Person("child", 17, "female", 60, 2000)


def scalar_reference(ga, individual):
    people, excess, fully_checked = [], 0.0, True
    for pi, person in enumerate(ga.people):
        portions = [
            (ga.foods[fid], grams)
            for _, _, fid, grams in ga.portions(individual, pi)
            if grams > 0
        ]
        totals, known = {}, {}
        for field in ALL_FIELDS:
            known[field] = sum(
                grams / 100 * food.per100g[field]
                for food, grams in portions
                if food.per100g.get(field) is not None
            )
            totals[field] = (
                None
                if any(food.per100g.get(field) is None for food, _ in portions)
                else known[field]
            )
        targets = person.targets()
        scores = {k: WEIGHTS[k] * min(totals[k] / targets[k], 1) for k in NUTRIENTS}
        checks = {}
        for field, limit in UL.items():
            ratio = max(0, known[field] / limit - 1)
            if totals[field] is None and ratio <= EPS:
                ratio = None
            checks[field] = {
                "actual": totals[field],
                "limit": limit,
                "excess_ratio": ratio,
                "known_lower_bound": known[field],
                "fully_assessed": totals[field] is not None,
                "status": "not_assessed"
                if ratio is None
                else ("exceeded" if ratio > EPS else "checked_no_excess"),
            }
            excess += ratio or 0
            fully_checked &= totals[field] is not None
        people.append(
            {
                "name": person.name,
                "totals": totals,
                "targets": targets,
                "scores": scores,
                "score": sum(scores.values()),
                "deficits": {k: max(0, targets[k] - totals[k]) for k in NUTRIENTS},
                "ul_checks": checks,
            }
        )
    base = sum(p["score"] for p in people) / len(people)
    status = (
        "exceeded"
        if excess > EPS
        else ("checked_no_excess" if fully_checked else "not_fully_assessed")
    )
    return Evaluation(
        people,
        base,
        base - ga.config.excess_penalty * excess / len(people),
        excess,
        status,
    )


class TorchFitnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.foods, _ = load_foods(ROOT / "dataset")
        cls.people = [
            Person("A", 30, "male", 75, 2400),
            Person("B", 28, "female", 60, 2000),
        ]

    def setUp(self):
        self.ga = MealGA(self.foods, self.people, seed=29)

    def assert_tree_close(self, a, b):
        if isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for k in a:
                self.assert_tree_close(a[k], b[k])
        elif isinstance(a, list):
            self.assertEqual(len(a), len(b))
            for va, vb in zip(a, b):
                self.assert_tree_close(va, vb)
        elif isinstance(a, float):
            self.assertTrue(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-8), (a, b))
        else:
            self.assertEqual(a, b)

    def test_population_tensor_shape_dtype_and_conservation(self):
        population = [self.ga.initialize() for _ in range(4)]
        grams = self.ga.population_weights(population)
        self.assertEqual(grams.shape, (4, 2, 30))
        if self.ga.backend == "torch":
            self.assertEqual(grams.dtype, torch.float64)
            self.assertEqual(grams.device.type, "cpu")
            self.assertFalse(grams.requires_grad)
        else:
            self.assertEqual(str(grams.dtype), "float64")
            self.assertEqual(self.ga.backend_info()["device"], "cpu")
        for b, individual in enumerate(population):
            for p in range(2):
                expected = sum(g for _, _, _, g in self.ga.portions(individual, p))
                self.assertAlmostEqual(grams[b, p].sum().item(), expected)

    def test_batch_and_single_match_scalar_reference(self):
        population = [self.ga.initialize() for _ in range(12)]
        for individual in population:
            mutation.mutate_kind(self.ga, individual)
            mutation.mutate_portions(self.ga, individual)
        evaluations = self.ga.evaluate_population(population)
        for individual, batch in zip(population, evaluations):
            self.assert_tree_close(
                asdict(batch), asdict(scalar_reference(self.ga, individual))
            )
            self.assert_tree_close(asdict(batch), asdict(self.ga.evaluate(individual)))

    def test_empty_population(self):
        self.assertEqual(self.ga.evaluate_population([]), [])
        self.assertEqual(self.ga.population_weights([]).shape, (0, 2, 30))

    def test_unconsumed_unknown_food_does_not_propagate_missingness(self):
        fruits = [f.id for f in self.foods if f.role == "fruit"]
        unknown, known = fruits[:2]
        foods = [
            Food(
                f.id,
                f.name,
                f.category,
                f.role,
                {**f.per100g, "ul_applicable_e_mg": None if f.id == unknown else 0},
                f.mix_group,
            )
            for f in self.foods
        ]
        ga = MealGA(foods, self.people, seed=29)
        individual = ga.initialize()
        for meal in individual.meals:
            meal.fruit.food_ids, meal.fruit.units = [known], [10]
        for person in ga.evaluate(individual).people:
            self.assertEqual(
                person["ul_checks"]["ul_applicable_e_mg"]["status"], "checked_no_excess"
            )
            self.assertEqual(person["totals"]["ul_applicable_e_mg"], 0)
        individual.meals[0].fruit.food_ids = [known, unknown]
        individual.meals[0].fruit.units = [5, 5]
        for person in ga.evaluate(individual).people:
            self.assertEqual(
                person["ul_checks"]["ul_applicable_e_mg"]["status"], "not_assessed"
            )
            self.assertIsNone(person["totals"]["ul_applicable_e_mg"])

    def test_re_evaluation_uses_current_portions(self):
        individual = self.ga.initialize()
        old = self.ga.evaluate(individual)
        individual.meals[0].dishes[0].weight_g *= 1.1
        new = self.ga.evaluate(individual)
        for a, b in zip(old.people, new.people):
            self.assertGreater(b["totals"]["energy_kcal"], a["totals"]["energy_kcal"])
        self.assert_tree_close(
            asdict(new), asdict(scalar_reference(self.ga, individual))
        )

    @unittest.skipIf(torch is None, "Optional PyTorch backend is not installed")
    def test_auto_device_and_unsupported_device(self):
        auto = resolve_device("auto")
        self.assertEqual(auto.type, "cuda" if torch.cuda.is_available() else "cpu")
        for device in ("invalid-device", "mps", "meta"):
            with self.assertRaises(ValueError):
                resolve_device(device)
        if not torch.cuda.is_available():
            with self.assertRaisesRegex(ValueError, "CUDA is not available"):
                resolve_device("cuda")

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_matches_cpu(self):
        gpu = MealGA(self.foods, self.people, Config(device="cuda"), seed=29)
        population = [self.ga.initialize() for _ in range(8)]
        self.assertEqual(gpu.nutrient_matrix.device.type, "cuda")
        for cpu, cuda in zip(
            self.ga.evaluate_population(population), gpu.evaluate_population(population)
        ):
            self.assert_tree_close(asdict(cpu), asdict(cuda))


if __name__ == "__main__":
    unittest.main()
