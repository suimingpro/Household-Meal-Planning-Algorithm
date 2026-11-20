"""Adult household three-meal genetic algorithm: organize genetic operations and use PyTorch for batch nutrient and fitness calculations."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:
    torch = None

import device_utils
from cnf_data import load_cnf_data
from ga_models import (
    ALL_FIELDS,
    CATEGORIES,
    EPS,
    MEALS,
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
from genetic_operators import evolution


def load_foods(
    path: str | Path, candidates_path: str | Path | None = None
) -> tuple[list[Food], dict[str, Any]]:
    """Load candidate foods from a CNF data directory or custom JSON file.

    Args:
        path: Directory containing CNF relational tables, or a JSON file
            containing foods and optional metadata.
        candidates_path: CNF candidate-list path; defaults to candidates.json
            in the data directory. Cannot be specified when reading custom food JSON.

    Returns:
        A pair containing a list of food objects and metadata such as data
        sources and field mappings.

    Raises:
        OSError: An input file does not exist or cannot be read.
        ValueError: Invalid JSON format, failed CNF data validation, or improper
            use of the candidate-list argument.
        KeyError: Required input fields are missing.
    """
    if Path(path).is_dir():
        rows, metadata = load_cnf_data(path, candidates_path)
        return [Food(**row) for row in rows], metadata
    if candidates_path is not None:
        raise ValueError("--candidates is only valid with a CNF dataset directory")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Food(**row) for row in data["foods"]], data.get("metadata", {})


def bounded_fruit_weights(values: list[float]) -> list[float]:
    """Repair one member's fruit weights to 250 g per day and 50–150 g per meal.

    First scale proportionally to 250 g, then find a common offset by binary
    search to preserve the total while satisfying lower and upper bounds.
    Start from equal meal shares if all inputs are 0. The original list is not modified.

    Args:
        values: Three nonnegative finite weights in g, ordered as breakfast,
            lunch, and dinner.

    Returns:
        Three repaired floating-point weights totaling 250 g within numerical precision.

    Raises:
        ValueError: Length is not 3, or values include negatives, infinity, or NaN.
    """
    if len(values) != 3 or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("Three finite nonnegative fruit weights required")
    total = sum(values)
    scaled = [v * 250 / total for v in values] if total else [250 / 3] * 3
    lo, hi = min(scaled) - 150, max(scaled) - 50
    for _ in range(80):
        shift = (lo + hi) / 2
        if sum(min(150, max(50, v - shift)) for v in scaled) > 250:
            lo = shift
        else:
            hi = shift
    return [min(150, max(50, v - (lo + hi) / 2)) for v in scaled]


class MealGA:
    """Genetic algorithm runner that generates and optimizes one day of three meals for an adult household.

    Handles input validation, individual initialization, portion allocation,
    PyTorch batch scoring, evolution, and result organization; parent selection,
    survivor selection, crossover, and mutation are delegated to genetic_operators.
    Each instance uses an independent random number generator. Nutrient calculations
    use float64 without constructing a gradient graph.

    Attributes:
        config: Runtime parameters such as population size, generations,
            probabilities, portions, and device.
        rng: random.Random instance initialized from seed and shared by all
            genetic operations.
        people: Adult members in input order.
        foods: Candidate food dictionary keyed by food code.
        device: PyTorch computing device actually used.
        nutrient_matrix: Per-100-g nutrient matrix shaped [food count, nutrient field count].
        missing_matrix: Missing-value markers with the same shape as the nutrient
            matrix, used to propagate unknown UL status.
        target_matrix: Individual daily target matrix shaped [member count,
            scored nutrient count].
    """

    def __init__(
        self,
        foods: list[Food],
        people: list[Person],
        config: Config | None = None,
        seed: int | None = None,
        *,
        backend: str = "auto",
    ):
        """Validate members and the candidate pool, and construct fixed nutrient, target, and UL tensors.

        Args:
            foods: 25–40 foods with unique codes, including exactly three staples
                and at least two fruits. Each mixing compatibility group has at
                least four foods; nutrient values required for scoring must not be missing.
            people: Nonempty adult member list with unique member names.
            config: Algorithm configuration; uses Config defaults when omitted.
            seed: Random seed for genetic operations; None means no fixed seed.
            backend: auto prefers the existing PyTorch backend and uses NumPy CPU
                when PyTorch is not installed.

        Raises:
            ValueError: Members, candidate pool, nutrient values, or device
                configuration do not meet requirements.
        """
        self.config = config or Config()
        self.rng = random.Random(seed)
        self.people = people
        if not people or len({p.name for p in people}) != len(people):
            raise ValueError("At least one adult and unique member names required")
        self.foods = {f.id: f for f in foods}
        if not 25 <= len(foods) <= 40 or len(self.foods) != len(foods):
            raise ValueError("Provide 25–40 foods with unique IDs")
        for f in foods:
            if f.category not in CATEGORIES or f.role not in {
                "staple",
                "fruit",
                "mixed",
                "standalone",
            }:
                raise ValueError(f"Invalid food category/role: {f.id}")
            if (f.role == "fruit") != (f.category == "fruit"):
                raise ValueError(f"Fruit role/category mismatch: {f.id}")
            if f.role == "staple" and f.category != "starch":
                raise ValueError(f"Staple must be starch: {f.id}")
            for k in ALL_FIELDS:
                value = f.per100g.get(k)
                if value is None:
                    if k not in {
                        "preformed_a_ug",
                        "ul_applicable_e_mg",
                        "synthetic_folic_acid_ug",
                    }:
                        raise ValueError(
                            f"Missing nutrient {k}: {f.id}; unknown is not zero"
                        )
                elif (
                    not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError(f"Invalid nutrient {k}: {f.id}")
        self.staples = [f.id for f in foods if f.role == "staple"]
        self.fruits = [f.id for f in foods if f.role == "fruit"]
        self.sides = [f.id for f in foods if f.role in {"mixed", "standalone"}]
        if len(self.staples) != 3 or len(self.fruits) < 2 or not self.sides:
            raise ValueError(
                "Exactly 3 staples, at least 2 fruits, and side foods required"
            )
        if any(self.foods[f].per100g["carbohydrate_g"] <= 0 for f in self.staples):
            raise ValueError("Staples require positive carbohydrate values")
        groups: dict[str, int] = {}
        for f in foods:
            if f.role == "mixed":
                groups[f.mix_group] = groups.get(f.mix_group, 0) + 1
        if any(v < 4 for v in groups.values()):
            raise ValueError(
                "Each mixed-food compatibility group needs at least 4 foods"
            )
        for group in groups:
            limits = sorted(
                (f.max_percent for f in foods if f.role == "mixed" and f.mix_group == group),
                reverse=True,
            )
            if sum(limits[:4]) < 100:
                raise ValueError(f"Mixed-food group {group}: proportion caps cannot sum to 100% within 4 foods")
        self.factor_sum = sum(p.factor for p in people)
        if backend not in {"auto", "numpy", "torch"}:
            raise ValueError(f"Unknown scoring backend: {backend}")
        self.backend = ("torch" if torch is not None else "numpy") if backend == "auto" else backend
        if self.backend == "torch":
            if torch is None:
                raise ValueError("The torch backend requires PyTorch to be installed")
            self.device = device_utils.resolve_device(self.config.device)
            self.dtype = torch.float64
        else:
            if self.config.device not in {"cpu", "auto"}:
                raise ValueError("NumPy only supports CPU; CUDA requires PyTorch")
            self.device = "cpu"
            self.dtype = np.dtype("float64")
        self.food_index = {fid: i for i, fid in enumerate(self.foods)}
        self.field_index = {name: i for i, name in enumerate(ALL_FIELDS)}
        # Missing UL fields have a separate mask. Their zeros in the arithmetic
        # matrix are only used to calculate a known lower bound, never a pass.
        self.nutrient_matrix = self._array([
            [f.per100g.get(k) if f.per100g.get(k) is not None else 0.0 for k in ALL_FIELDS]
            for f in self.foods.values()
        ])
        self.missing_matrix = self._array([
            [f.per100g.get(k) is None for k in ALL_FIELDS] for f in self.foods.values()
        ])
        self.nutrient_columns = [self.field_index[k] for k in NUTRIENTS]
        self.ul_columns = [self.field_index[k] for k in UL]
        self.target_matrix = self._array([[p.targets()[k] for k in NUTRIENTS] for p in self.people])
        self.score_weights = self._array(list(WEIGHTS.values()))
        self.ul_limits = self._array(list(UL.values()))

    def _array(self, values):
        if self.backend == "torch":
            return torch.tensor(values, dtype=self.dtype, device=self.device)
        return np.asarray(values, dtype=self.dtype)

    def backend_info(self) -> dict[str, str]:
        """Return the actual evaluator and numeric environment used for output."""
        return {
            "framework": "pytorch" if self.backend == "torch" else "numpy",
            "version": str(torch.__version__) if self.backend == "torch" else np.__version__,
            "device": str(self.device),
            "dtype": str(self.dtype),
        }

    def slot_count(self, meal: int) -> int:
        """Return the side-dish slot count for a meal: n for breakfast/dinner, n+1 for lunch.

        Args:
            meal: Meal index: 0 for breakfast, 1 for lunch, 2 for dinner.

        Returns:
            Number of shared household side-dish combinations for this meal.
        """
        return len(self.people) + (meal == 1)

    def baseline(self, meal: int) -> float:
        """Calculate baseline weight B in g for each side-dish slot of a meal.

        Args:
            meal: Meal index, 0, 1, or 2.

        Returns:
            The reference adult's meal budget multiplied by the sum of household
            portion factors, divided by slot count. Member count only enters
            through portion factors and slot count; it is not multiplied again.
        """
        return (
            self.config.side_budgets_g[meal] * self.factor_sum / self.slot_count(meal)
        )

    def staple_weights(self, meal: int, food_id: str) -> list[float]:
        """Calculate staple weights immediately from individual carbohydrate targets and meal shares.

        Args:
            meal: Meal index, 0, 1, or 2.
            food_id: Selected staple food code; its carbohydrate content per
                100 g must be greater than 0.

        Returns:
            Edible weights in g in member order. Each person's weight equals
            daily carbohydrate target × staple supply fraction × meal share
            ÷ carbohydrate content per 100 g × 100.
        """
        c = self.foods[food_id].per100g["carbohydrate_g"]
        return [
            p.targets()["carbohydrate_g"]
            * self.config.staple_fraction
            * self.config.meal_shares[meal]
            / c
            * 100
            for p in self.people
        ]

    def compatible(self, ids: list[str]) -> bool:
        """Determine whether foods can form a side-dish combination, without checking actual proportions or weights.

        Args:
            ids: List of food codes already present in the candidate pool.

        Returns:
            True for a standalone item or a mixable food appearing alone.
            Multiple foods must all have the mixed role, belong to the same
            mix_group, and have proportion limits summing to at least 100%.
            Restricted foods cannot appear alone; fruit and staples are not
            mixed as side dishes.
        """
        fs = [self.foods[f] for f in ids]
        return sum(f.max_percent for f in fs) >= 100 and ((len(fs) == 1 and fs[0].role in {"mixed", "standalone"}) or (
            all(f.role == "mixed" for f in fs) and len({f.mix_group for f in fs}) == 1
        ))

    def new_dish(self, meal: int) -> Dish:
        """Randomly create a side-dish combination that meets role compatibility rules.

        Args:
            meal: Meal index used to determine baseline weight B for the slot.

        Returns:
            A Dish containing food codes, proportion units, and total household
            weight. The mixed ingredient count is obtained by rounding a
            truncated normal N(2.5, 1) draw to 1–4; standalone items always use one.
            Proportions use 10% units and sum to 100%; weight is sampled uniformly
            within 0.8B–1.2B. Only ingredient combinations permitted by proportion
            limits are accepted; remaining proportions are allocated only to
            ingredients that have not reached their limits.
        """
        first = self.rng.choice(self.sides)
        food = self.foods[first]
        if food.role == "standalone":
            ids = [first]
        else:
            pool = [
                f
                for f in self.sides
                if self.foods[f].role == "mixed"
                and self.foods[f].mix_group == food.mix_group
            ]
            capacities = sorted((self.foods[f].max_percent for f in pool), reverse=True)
            while True:
                sample = self.rng.gauss(2.5, 1)
                if 0.5 <= sample < 4.5:
                    n = math.floor(sample + 0.5)
                    if sum(capacities[:n]) >= 100:
                        break
            ids = self.rng.sample(pool, n)
            while not self.compatible(ids):
                ids = self.rng.sample(pool, n)
        units = [1] * len(ids)
        for _ in range(10 - len(ids)):
            available = [i for i, fid in enumerate(ids) if units[i] < self.foods[fid].max_percent // 10]
            units[self.rng.choice(available)] += 1
        return Dish(ids, units, self.baseline(meal) * self.rng.uniform(0.8, 1.2))

    def initialize(self) -> Individual:
        """Generate and validate one random individual, not the entire population.

        Returns:
            An Individual containing three meals. Each meal selects its staple
            independently. One or two fruits are selected with equal probability;
            two fruits each occupy 50%. Each person's 250 g daily fruit is
            initially divided equally among three meals.
        """
        meals = []
        for m in range(3):
            fruit_ids = self.rng.sample(
                self.fruits, 1 if self.rng.random() < 0.5 else 2
            )
            fruit = Fruit(
                fruit_ids,
                [10] if len(fruit_ids) == 1 else [5, 5],
                [250 / 3] * len(self.people),
            )
            meals.append(
                Meal(
                    [self.new_dish(m) for _ in range(self.slot_count(m))],
                    self.rng.choice(self.staples),
                    fruit,
                )
            )
        result = Individual(meals)
        self.validate(result)
        return result

    def validate(self, individual: Individual) -> None:
        """Check an individual's combination counts, food roles, proportions, weights, and fruit totals.

        Args:
            individual: Complete three-meal individual to check; checking does
                not modify its contents.

        Raises:
            ValueError: Incorrect meal or slot count, duplicate or invalid codes,
                incompatible foods, invalid proportions, side-dish weight outside
                0.5B–1.5B, or fruit violating the per-person constraints of 250 g
                per day and 50–150 g per meal.
        """
        if len(individual.meals) != 3:
            raise ValueError("Exactly three meals required")
        for m, meal in enumerate(individual.meals):
            if (
                len(meal.dishes) != self.slot_count(m)
                or meal.staple_id not in self.staples
            ):
                raise ValueError("Invalid meal slots/staple")
            for module in [*meal.dishes, meal.fruit]:
                fruit = isinstance(module, Fruit)
                minimum, maximum = (2, 2) if fruit else (1, 4)
                if (
                    not 1 <= len(module.food_ids) <= maximum
                    or len(set(module.food_ids)) != len(module.food_ids)
                    or len(module.units) != len(module.food_ids)
                    or any(f not in self.foods for f in module.food_ids)
                    or any(type(u) is not int or u < minimum for u in module.units)
                    or sum(module.units) != 10
                ):
                    raise ValueError("Invalid food IDs or proportion units")
                if fruit:
                    if any(f not in self.fruits for f in module.food_ids):
                        raise ValueError("Non-fruit in fruit module")
                else:
                    if any(u * 10 > self.foods[fid].max_percent for fid, u in zip(module.food_ids, module.units)):
                        raise ValueError("Ingredient proportion exceeds max_percent")
                    if not self.compatible(module.food_ids):
                        raise ValueError("Incompatible side ingredients")
            for dish in meal.dishes:
                b = self.baseline(m)
                if (
                    not math.isfinite(dish.weight_g)
                    or not 0.5 * b - EPS <= dish.weight_g <= 1.5 * b + EPS
                ):
                    raise ValueError("Dish weight outside slot bounds")
            weights = meal.fruit.person_weights_g
            if len(weights) != len(self.people) or any(
                not math.isfinite(w) or not 50 - EPS <= w <= 150 + EPS for w in weights
            ):
                raise ValueError("Fruit meal weights outside bounds")
        for i in range(len(self.people)):
            if not math.isclose(
                sum(m.fruit.person_weights_g[i] for m in individual.meals),
                250,
                abs_tol=EPS,
            ):
                raise ValueError("Daily fruit must equal 250g per adult")

    def repair_fruit(self, individual: Individual) -> None:
        """Repair each person's three-meal fruit weights in place, preserving ingredient codes and within-meal proportions.

        Args:
            individual: Individual to repair, usually a child after inheriting
                fruit configurations across meals.

        Notes:
            After repair, each member has 250 g of fruit per day and 50–150 g per
            meal. This method only repairs weights; it does not repair other
            invalid encodings or calculate nutrients.
        """
        for p in range(len(self.people)):
            weights = bounded_fruit_weights(
                [m.fruit.person_weights_g[p] for m in individual.meals]
            )
            for meal, w in zip(individual.meals, weights):
                meal.fruit.person_weights_g[p] = w

    def portions(self, individual: Individual, person: int):
        """Yield the edible grams of staples, side dishes, and fruit allocated to a member, item by item.

        Args:
            individual: Individual containing three meals.
            person: Member index in people.

        Yields:
            (meal index, module name, food code, actual grams). Staple weights
            follow the formula; shared side dishes are allocated in proportion
            to the member's E/2000 relative to the sum of household factors.
            Fruit is allocated according to the member's meal weight and
            ingredient proportions. The same food may be yielded multiple times.
        """
        fraction = self.people[person].factor / self.factor_sum
        for m, meal in enumerate(individual.meals):
            yield (
                m,
                "staple",
                meal.staple_id,
                self.staple_weights(m, meal.staple_id)[person],
            )
            for j, dish in enumerate(meal.dishes):
                for fid, units in zip(dish.food_ids, dish.units):
                    yield m, f"dish_{j + 1}", fid, dish.weight_g * fraction * units / 10
            for fid, units in zip(meal.fruit.food_ids, meal.fruit.units):
                yield m, "fruit", fid, meal.fruit.person_weights_g[person] * units / 10

    def evaluate(self, individual: Individual) -> Evaluation:
        """Evaluate one individual through the shared batch-scoring logic.

        Args:
            individual: Individual to validate and recalculate nutrients for.

        Returns:
            Per-member nutrient details, household base score, fitness, total
            normalized excess, and UL status.
        """
        return self.evaluate_population([individual])[0]

    def population_weights(self, population: list[Individual]) -> torch.Tensor | np.ndarray:
        """Validate a population and encode per-person food intakes as a float64 tensor.

        Args:
            population: List of individuals to encode; may be empty.

        Returns:
            A gram-weight tensor on self.device, shaped [population size,
            member count, food count]. Food column order is determined by
            food_index. Foods with the same code across meals and modules are
            accumulated into the same column; an empty population retains the
            member-count and food-count dimensions.

        Raises:
            ValueError: Any individual fails validity checks.
        """
        if not population:
            return self._array([]).reshape(0, len(self.people), len(self.foods))
        weights = []
        for individual in population:
            self.validate(individual)
            members = []
            for p in range(len(self.people)):
                row = [0.0] * len(self.foods)
                for _, _, fid, grams in self.portions(individual, p):
                    row[self.food_index[fid]] += grams
                members.append(row)
            weights.append(members)
        return self._array(weights)

    def evaluate_population(self, population: list[Individual]) -> list[Evaluation]:
        """Calculate nutrients, adequacy scores, deficits, and UL excess for an entire generation in batches.

        Args:
            population: Individuals to rescore, without using cached parent
                nutrient results.

        Returns:
            Evaluation objects in input order; empty input returns an empty list.
            The household base score is the mean member score, and fitness
            subtracts the per-person normalized excess penalty. Incomplete UL
            data retain unknown status; excess is still identified when the
            known intake lower bound exceeds the limit.

        Raises:
            ValueError: Any individual fails validity checks.

        Notes:
            All nutrients from all food sources are accumulated. Calculations
            use float64 with automatic differentiation disabled, returning
            ordinary Python objects for selection, recording, and JSON serialization.
        """
        if not population:
            return []
        grams = self.population_weights(population)
        if self.backend == "torch":
            batch = self._torch_scores(grams)
        else:
            batch = self._numpy_scores(grams)
        (totals_cpu, missing_cpu, scores_cpu, deficits_cpu, person_scores_cpu,
         ratios_cpu, assessed_cpu, summary_cpu, checked_cpu) = batch
        evaluations = []
        for b in range(len(population)):
            results = []
            for p, person in enumerate(self.people):
                totals = {
                    k: None if missing_cpu[b][p][j] else totals_cpu[b][p][j]
                    for j, k in enumerate(ALL_FIELDS)
                }
                checks = {}
                for j, (k, limit) in enumerate(UL.items()):
                    ratio = ratios_cpu[b][p][j] if assessed_cpu[b][p][j] else None
                    checks[k] = {
                        "actual": totals[k],
                        "limit": limit,
                        "excess_ratio": ratio,
                        "known_lower_bound": totals_cpu[b][p][self.field_index[k]],
                        "fully_assessed": totals[k] is not None,
                        "status": "not_assessed"
                        if ratio is None
                        else ("exceeded" if ratio > EPS else "checked_no_excess"),
                    }
                results.append(
                    {
                        "name": person.name,
                        "totals": totals,
                        "targets": person.targets(),
                        "scores": dict(zip(NUTRIENTS, scores_cpu[b][p])),
                        "score": person_scores_cpu[b][p],
                        "deficits": dict(zip(NUTRIENTS, deficits_cpu[b][p])),
                        "ul_checks": checks,
                    }
                )
            base, fit, excess = summary_cpu[b]
            status = (
                "exceeded"
                if excess > EPS
                else ("checked_no_excess" if checked_cpu[b] else "not_fully_assessed")
            )
            evaluations.append(Evaluation(results, base, fit, excess, status))
        return evaluations

    def _torch_scores(self, grams):
        """Original PyTorch float64 batch arithmetic, with no gradient tracking."""
        with torch.inference_mode():
            # [B, P, F] @ [F, K] -> [B, P, K]. Repeated ingredients have
            # already been combined, so all meal/module sources contribute once.
            known_totals = (grams / 100) @ self.nutrient_matrix
            missing = (grams.gt(0).to(self.dtype) @ self.missing_matrix).gt(0)
            actual = known_totals[..., self.nutrient_columns]
            scores = torch.clamp(actual / self.target_matrix, max=1) * self.score_weights
            deficits = torch.clamp(self.target_matrix - actual, min=0)
            person_scores = scores.sum(dim=-1)
            base_scores = person_scores.mean(dim=-1)
            ul_missing = missing[..., self.ul_columns]
            ul_excess = torch.clamp(
                known_totals[..., self.ul_columns] / self.ul_limits - 1, min=0
            )
            # With incomplete data, only a known excess above the tolerance counts.
            assessed = ~ul_missing | ul_excess.gt(EPS)
            ul_excess = torch.where(assessed, ul_excess, 0.0)
            total_excess = ul_excess.sum(dim=(-2, -1))
            fitness = base_scores - self.config.excess_penalty * total_excess / len(
                self.people
            )
            fully_checked = ~ul_missing.any(dim=(-2, -1))

            # Convert complete batches, avoiding a device synchronization per scalar.
            totals_cpu = known_totals.cpu().tolist()
            missing_cpu = missing.cpu().tolist()
            scores_cpu, deficits_cpu = scores.cpu().tolist(), deficits.cpu().tolist()
            person_scores_cpu = person_scores.cpu().tolist()
            ratios_cpu, assessed_cpu = ul_excess.cpu().tolist(), assessed.cpu().tolist()
            summary_cpu = (
                torch.stack((base_scores, fitness, total_excess), dim=-1).cpu().tolist()
            )
            checked_cpu = fully_checked.cpu().tolist()
        return (totals_cpu, missing_cpu, scores_cpu, deficits_cpu, person_scores_cpu,
                ratios_cpu, assessed_cpu, summary_cpu, checked_cpu)

    def _numpy_scores(self, grams):
        """Same batch arithmetic on CPU, including the separate missing-data mask."""
        known_totals = (grams / 100) @ self.nutrient_matrix
        missing = ((grams > 0).astype(self.dtype) @ self.missing_matrix) > 0
        actual = known_totals[..., self.nutrient_columns]
        scores = np.minimum(actual / self.target_matrix, 1) * self.score_weights
        deficits = np.maximum(self.target_matrix - actual, 0)
        person_scores = scores.sum(axis=-1)
        base_scores = person_scores.mean(axis=-1)
        ul_missing = missing[..., self.ul_columns]
        ul_excess = np.maximum(known_totals[..., self.ul_columns] / self.ul_limits - 1, 0)
        assessed = ~ul_missing | (ul_excess > EPS)
        ul_excess = np.where(assessed, ul_excess, 0.0)
        total_excess = ul_excess.sum(axis=(-2, -1))
        fitness = base_scores - self.config.excess_penalty * total_excess / len(self.people)
        fully_checked = ~ul_missing.any(axis=(-2, -1))
        arrays = (known_totals, missing, scores, deficits, person_scores, ul_excess,
                  assessed, np.stack((base_scores, fitness, total_excess), axis=-1),
                  fully_checked)
        return tuple(array.tolist() for array in arrays)

    @staticmethod
    def signature(individual: Individual) -> str:
        """Build a normalized JSON string for plan deduplication and population diversity statistics.

        Args:
            individual: Individual to generate a signature for; it is not modified.

        Returns:
            A signature that ignores within-meal combination order and ingredient
            ordering and rounds weights to two decimal places. Meal order, staple
            type, proportions, and each member's fruit weights still distinguish plans.
        """
        # Ingredient order is irrelevant; weights rounded only for diversity/deduplication.
        meals = []
        for meal in individual.meals:
            dishes = sorted(
                (tuple(sorted(zip(d.food_ids, d.units))), round(d.weight_g, 2))
                for d in meal.dishes
            )
            meals.append(
                (
                    meal.staple_id,
                    dishes,
                    sorted(zip(meal.fruit.food_ids, meal.fruit.units)),
                    [round(w, 2) for w in meal.fruit.person_weights_g],
                )
            )
        return json.dumps(meals, sort_keys=True)

    def run(self) -> tuple[list[tuple[Individual, Evaluation]], list[dict[str, Any]]]:
        """Run evolution through the separate evolution module.

        Returns:
            A (preferred plans, per-generation statistics) pair. Plans contain
            up to three distinct (Individual, Evaluation) pairs; statistics
            cover generation 0 through the configured generation count.

        Notes:
            This method is the shared execution entry point. Selection, crossover,
            mutation, and the generation loop are all performed by modules in
            genetic_operators, using and advancing this instance's rng state.
        """
        return evolution.evolve(self)

    def report(self, individual: Individual, evaluation: Evaluation) -> dict[str, Any]:
        """Organize a plan and its corresponding evaluation as display data for JSON serialization.

        Args:
            individual: Complete three-meal plan to display.
            evaluation: Latest evaluation of this individual; the caller must
                ensure that they correspond.

        Returns:
            A dictionary containing ingredients for three meals, household
            combination weights, each member's actual weights, nutrient totals,
            targets, deficits, UL checks, and scores. Displayed integer scores
            use rounding to the nearest integer with halves rounded up, while
            retaining unrounded original scores. This method does not rescore
            or modify the original evaluation.
        """
        meals = []
        for m, meal in enumerate(individual.meals):
            meals.append(
                {
                    "meal": MEALS[m],
                    "staple": {
                        "id": meal.staple_id,
                        "name": self.foods[meal.staple_id].name,
                        "person_weights_g": self.staple_weights(m, meal.staple_id),
                    },
                    "dishes": [
                        {
                            "total_weight_g": dish.weight_g,
                            "ingredients": [
                                {
                                    "id": fid,
                                    "name": self.foods[fid].name,
                                    "percent": u * 10,
                                    "household_weight_g": dish.weight_g * u / 10,
                                }
                                for fid, u in zip(dish.food_ids, dish.units)
                            ],
                        }
                        for dish in meal.dishes
                    ],
                    "fruit": {
                        "person_weights_g": meal.fruit.person_weights_g,
                        "ingredients": [
                            {"id": fid, "name": self.foods[fid].name, "percent": u * 10}
                            for fid, u in zip(meal.fruit.food_ids, meal.fruit.units)
                        ],
                    },
                }
            )
        people = copy.deepcopy(evaluation.people)
        for i, person in enumerate(people):
            person["score_rounded"] = math.floor(person["score"] + 0.5)
            person["scores_rounded"] = {
                k: math.floor(v + 0.5) for k, v in person["scores"].items()
            }
            person["portions"] = [
                {
                    "meal": MEALS[m],
                    "module": module,
                    "id": fid,
                    "name": self.foods[fid].name,
                    "weight_g": grams,
                }
                for m, module, fid, grams in self.portions(individual, i)
            ]
        return {
            "base_score": evaluation.base_score,
            "fitness": evaluation.fitness,
            "normalized_ul_excess": evaluation.excess,
            "ul_status": evaluation.ul_status,
            "meals": meals,
            "people": people,
        }


def main() -> None:
    """Run the command-line entry point: load inputs, run the genetic algorithm, and write JSON results.

    Arguments are read from the command line by argparse, supporting dataset,
    candidate pool, member file, population size, generations, seed, staple
    mutation probability, computing device, and output path. On completion,
    print the data source, PyTorch device, and summaries of preferred plans to
    standard output. Invalid inputs or configurations are reported by the
    argument parser or corresponding validation.
    """
    parser = argparse.ArgumentParser(
        description="成人家庭一日三餐遗传算法（默认使用CNF 2026数据集）"
    )
    parser.add_argument(
        "--foods", default="dataset", help="CNF数据目录，或自定义食材JSON"
    )
    parser.add_argument(
        "--candidates", help="CNF候选清单JSON，默认使用数据目录下的candidates.json"
    )
    parser.add_argument("--people", default="data/people.json")
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--staple-mutation", type=float, default=0.0)
    parser.add_argument(
        "--device", default="cpu", help="PyTorch设备：cpu、cuda、cuda:N或auto"
    )
    parser.add_argument("--backend", choices=("auto", "numpy", "torch"), default="auto")
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()
    try:
        foods, metadata = load_foods(args.foods, args.candidates)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    people = [
        Person(**p) for p in json.loads(Path(args.people).read_text(encoding="utf-8"))
    ]
    config = Config(
        population_size=args.population,
        generations=args.generations,
        staple_mutation_probability=args.staple_mutation,
        device=args.device,
    )
    try:
        ga = MealGA(foods, people, config, args.seed, backend=args.backend)
    except ValueError as error:
        parser.error(str(error))
    plans, history = ga.run()
    output = {
        "data_metadata": metadata,
        "config": asdict(config),
        "seed": args.seed,
        "backend": ga.backend_info(),
        "members": [asdict(p) for p in people],
        "history": history,
        "plans": [ga.report(ind, ev) for ind, ev in plans],
    }
    Path(args.output).write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    if metadata.get("demo_only"):
        print("注意：当前为未核对的演示食材数据，结果仅用于算法测试。")
    else:
        print(
            f"食材数据：{metadata.get('source', args.foods)}，候选食材{len(foods)}种。"
        )
    print(f"Scoring backend: {ga.backend_info()}")
    for i, (_, ev) in enumerate(plans, 1):
        print(
            f"方案{i}: 家庭平均分={ev.base_score:.2f}, 适应度={ev.fitness:.2f}, UL={ev.ul_status}"
        )
    print(f"已保存 {len(history)} 代统计及最多3个方案：{args.output}")


if __name__ == "__main__":
    main()
