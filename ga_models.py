"""Shared data models, nutrient constants, and runtime configuration for the three-meal planning algorithm.

This module uses standard-library dataclasses to represent foods, adult members,
three-meal encodings, and evaluation results, shared by data loading, genetic
operations, and PyTorch scoring, independently of the computing device.

Attributes:
    MEALS: Fixed breakfast, lunch, dinner order, corresponding to indices 0, 1, 2.
    CATEGORIES: The project's five food category identifiers; these do not restrict
        the nutrient fields actually accumulated.
    NUTRIENTS: The 11 nutrient indicators and their fixed order in the 100-point
        adequacy score. Field suffixes g, mg, and ug mean grams, milligrams,
        and micrograms, respectively.
    WEIGHTS: Weights of the scored nutrient indicators, totaling 100 points.
    UL: Daily tolerable upper intake limit parameters specified by the assignment;
        units are determined by field suffixes. The vitamin A and folate UL fields
        use preformed A and synthetic folic acid, respectively, and do not replace
        the adequacy fields.
    ALL_FIELDS: Ordered collection of energy, scoring indicators, and UL indicators,
        with duplicates removed.
    EPS: Floating-point tolerance for comparing weight bounds and excess levels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

MEALS = ("breakfast", "lunch", "dinner")
CATEGORIES = {"vegetable", "starch", "protein", "fruit", "other"}
NUTRIENTS = ("carbohydrate_g", "protein_g", "fat_g", "vitamin_a_rae_ug",
             "vitamin_c_mg", "vitamin_d_ug", "vitamin_e_mg", "folate_dfe_ug",
             "calcium_mg", "iron_mg", "potassium_mg")
WEIGHTS = dict(zip(NUTRIENTS, (20, 20, 20, 4, 4, 4, 4, 4, 20 / 3, 20 / 3, 20 / 3)))
# These are the assignment's model parameters, not age-specific clinical advice.
UL = {"preformed_a_ug": 3000, "vitamin_c_mg": 2000, "vitamin_d_ug": 100,
      "ul_applicable_e_mg": 1000, "synthetic_folic_acid_ug": 1000,
      "calcium_mg": 2500, "iron_mg": 45}
ALL_FIELDS = tuple(dict.fromkeys(("energy_kcal",) + NUTRIENTS + tuple(UL)))
EPS = 1e-7


@dataclass(frozen=True)
class Food:
    """A candidate food and its nutrient data per 100 g of edible food.

    Attributes:
        id: Unique food code; corresponds to Food_Code when CNF data are used.
        name: Display name, which should distinguish raw/cooked and processing states.
        category: One of the five project categories: vegetable, starch, protein,
            fruit, or other.
        role: staple for staple foods, fruit for fruit, mixed for foods that may
            be mixed, or standalone for separately served foods.
        per100g: Mapping from nutrient fields to amounts per 100 g. Energy is in
            kcal; other units are determined by field suffixes. Missing UL fields
            use None for unknown and must not be treated as 0.
        mix_group: Mixing compatibility group, defaulting to savory; foods from
            different compatibility groups are not mixed.
        max_percent: Upper limit on weight proportion in a side-dish combination,
            defaulting to 100, in increments of 10 percentage points. Values below
            100 require the mixed role and cannot be served alone.

    Notes:
        The loader and MealGA validate food roles, nutrient completeness, and
        compatibility groups. frozen only prevents attribute rebinding; it does
        not freeze data inside the per100g dictionary.
    """

    id: str
    name: str
    category: str
    role: str  # staple, fruit, mixed, standalone
    per100g: dict[str, float | None]
    mix_group: str = "savory"
    max_percent: int = 100

    def __post_init__(self):
        """Check that the proportion limit is an integer multiple of 10 in 10–100 and that the role allows mixing."""
        if type(self.max_percent) is not int or not 10 <= self.max_percent <= 100 or self.max_percent % 10:
            raise ValueError("max_percent must be an integer multiple of 10 in [10, 100]")
        if self.max_percent < 100 and self.role != "mixed":
            raise ValueError("Only mixed side ingredients can have max_percent below 100")


@dataclass(frozen=True)
class Person:
    """An adult member's reference group, body weight, and predefined daily energy requirement.

    Attributes:
        name: Member name; MealGA checks uniqueness within the household.
        age: Age, which must be an integer of at least 18.
        reference_group: male or female, used to select the model's nutrient targets.
        weight_kg: Body weight in kg; must be positive and finite.
        energy_kcal: Input reference energy requirement in kcal/day; must be
            positive and finite.

    Notes:
        This model does not predict energy requirements. Targets and ULs use
        project-specified parameters without further age subdivision. Age is
        used to validate adult status and record results.
    """

    name: str
    age: int
    reference_group: str  # male / female, used for reference targets
    weight_kg: float
    energy_kcal: float

    def __post_init__(self):
        """Check the reference group, adult age, body weight, and energy requirement after construction.

        Raises:
            ValueError: The group is not male/female, age does not meet the adult
                requirement, or weight or energy requirement is not positive and finite.
        """
        if self.reference_group not in {"male", "female"}:
            raise ValueError("reference_group must be male or female")
        if not isinstance(self.age, int) or self.age < 18:
            raise ValueError("This model accepts adults aged 18+ only")
        if any(not math.isfinite(v) or v <= 0 for v in (self.weight_kg, self.energy_kcal)):
            raise ValueError("Weight and reference energy must be positive and finite")

    @property
    def factor(self) -> float:
        """Return the dimensionless portion factor E/2000 relative to a 2000 kcal reference adult.

        This factor is used for side-dish budgets and individual portion allocation
        of shared household combinations.
        """
        return self.energy_kcal / 2000

    def targets(self) -> dict[str, float]:
        """Calculate this member's daily nutrient targets for adequacy scoring.

        Returns:
            A target dictionary constructed in NUTRIENTS order. Carbohydrate is
            E×0.55÷4 g, protein is 0.8×body weight in kg g, and fat is E×0.275÷9 g.
            Vitamins and minerals use project-specified fixed values or
            male/female group values.

        Notes:
            Vitamin A is in μg RAE and folate in μg DFE; these adequacy targets are not ULs.
        """
        male = self.reference_group == "male"
        return dict(zip(NUTRIENTS, (
            self.energy_kcal * .55 / 4, .8 * self.weight_kg,
            self.energy_kcal * .275 / 9, 900 if male else 700,
            90 if male else 75, 15, 15, 400, 1000,
            8 if male else 18, 3400 if male else 2600)))


@dataclass(frozen=True)
class Config:
    """Non-reassignable configuration for the genetic algorithm and three-meal portions.

    Attributes:
        population_size: Population size, default 100; at least 4 to support
            four-way tournaments.
        generations: Number of evolutionary updates after initial generation 0,
            default 50.
        elite_count: Number of elites directly retained by deep copy each
            generation, default 2; must be smaller than the population size.
        staple_fraction: Fraction of the daily carbohydrate target supplied by
            staples, default 0.8.
        meal_shares: Staple carbohydrate shares for breakfast, lunch, and dinner,
            default 0.25/0.4/0.35.
        side_budgets_g: Reference adult's side-dish budget for each meal,
            default 150/250/200 g.
        portion_mutation_probability: Independent portion-mutation probability
            for each non-elite child, default 0.3.
        crossover_probability: Probability of crossover rather than randomly
            copying a parent for each child, default 1.
        staple_mutation_probability: Independent probability of optional
            staple-type mutation, default 0.
        excess_penalty: Fitness penalty coefficient for per-person normalized
            UL excess, default 10.
        device: PyTorch device name, default cpu; also accepts cuda, cuda:N, or auto.

    Notes:
        The mutation module defines mutually exclusive ingredient-type mutation
        probabilities of 25%/5%/5%/5%. Device availability is resolved by
        device_utils; this data class does not check CUDA.
    """

    population_size: int = 100
    generations: int = 50
    elite_count: int = 2
    staple_fraction: float = .8
    meal_shares: tuple[float, ...] = (.25, .4, .35)
    side_budgets_g: tuple[float, ...] = (150, 250, 200)
    portion_mutation_probability: float = .3
    crossover_probability: float = 1.0
    staple_mutation_probability: float = 0.0
    excess_penalty: float = 10.0
    device: str = "cpu"

    def __post_init__(self):
        """Validate population, generations, elites, meal shares, probabilities, and other parameters after construction.

        Raises:
            ValueError: Population below 4, elite count or generations out of
                range, meal parameters not consisting of three positive finite
                values, shares not summing to 1, probabilities outside [0, 1],
                or a penalty coefficient that is not nonnegative and finite.
        """
        if self.population_size < 4 or not 0 <= self.elite_count < self.population_size:
            raise ValueError("Population >= 4 and 0 <= elite_count < population required")
        if self.generations < 0:
            raise ValueError("generations must be nonnegative")
        if (len(self.meal_shares) != 3 or len(self.side_budgets_g) != 3
                or any(not math.isfinite(x) or x <= 0 for x in self.meal_shares + self.side_budgets_g)
                or not math.isclose(sum(self.meal_shares), 1)):
            raise ValueError("Three positive meal shares summing to 1 and three budgets required")
        if any(not 0 <= p <= 1 for p in (self.staple_fraction,
                self.portion_mutation_probability, self.crossover_probability,
                self.staple_mutation_probability)):
            raise ValueError("Probabilities and staple fraction must be in [0, 1]")
        if not math.isfinite(self.excess_penalty) or self.excess_penalty < 0:
            raise ValueError("excess_penalty must be finite and nonnegative")


@dataclass
class Dish:
    """A shared household side-dish combination containing mixed ingredients or a standalone item.

    Attributes:
        food_ids: 1–4 distinct food codes; standalone items allow only one code.
        units: Integer proportion units corresponding one-to-one to the codes;
            each unit is 10%, each entry is at least 1, and the sum is 10.
            A single ingredient uses [10].
        weight_g: Total edible weight of this combination for the household in g,
            not the weight for each member.

    Notes:
        Individual portions are allocated in proportion to each E/2000 factor
        relative to the sum of household factors. This class does not automatically
        validate the encoding; MealGA.validate checks validity.
    """

    food_ids: list[str]
    units: list[int]  # tenths, always sum to 10
    weight_g: float


@dataclass
class Fruit:
    """Shared household fruit types, proportions, and per-person edible prepared weights for one meal.

    Attributes:
        food_ids: One or two distinct fruit codes.
        units: Proportion units of 10% for the corresponding ingredients, summing
            to 10. One fruit uses [10]; two fruits each use 2–8 units, or 20%–80%.
        person_weights_g: Total fruit weights for this meal in g, in member order.
            Each member has 50–150 g per meal and 250 g across three meals.

    Notes:
        All members use the same fruit proportions within a meal, but may have
        different total meal weights. The runner performs weight repair and
        cross-meal total validation.
    """

    food_ids: list[str]
    units: list[int]
    person_weights_g: list[float]


@dataclass
class Meal:
    """Side-dish combinations, staple code, and fruit configuration for one meal.

    Attributes:
        dishes: Shared household side-dish combinations; n each for breakfast
            and dinner, and n+1 for lunch.
        staple_id: Selected staple code; individual staple weights are derived
            immediately from nutrient targets.
        fruit: This meal's fruit configuration, including ingredient proportions
            and each member's fruit weight in grams.

    Notes:
        The meal is determined by its position in Individual.meals; no separate
        meal field is stored. Staple weights are not stored as independent genes.
    """

    dishes: list[Dish]
    staple_id: str
    fruit: Fruit


@dataclass
class Individual:
    """A complete household three-meal plan for one day in the genetic algorithm.

    Attributes:
        meals: A list of length 3, ordered as breakfast, lunch, and dinner.

    Notes:
        Nested lists and combinations are mutable. Crossover and elitism must
        use deep copies to prevent children from changing parents or multiple
        individuals from sharing the same encoding. Nutrient evaluations are
        stored separately in Evaluation.
    """

    meals: list[Meal]


@dataclass
class Evaluation:
    """An individual's per-person nutrient details, household fitness, and UL checks.

    Attributes:
        people: Nutrient totals, targets, scores, deficits, and UL-check
            dictionaries, in member order.
        base_score: Arithmetic mean of member base scores, up to 100 points,
            retaining floating-point precision.
        fitness: Base score minus the per-person normalized excess penalty;
            may be negative.
        excess: Sum of normalized excess ratios across all members and UL fields;
            when information is incomplete but the known intake lower bound
            exceeds the limit, that lower bound is used.
        ul_status: checked_no_excess means fully assessed with no excess;
            not_fully_assessed means no known excess but incomplete assessment;
            exceeded means known excess exists.

    Notes:
        Sorting does not use fitness alone; parent and survivor selection share
        the UL-priority rules in key.
    """

    people: list[dict[str, Any]]
    base_score: float
    fitness: float
    excess: float
    ul_status: str

    @property
    def key(self) -> tuple[float, ...]:
        """Return (UL status tier, excess level, negative fitness) for direct ascending comparison.

        Fully assessed with no excess precedes incomplete assessment, which
        precedes known excess. Within a tier, prefer a smaller total normalized
        excess, then higher fitness.

        Returns:
            A three-part comparison key; smaller values indicate better plans.

        Raises:
            KeyError: ul_status is not one of the three states supported by the model.
        """
        # Explicitly checked safe > incomplete UL assessment > known exceedance.
        tier = {"checked_no_excess": 0, "not_fully_assessed": 1, "exceeded": 2}[self.ul_status]
        return tier, self.excess, -self.fitness
