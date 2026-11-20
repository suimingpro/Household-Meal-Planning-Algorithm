"""Whole-dish crossover with fruit-weight repair and independent child copies."""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from ga_models import Individual, Meal

if TYPE_CHECKING:
    from meal_ga import MealGA


def crossover(ga: MealGA, a: Individual, b: Individual) -> Individual:
    """Produce one independent child from two parents.

    Args:
        ga: MealGA context providing randomness, configuration, and compatibility
            rules or repair methods.
        a: Parent A; must use the same member and slot configuration as parent B.
        b: Parent B.

    Returns:
        A deep-copied child. When crossover is triggered, each corresponding
        combination, staple, and fruit module independently inherits from one
        parent with 50% probability, and fruit weights are repaired. Otherwise,
        one parent is randomly copied.
    """
    if ga.rng.random() >= ga.config.crossover_probability:
        return copy.deepcopy(ga.rng.choice([a, b]))
    meals = []
    for ma, mb in zip(a.meals, b.meals):
        dishes = [copy.deepcopy(ga.rng.choice([da, db])) for da, db in zip(ma.dishes, mb.dishes)]
        meals.append(Meal(dishes, ga.rng.choice([ma.staple_id, mb.staple_id]),
                          copy.deepcopy(ga.rng.choice([ma.fruit, mb.fruit]))))
    child = Individual(meals)
    ga.repair_fruit(child)
    return child
