"""Ingredient, portion and optional staple mutations for non-elite offspring."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ga_models import EPS, Dish, Fruit, Individual

if TYPE_CHECKING:
    from meal_ga import MealGA


def modules(individual: Individual) -> list[Dish | Fruit]:
    """Return references to modules that can participate in side-dish or fruit mutation.

    Args:
        individual: Individual to traverse.

    Returns:
        All Dish and Fruit objects in meal order, excluding staples.
        Objects in the list still belong to the original individual; modifying
        them directly modifies that individual.
    """
    return [module for meal in individual.meals for module in [*meal.dishes, meal.fruit]]


def mutate_staple(ga: MealGA, individual: Individual) -> None:
    """Perform one optional staple-type mutation in place with the configured probability.

    Args:
        ga: MealGA context providing randomness, configuration, and compatibility
            rules or repair methods.
        individual: Non-elite child that may be modified.

    Notes:
        When triggered, randomly change one meal's staple code. Staple weights
        are derived values and are recalculated from the code using the formula
        at the next calculation; no independent weight gene is produced.
        The default probability is 0.
    """
    if ga.rng.random() < ga.config.staple_mutation_probability:
        meal = ga.rng.choice(individual.meals)
        meal.staple_id = ga.rng.choice([f for f in ga.staples if f != meal.staple_id])


def replacement_pool(ga: MealGA, module: Dish | Fruit, slot: int,
                     same_category: bool = False) -> list[str]:
    """Filter replacement codes for a specified ingredient slot without changing the combination.

    Args:
        ga: MealGA context providing randomness, configuration, and compatibility
            rules or repair methods.
        module: Side-dish combination or fruit module containing the ingredient
            to replace.
        slot: Index of the ingredient position within the module.
        same_category: Whether to additionally require replacements to belong
            to the same broad category as the original ingredient.

    Returns:
        Candidate foods that preserve role and mixing compatibility without
        introducing duplicate codes.
    """
    old = ga.foods[module.food_ids[slot]]
    pool = ga.fruits if isinstance(module, Fruit) else ga.sides
    candidates = []
    for fid in pool:
        food = ga.foods[fid]
        if fid in module.food_ids or (same_category and food.category != old.category):
            continue
        # Role is preserved even for a one-ingredient dish (milk stays standalone).
        if food.role != old.role:
            continue
        if module.units[slot] * 10 > food.max_percent:
            continue
        ids = module.food_ids.copy()
        ids[slot] = fid
        if isinstance(module, Fruit) or ga.compatible(ids):
            candidates.append(fid)
    return candidates


def mutate_kind(ga: MealGA, individual: Individual) -> str:
    """Attempt one ingredient-type mutation in place using mutually exclusive probabilities.

    Args:
        ga: MealGA context providing randomness, configuration, and compatibility
            rules or repair methods.
        individual: Non-elite child that may be modified.

    Returns:
        The selected event name: same_category, copy_other, delete, add, or none.
        Their probabilities are 25%, 5%, 5%, 5%, and 60%, respectively. If a
        selected event cannot be executed, leave the individual unchanged but
        still return that event's name. Staples do not participate in these events.
    """
    draw = ga.rng.random()
    event = "same_category" if draw < .25 else "copy_other" if draw < .30 else (
        "delete" if draw < .35 else "add" if draw < .40 else "none")
    all_modules = modules(individual)
    options = []
    if event in {"same_category", "copy_other"}:
        for module in all_modules:
            for slot in range(len(module.food_ids)):
                pool = replacement_pool(ga, module, slot, event == "same_category")
                if event == "copy_other":
                    source_ids = {fid for other in all_modules if other is not module
                                  and isinstance(other, Fruit) == isinstance(module, Fruit)
                                  for fid in other.food_ids}
                    pool = [f for f in pool if f in source_ids]
                if pool:
                    options.append((module, slot, pool))
        if options:
            module, slot, pool = ga.rng.choice(options)
            module.food_ids[slot] = ga.rng.choice(pool)
    elif event == "delete":
        options = []
        for module in all_modules:
            if len(module.food_ids) < 2:
                continue
            slots = [i for i in range(len(module.food_ids))
                     if sum(ga.foods[fid].max_percent for j, fid in enumerate(module.food_ids) if j != i) >= 100]
            if slots:
                options.append((module, slots))
        if options:
            module, slots = ga.rng.choice(options)
            slot = ga.rng.choice(slots)
            module.food_ids.pop(slot)
            removed = module.units.pop(slot)
            for _ in range(removed):
                recipients = [i for i, fid in enumerate(module.food_ids)
                              if module.units[i] * 10 < ga.foods[fid].max_percent]
                module.units[ga.rng.choice(recipients)] += 1
    elif event == "add":
        for module in all_modules:
            fruit = isinstance(module, Fruit)
            if len(module.food_ids) >= (2 if fruit else 4):
                continue
            pool = [f for f in (ga.fruits if fruit else ga.sides) if f not in module.food_ids
                    and (fruit or ga.compatible(module.food_ids + [f]))]
            transfer = 2 if fruit else 1
            donors = [i for i, u in enumerate(module.units) if u >= transfer + (2 if fruit else 1)]
            if pool and donors:
                options.append((module, pool, donors, transfer))
        if options:
            module, pool, donors, transfer = ga.rng.choice(options)
            module.units[ga.rng.choice(donors)] -= transfer
            module.food_ids.append(ga.rng.choice(pool))
            module.units.append(transfer)
    return event


def mutate_portions(ga: MealGA, individual: Individual) -> str:
    """Attempt one portion mutation in place with the configured probability, independently of ingredient-type mutation.

    Args:
        ga: MealGA context providing randomness, configuration, and compatibility
            rules or repair methods.
        individual: Non-elite child that may be modified.

    Returns:
        One of side_ratio, side_weight, fruit_ratio, or fruit_transfer;
        returns none when not triggered. The default trigger probability is 30%.
        The operation is randomly selected from currently executable categories,
        preserving proportion and weight bounds and each person's daily fruit total.
    """
    if ga.rng.random() >= ga.config.portion_mutation_probability:
        return "none"
    operations: dict[str, list[Any]] = {"side_ratio": [], "side_weight": [],
                                        "fruit_ratio": [], "fruit_transfer": []}
    for mi, meal in enumerate(individual.meals):
        for dish in meal.dishes:
            operations["side_weight"].append((mi, dish))
            for donor, units in enumerate(dish.units):
                if units >= 2:
                    for receiver in range(len(dish.units)):
                        if receiver != donor and dish.units[receiver] * 10 < ga.foods[dish.food_ids[receiver]].max_percent:
                            operations["side_ratio"].append((dish, donor, receiver))
        fruit = meal.fruit
        if len(fruit.food_ids) == 2:
            choices = [(d, step) for d in range(2) for step in (1, 2)
                       if fruit.units[d] - step >= 2 and fruit.units[1 - d] + step <= 8]
            if choices:
                operations["fruit_ratio"].append((fruit, choices))
    for donor in range(3):
        for receiver in range(3):
            if donor != receiver and all(
                    d >= 60 - EPS and r <= 140 + EPS for d, r in zip(
                        individual.meals[donor].fruit.person_weights_g,
                        individual.meals[receiver].fruit.person_weights_g)):
                operations["fruit_transfer"].append((donor, receiver))
    event = ga.rng.choice([k for k, v in operations.items() if v])
    choice = ga.rng.choice(operations[event])
    if event == "side_ratio":
        dish, donor, receiver = choice
        dish.units[donor] -= 1
        dish.units[receiver] += 1
    elif event == "side_weight":
        mi, dish = choice
        b = ga.baseline(mi)
        dish.weight_g = min(1.5 * b, max(.5 * b, dish.weight_g * ga.rng.choice([.9, 1.1])))
    elif event == "fruit_ratio":
        fruit, choices = choice
        # Choose 10/20 points equally when both are executable.
        step = ga.rng.choice(sorted({s for _, s in choices}))
        donor = ga.rng.choice([d for d, s in choices if s == step])
        fruit.units[donor] -= step
        fruit.units[1 - donor] += step
    else:
        donor, receiver = choice
        for p in range(len(ga.people)):
            individual.meals[donor].fruit.person_weights_g[p] -= 10
            individual.meals[receiver].fruit.person_weights_g[p] += 10
    return event
