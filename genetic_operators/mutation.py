"""Ingredient, portion and optional staple mutations for non-elite offspring."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ga_models import EPS, OLIVE_OIL_ID, Dish, Fruit, Individual, dish_unit_limit

if TYPE_CHECKING:
    from meal_ga import MealGA


def modules(individual: Individual) -> list[Dish | Fruit]:
    """返回可参与非主食或水果突变的模块引用列表。
    
    Args:
        individual: 要遍历的个体。
    
    Returns:
        按餐次依次排列的全部Dish与Fruit对象，主食不包含在内。
        列表中的对象仍属于原个体，修改它们会直接修改该个体。
    """
    return [module for meal in individual.meals for module in [*meal.dishes, meal.fruit]]


def mutate_staple(ga: MealGA, individual: Individual) -> None:
    """按配置概率原地执行一次可选的主食种类突变。
    
    Args:
        ga: 提供随机数、配置及兼容规则或修复方法的MealGA上下文。
        individual: 允许修改的非精英子代。
    
    Notes:
        触发时随机改变一餐的主食编号。主食克数是派生值，下次计算
        时随编号按公式重算，不产生独立重量基因。默认概率为0。
    """
    if ga.rng.random() < ga.config.staple_mutation_probability:
        meal = ga.rng.choice(individual.meals)
        meal.staple_id = ga.rng.choice([f for f in ga.staples if f != meal.staple_id])


def replacement_pool(ga: MealGA, module: Dish | Fruit, slot: int,
                     same_category: bool = False) -> list[str]:
    """筛选指定食材位置可用的替代编号，不修改组合。
    
    Args:
        ga: 提供随机数、配置及兼容规则或修复方法的MealGA上下文。
        module: 待替换食材所在的非主食组合或水果模块。
        slot: 模块内食材位置的索引。
        same_category: 是否额外要求替代品与原食材属于同一大类。
    
    Returns:
        保持角色及混合兼容性、不会引入重复编号的候选食材列表。
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
        ids = module.food_ids.copy()
        ids[slot] = fid
        if isinstance(module, Fruit) or (
            module.units[slot] <= dish_unit_limit(fid) and ga.compatible(ids)
        ):
            candidates.append(fid)
    return candidates


def mutate_kind(ga: MealGA, individual: Individual) -> str:
    """按互斥概率原地尝试一次种类突变。
    
    Args:
        ga: 提供随机数、配置及兼容规则或修复方法的MealGA上下文。
        individual: 允许修改的非精英子代。
    
    Returns:
        抽中的事件名：same_category、copy_other、delete、add或none。
        对应概率为25%、5%、5%、5%、60%；抽中但无法执行时保持原样，
        仍返回抽中的事件名。主食不参加这些事件。
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
        for module in all_modules:
            if len(module.food_ids) < 2:
                continue
            slots = list(range(len(module.food_ids)))
            if len(slots) == 2 and isinstance(module, Dish) and OLIVE_OIL_ID in module.food_ids:
                # Deleting the other ingredient would leave a 100% olive-oil dish.
                slots = [module.food_ids.index(OLIVE_OIL_ID)]
            options.append((module, slots))
        if options:
            module, slots = ga.rng.choice(options)
            slot = ga.rng.choice(slots)
            module.food_ids.pop(slot)
            removed = module.units.pop(slot)
            limits = [dish_unit_limit(fid) if isinstance(module, Dish) else 10
                      for fid in module.food_ids]
            for _ in range(removed):
                eligible = [i for i, u in enumerate(module.units) if u < limits[i]]
                module.units[ga.rng.choice(eligible)] += 1
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
    """独立于种类突变，按配置概率原地尝试一次分量突变。
    
    Args:
        ga: 提供随机数、配置及兼容规则或修复方法的MealGA上下文。
        individual: 允许修改的非精英子代。
    
    Returns:
        side_ratio、side_weight、fruit_ratio、fruit_transfer之一；
        未触发时返回none。默认触发概率30%，操作从当前可执行类别
        中随机选择，保持比例、克数边界及每人水果全天总量。
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
                        if (receiver != donor and
                                dish.units[receiver] < dish_unit_limit(dish.food_ids[receiver])):
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
