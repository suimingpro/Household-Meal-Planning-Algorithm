"""Whole-dish crossover with fruit-weight repair and independent child copies."""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from ga_models import Individual, Meal

if TYPE_CHECKING:
    from meal_ga import MealGA


def crossover(ga: MealGA, a: Individual, b: Individual) -> Individual:
    """从两个父代产生一个独立子代。
    
    Args:
        ga: 提供随机数、配置及兼容规则或修复方法的MealGA上下文。
        a: 父代A，需与父代B采用相同的成员与槽位配置。
        b: 父代B。
    
    Returns:
        深拷贝的子代。触发交叉时各对应组合、主食及水果模块独立
        以50%概率继承父代之一，并修复水果重量；否则随机复制一方。
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
