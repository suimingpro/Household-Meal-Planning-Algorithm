"""Independent tournament parent selection using the shared UL-first order."""
from __future__ import annotations

import random

from ga_models import Evaluation, Individual


def select_parent(population: list[Individual], evaluations: list[Evaluation],
                  rng: random.Random, tournament_size: int = 4) -> Individual:
    """通过锦标赛选择一个父代，默认抽取4个不同候选。
    
    Args:
        population: 待选择的种群。
        evaluations: 与种群等长且顺序对应的最新评分结果。
        rng: 执行无放回抽样的实例随机数生成器。
        tournament_size: 每轮候选数量，默认4；种群至少包含该数量个体。
    
    Returns:
        四个不同候选中按UL优先规则选出的最优父代引用，不作深拷贝。
    """
    if len(population) != len(evaluations):
        raise ValueError("Population and evaluations must have equal lengths")
    if not 1 <= tournament_size <= len(population):
        raise ValueError("Tournament size must be between 1 and population size")
    candidates = rng.sample(range(len(population)), tournament_size)
    return population[min(candidates, key=lambda i: evaluations[i].key)]


def select_parents(population: list[Individual], evaluations: list[Evaluation],
                   rng: random.Random, tournament_size: int = 4
                   ) -> tuple[Individual, Individual]:
    """Run two independent tournaments; draws and winning parents may overlap."""
    return (select_parent(population, evaluations, rng, tournament_size),
            select_parent(population, evaluations, rng, tournament_size))
