"""按UL优先规则选择直接进入下一代的精英幸存者。

本模块与父代选择共用Evaluation.key比较规则：完整检查且无超限优先，
其次为无已知超限但检查不完整，最后为已知超限；同等级先比较归一化
超限总量，再比较适应度。选择结果深拷贝，不修改原种群或重新计算评分。
下一代的其他位置由进化模块通过交叉、突变产生的子代填充。
"""
from __future__ import annotations

import copy

from ga_models import Evaluation, Individual


def rank_population(evaluations: list[Evaluation]) -> list[int]:
    """按UL状态、超限程度和适应度返回从优到劣的个体索引。

    Args:
        evaluations: 与种群位置对应的最新评分结果列表，可为空。

    Returns:
        按Evaluation.key升序排列的索引列表，空输入返回空列表。
        比较键相同时保留原有顺序，不修改输入评分列表。

    Notes:
        使用评分对象的key属性，不按基础分或适应度单独排序。
    """
    return sorted(range(len(evaluations)), key=lambda i: evaluations[i].key)


def select_survivors(population: list[Individual], evaluations: list[Evaluation],
                     count: int) -> list[Individual]:
    """选取最优count个精英并深拷贝，供下一代直接保留。

    Args:
        population: 当前种群，包含待选择的个体。
        evaluations: 与种群等长且顺序一一对应的最新评分结果。
        count: 需要保留的精英数量，应为0到种群大小之间的整数。

    Returns:
        按优劣顺序排列的精英副本列表；count为0时返回空列表。
        每个个体及其嵌套组合、比例和水果重量均深拷贝，修改返回结果
        不会影响原种群，也不会让多个副本共享原个体的可变编码。

    Raises:
        ValueError: 种群与评分列表长度不同，或count超出允许范围。

    Notes:
        此函数只负责精英保留，不执行交叉、突变或营养评分；调用方应
        保证评分对应当前个体，并在后续繁衍中跳过这些精英。
    """
    if len(population) != len(evaluations):
        raise ValueError("Population and evaluations must have equal lengths")
    if not 0 <= count <= len(population):
        raise ValueError("Survivor count must be between 0 and population size")
    return [copy.deepcopy(population[i]) for i in rank_population(evaluations)[:count]]
