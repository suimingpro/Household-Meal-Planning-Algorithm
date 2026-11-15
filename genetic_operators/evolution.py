"""进化流程：组织幸存者选择、父代选择、交叉、突变与逐代统计。"""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from ga_models import Evaluation, Individual
from genetic_operators import crossover as crossover_operator
from genetic_operators import mutation, parent_selection, survivor_selection

if TYPE_CHECKING:
    from meal_ga import MealGA


def evolve(ga: MealGA) -> tuple[list[tuple[Individual, Evaluation]], list[dict[str, Any]]]:
    """执行初始化、逐代评分、精英保留和非精英子代繁衍。

    Args:
        ga: 提供初始化、评分、合法性检查、配置及实例随机数生成器的运行器。

    Returns:
        (优选方案, 逐代统计)二元组。优选方案为历代最多3个签名不同的
        (Individual, Evaluation)，按UL优先比较规则排序；统计包含
        第0代至配置代数的最优分、平均分、超限比例和多样性。

    Notes:
        每代精英直接深拷贝；每对父代来自两轮独立四元锦标赛，并生成
        两个独立子代，直到下一代达到配置规模。末尾仅需一个时截断。
        本方法会推进实例的rng状态，再次调用不会自动重置随机种子。
    """
    population = [ga.initialize() for _ in range(ga.config.population_size)]
    history = []
    archive: dict[str, tuple[Individual, Evaluation]] = {}
    for generation in range(ga.config.generations + 1):
        evaluations = ga.evaluate_population(population)
        order = survivor_selection.rank_population(evaluations)
        seen = set()
        for i in order:
            signature = ga.signature(population[i])
            if signature in seen:
                continue
            seen.add(signature)
            archive[signature] = (copy.deepcopy(population[i]), evaluations[i])
            if len(seen) == 3:
                break
        # Keeping the best three distinct plans from all generations is sufficient.
        archive = dict(sorted(archive.items(), key=lambda item: item[1][1].key)[:3])
        best = evaluations[order[0]]
        history.append(
            {
                "generation": generation,
                "best_score": best.base_score,
                "best_fitness": best.fitness,
                "best_excess": best.excess,
                "best_ul_status": best.ul_status,
                "max_score": max(e.base_score for e in evaluations),
                "mean_score": sum(e.base_score for e in evaluations)
                / len(evaluations),
                "exceeded_fraction": sum(
                    e.ul_status == "exceeded" for e in evaluations
                )
                / len(evaluations),
                "unassessed_fraction": sum(
                    e.ul_status == "not_fully_assessed" for e in evaluations
                )
                / len(evaluations),
                "diversity": len({ga.signature(ind) for ind in population})
                / len(population),
            }
        )
        if generation == ga.config.generations:
            break
        next_population = survivor_selection.select_survivors(
            population, evaluations, ga.config.elite_count
        )
        while len(next_population) < ga.config.population_size:
            # Two independent four-way tournaments; overlap between draws is allowed.
            a, b = parent_selection.select_parents(
                population, evaluations, ga.rng
            )
            for _ in range(2):
                child = crossover_operator.crossover(ga, a, b)
                mutation.mutate_kind(ga, child)
                mutation.mutate_portions(ga, child)
                mutation.mutate_staple(ga, child)
                ga.validate(child)
                next_population.append(child)
                if len(next_population) == ga.config.population_size:
                    break
        population = next_population
    return sorted(archive.values(), key=lambda pair: pair[1].key), history
