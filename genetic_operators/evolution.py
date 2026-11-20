"""Evolution workflow: organize survivor selection, parent selection, crossover, mutation, and per-generation statistics."""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from ga_models import Evaluation, Individual
from genetic_operators import crossover as crossover_operator
from genetic_operators import mutation, parent_selection, survivor_selection

if TYPE_CHECKING:
    from meal_ga import MealGA


def evolve(ga: MealGA) -> tuple[list[tuple[Individual, Evaluation]], list[dict[str, Any]]]:
    """Perform initialization, per-generation scoring, elitism, and non-elite offspring reproduction.

    Args:
        ga: Runner providing initialization, scoring, validity checks,
            configuration, and an instance random number generator.

    Returns:
        A (preferred plans, per-generation statistics) pair. Preferred plans are
        up to three (Individual, Evaluation) pairs with distinct signatures
        across generations, ordered by the UL-priority comparison rules.
        Statistics include best score, mean score, excess fraction, and diversity
        from generation 0 through the configured generation count.

    Notes:
        Elites are directly deep-copied each generation. Each parent pair comes
        from two independent four-way tournaments and produces two independent
        children until the next generation reaches its configured size; truncate
        if only one is needed at the end. This method advances the instance's
        rng state; calling it again does not automatically reset the random seed.
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
