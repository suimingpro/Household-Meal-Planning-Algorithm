"""Select elite survivors that enter the next generation directly under UL-priority rules.

This module shares Evaluation.key comparison rules with parent selection:
fully assessed with no excess first, then no known excess but incomplete
assessment, and finally known excess. Within a tier, compare total normalized
excess, then fitness. Selection results are deep-copied without modifying the
original population or recalculating scores. The evolution module fills the
remaining next-generation positions with offspring produced by crossover and mutation.
"""
from __future__ import annotations

import copy

from ga_models import Evaluation, Individual


def rank_population(evaluations: list[Evaluation]) -> list[int]:
    """Return individual indices from best to worst by UL status, excess level, and fitness.

    Args:
        evaluations: Latest evaluations corresponding to population positions;
            may be empty.

    Returns:
        Indices in ascending Evaluation.key order; empty input returns an empty
        list. Equal comparison keys retain their original order. The input
        evaluation list is not modified.

    Notes:
        Uses the evaluation object's key property, not base score or fitness alone.
    """
    return sorted(range(len(evaluations)), key=lambda i: evaluations[i].key)


def select_survivors(population: list[Individual], evaluations: list[Evaluation],
                     count: int) -> list[Individual]:
    """Select and deep-copy the best count elites for direct retention in the next generation.

    Args:
        population: Current population containing individuals to select from.
        evaluations: Latest evaluations with the same length and one-to-one
            order as the population.
        count: Number of elites to retain; should be an integer from 0 to the
            population size.

    Returns:
        Elite copies ordered from best to worst; returns an empty list when
        count is 0. Each individual and its nested combinations, proportions,
        and fruit weights are deep-copied. Modifying returned results does not
        affect the original population or cause multiple copies to share the
        original individual's mutable encoding.

    Raises:
        ValueError: Population and evaluation list lengths differ, or count is
            outside the allowed range.

    Notes:
        This function only retains elites; it does not perform crossover,
        mutation, or nutrient scoring. The caller should ensure that evaluations
        correspond to current individuals and skip these elites during subsequent
        reproduction.
    """
    if len(population) != len(evaluations):
        raise ValueError("Population and evaluations must have equal lengths")
    if not 0 <= count <= len(population):
        raise ValueError("Survivor count must be between 0 and population size")
    return [copy.deepcopy(population[i]) for i in rank_population(evaluations)[:count]]
