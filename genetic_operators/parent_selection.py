"""Independent tournament parent selection using the shared UL-first order."""
from __future__ import annotations

import random

from ga_models import Evaluation, Individual


def select_parent(population: list[Individual], evaluations: list[Evaluation],
                  rng: random.Random, tournament_size: int = 4) -> Individual:
    """Select a parent by tournament, drawing four distinct candidates by default.

    Args:
        population: Population to select from.
        evaluations: Latest evaluations with the same length and order as the population.
        rng: Instance random number generator performing sampling without replacement.
        tournament_size: Number of candidates per round, default 4; the population
            must contain at least this many individuals.

    Returns:
        A reference to the best parent among four distinct candidates under
        UL-priority rules, without a deep copy.
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
