# ─────────────────────────────────────────────
#  core/genetic.py  –  Genetic Algorithm
# ─────────────────────────────────────────────

import random
import copy
from core.data import DOMAINS, VARIABLES
from core.csp import is_valid
from core.fitness import evaluate_fitness


# ── GA Hyperparameters ─────────────────────────────────────────────────────────
POPULATION_SIZE  = 60
ELITE_SIZE       = 2      # Only carry the absolute best unchanged — keeps pressure low
MUTATION_RATE    = 0.25   # Higher mutation = more exploration
MAX_REPAIR_TRIES = 20     # Attempts to fix an invalid config before discarding
TOURNAMENT_SIZE  = 5      # Larger tournament = stronger selection but more variety
SHARING_SIGMA    = 3      # Fitness sharing radius (num components that differ)
SHARING_ALPHA    = 1.0    # Sharing function shape


# ── Similarity ─────────────────────────────────────────────────────────────────

def hamming_distance(c1: dict, c2: dict) -> int:
    """Count how many components differ between two configs."""
    return sum(1 for v in VARIABLES if c1[v] != c2[v])


def sharing_factor(config: dict, population: list[dict]) -> float:
    """
    Fitness sharing: penalize a config whose neighbours are too similar.
    The more clones/near-clones in the population, the more its
    effective fitness is divided down — forcing the GA to explore alternatives.
    """
    niche_count = 0.0
    for other in population:
        d = hamming_distance(config, other)
        if d < SHARING_SIGMA:
            niche_count += 1.0 - (d / SHARING_SIGMA) ** SHARING_ALPHA
    return max(niche_count, 1.0)


def shared_fitness(config: dict, population: list[dict], weights: dict) -> float:
    return evaluate_fitness(config, weights) / sharing_factor(config, population)


# ── Selection ──────────────────────────────────────────────────────────────────

def select(population: list[dict], weights: dict) -> dict:
    """
    Tournament selection using shared fitness to preserve diversity.
    """
    tournament = random.sample(population, min(TOURNAMENT_SIZE, len(population)))
    return max(tournament, key=lambda c: shared_fitness(c, tournament, weights))


# ── Crossover ─────────────────────────────────────────────────────────────────

def crossover(parent1: dict, parent2: dict) -> tuple[dict, dict]:
    """
    Uniform crossover: for each component, randomly inherit from either parent.
    Returns two children.
    """
    child1, child2 = {}, {}
    for var in VARIABLES:
        if random.random() < 0.5:
            child1[var], child2[var] = parent1[var], parent2[var]
        else:
            child1[var], child2[var] = parent2[var], parent1[var]
    return child1, child2


# ── Mutation ──────────────────────────────────────────────────────────────────

def mutate(config: dict, rate: float = MUTATION_RATE) -> dict:
    """
    Random resetting mutation: with `rate` probability,
    replace each component with a random value from its domain.
    Rate is adaptive — increases when the GA stagnates.
    """
    mutated = copy.deepcopy(config)
    for var in VARIABLES:
        if random.random() < rate:
            mutated[var] = random.choice(DOMAINS[var])
    return mutated


# ── Repair ────────────────────────────────────────────────────────────────────

def repair(config: dict) -> dict | None:
    """
    Try to fix a constraint-violating config by randomly re-sampling
    components that are likely causing violations.
    Returns a valid config or None if repair fails after MAX_REPAIR_TRIES.
    """
    fixed = copy.deepcopy(config)

    for _ in range(MAX_REPAIR_TRIES):
        if is_valid(fixed):
            return fixed

        # Targeted repairs based on known constraint patterns
        engine = fixed["engine"]

        # Rule 1: Electric must have None transmission
        if engine == "Electric" and fixed["transmission"] != "None":
            fixed["transmission"] = "None"

        # Rule 2: Electric must not have turbo
        if engine == "Electric" and fixed["turbo"] == "Yes":
            fixed["turbo"] = "No"

        # Rule 3: Non-electric must not have None transmission
        if engine != "Electric" and fixed["transmission"] == "None":
            fixed["transmission"] = random.choice(["Manual", "Automatic"])

        # Rule 4: V8 needs strong suspension
        if engine == "V8" and fixed["suspension"] == "Standard":
            fixed["suspension"] = random.choice(["Sport", "Heavy-Duty"])

        # Rule 5: Turbo needs strong suspension
        if fixed["turbo"] == "Yes" and fixed["suspension"] == "Standard":
            fixed["suspension"] = random.choice(["Sport", "Heavy-Duty"])

        # Rule 6: Servis tires cannot have Carbon rims
        if fixed["tires"] == "Servis" and fixed["rims"] == "Carbon":
            fixed["rims"] = random.choice(["Alloy", "Steel"])

        # Rule 7: Carbon rims need strong suspension
        if fixed["rims"] == "Carbon" and fixed["suspension"] == "Standard":
            fixed["suspension"] = random.choice(["Sport", "Heavy-Duty"])

    return fixed if is_valid(fixed) else None


# ── Main GA Loop ──────────────────────────────────────────────────────────────

def run_ga(
    initial_population: list[dict],
    weights: dict,
    generations: int = 30,
    progress_callback=None,
    stop_flag: callable = None   # Pass lambda: bool; returns True to abort
) -> tuple[list[dict],list[float],list[float]]:
    """
    Run the Genetic Algorithm.

    Returns:
        - final population sorted by fitness (best first)
        - history: list of best fitness per generation (for chart)
        - mutation_history: adaptive mutation rate per generation

    """
    population = initial_population[:POPULATION_SIZE]

    # Pad if initial population is small
    while len(population) < POPULATION_SIZE:
        population.append(random.choice(initial_population))

    history = []         # Best fitness per generation
    mutation_history = []  # Adaptive mutation rate per generation
    stagnation = 0
    current_mutation = MUTATION_RATE

    for gen in range(generations):
        # Bail out immediately if the window was closed
        if stop_flag and stop_flag():
            break
        # Evaluate & sort
        scored = sorted(
            population,
            key=lambda c: evaluate_fitness(c, weights),
            reverse=True
        )

        best_fitness = evaluate_fitness(scored[0], weights)
        history.append(best_fitness)

        # Adaptive mutation: ramp up if stagnating, cool down if improving
        if len(history) > 1:
            if best_fitness <= history[-2]:
                stagnation += 1
            else:
                stagnation = 0

        # After 3 stagnant generations, boost mutation to escape local optimum
        if stagnation >= 3:
            current_mutation = min(0.6, MUTATION_RATE + stagnation * 0.08)
        else:
            current_mutation = MUTATION_RATE

        if progress_callback:
            progress_callback(gen + 1, best_fitness)

        mutation_history.append(round(current_mutation, 3))

        # Elitism: carry top configs unchanged
        next_gen = scored[:ELITE_SIZE]

        # Fill rest of population through crossover + mutation
        while len(next_gen) < POPULATION_SIZE:
            p1 = select(scored, weights)
            p2 = select(scored, weights)
            child1, child2 = crossover(p1, p2)

            for child in [child1, child2]:
                child = mutate(child, current_mutation)
                if is_valid(child):
                    next_gen.append(child)
                else:
                    repaired = repair(child)
                    if repaired:
                        next_gen.append(repaired)

            if len(next_gen) >= POPULATION_SIZE:
                break

        population = next_gen[:POPULATION_SIZE]

    # Final sort by raw fitness
    population.sort(key=lambda c: evaluate_fitness(c, weights), reverse=True)
    return population, history, mutation_history


def pick_diverse_top(population: list[dict], weights: dict, n: int = 5, min_diff: int = 2) -> list[dict]:
    """
    Pick the top-n configs that are meaningfully different from each other.
    A config is only added if it differs by at least `min_diff` components
    from every already-selected config.
    Falls back to best available if not enough diverse configs exist.
    """
    ranked = sorted(population, key=lambda c: evaluate_fitness(c, weights), reverse=True)
    selected = []
    for candidate in ranked:
        if all(hamming_distance(candidate, s) >= min_diff for s in selected):
            selected.append(candidate)
        if len(selected) == n:
            break
    # Pad with next best if needed
    for candidate in ranked:
        if len(selected) == n:
            break
        if candidate not in selected:
            selected.append(candidate)
    return selected