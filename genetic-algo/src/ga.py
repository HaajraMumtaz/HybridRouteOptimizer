"""
GA with time-propagated cost + quadratic deadline penalties.
No routing, no ML inside the loop — just table lookups.
"""
import random
from datetime import datetime, timedelta
from features import TIME_SLOTS, ALL_SLOTS

PENALTY_COEFF = 0.01   # lateness² × coeff → penalty points


# ── slot resolution ───────────────────────────────────────────────────────────
def _hour_to_slot_idx(hour: int) -> int:
    for i, (_, (s, e)) in enumerate(TIME_SLOTS.items()):
        if s <= hour < e:
            return i
    return len(ALL_SLOTS) - 1


# ── time-propagated cost + penalties ─────────────────────────────────────────
def _route_cost_timed(
    individual: list,
    table: dict,
    start_time: datetime,
) -> float:
    """
    Walk route with running clock.
    At each stop, if a deadline exists and arrival is late,
    apply quadratic penalty: lateness_minutes² × PENALTY_COEFF.
    """
    total   = 0.0
    current = start_time

    for a, b in zip(individual[:-1], individual[1:]):
        slot_idx = _hour_to_slot_idx(current.hour)
        costs    = table.get((a.osm_node, b.osm_node))
        leg_cost = costs[slot_idx] if costs else 9999.0
        total   += leg_cost
        current += timedelta(minutes=leg_cost)

        # penalty at destination (b)
        if b.deadline is not None and current > b.deadline:
            lateness = (current - b.deadline).total_seconds() / 60.0
            total   += (lateness ** 2) * PENALTY_COEFF

    return total


# ── crossover — Ordered Crossover (OX) ───────────────────────────────────────
def _ox_crossover(p1: list, p2: list) -> list:
    n    = len(p1)
    i, j = sorted(random.sample(range(n), 2))
    segment   = p1[i:j+1]
    seg_set   = {s.osm_node for s in segment}
    remainder = [s for s in p2 if s.osm_node not in seg_set]
    return remainder[:i] + segment + remainder[i:]


# ── mutation — Inversion ──────────────────────────────────────────────────────
def _inversion_mutate(ind: list) -> list:
    ind = ind[:]
    if len(ind) < 2:
        return ind
    i, j = sorted(random.sample(range(len(ind)), 2))
    ind[i:j+1] = ind[i:j+1][::-1]
    return ind


# ── adaptive mutation rate ────────────────────────────────────────────────────
def _mutation_rate(generation: int, total: int) -> float:
    progress = generation / total
    if progress < 0.33:
        return 0.10
    if progress < 0.66:
        return 0.03
    return 0.005


# ── GA ────────────────────────────────────────────────────────────────────────
def run_ga(
    stops: list,
    table: dict,
    start_time: datetime,
    pop_size: int    = 15,
    generations: int = 50,
    elite_k: int     = 2,
) -> tuple[list, float]:
    if len(stops) <= 2:
        return stops, _route_cost_timed(stops, table, start_time)

    def fit(ind):
        return _route_cost_timed(ind, table, start_time)

    population = [random.sample(stops, len(stops)) for _ in range(pop_size)]
    best       = min(population, key=fit)
    best_val   = fit(best)

    for gen in range(generations):
        ranked   = sorted(population, key=fit)
        elites   = ranked[:elite_k]
        mut_rate = _mutation_rate(gen, generations)

        new_pop  = [e[:] for e in elites]

        while len(new_pop) < pop_size:
            p1, p2 = random.sample(elites + ranked[elite_k:elite_k+4], 2)
            child  = _ox_crossover(p1, p2)
            if random.random() < mut_rate:
                child = _inversion_mutate(child)
            new_pop.append(child)

        population = new_pop
        gen_val    = fit(ranked[0])
        if gen_val < best_val:
            best_val = gen_val
            best     = ranked[0]

    return best, best_val