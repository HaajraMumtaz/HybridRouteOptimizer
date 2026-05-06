"""
Precomputes the full cost table:
    table[(node_a, node_b)][slot_index] = adjusted_time_minutes

Call build_cost_table() once at startup. GA reads from it directly.
"""

import pandas as pd
from catboost import CatBoostRegressor, Pool

from routing  import dijkstra_from, aggregate_path_features
from features import ALL_FEATURES, CAT_FEATURES, build_features, slot_contexts


def build_cost_table(
    G,
    stops: list,          # list of Stop dataclass objects
    model: CatBoostRegressor,
    weather: str,
    now=None,
) -> dict:
    """
    Returns:
        table[(osm_a, osm_b)][slot_idx] = adjusted_minutes (float)
        base[(osm_a, osm_b)]            = base_time in minutes (float)
        route_feats[(osm_a, osm_b)]     = aggregate_path_features dict
    """
    nodes     = [s.osm_node for s in stops]
    slot_ctxs = slot_contexts(now)           # 6 dicts, one per slot
    n_slots   = len(slot_ctxs)

    # ── Step 1: Dijkstra from every stop node (distance-only) ──────────────
    print("  Running Dijkstra from each stop...")
    dijk: dict[int, tuple[dict, dict]] = {}
    for stop in stops:
        lengths, paths = dijkstra_from(G, stop.osm_node)
        dijk[stop.osm_node] = (lengths, paths)

    # ── Step 2: Aggregate path features for every pair ─────────────────────
    print("  Aggregating path features...")
    route_feats: dict[tuple, dict]  = {}
    base:        dict[tuple, float] = {}

    for a in stops:
        _, paths_from_a = dijk[a.osm_node]
        for b in stops:
            if a.osm_node == b.osm_node:
                continue
            pair = (a.osm_node, b.osm_node)
            if b.osm_node not in paths_from_a:
                route_feats[pair] = None
                base[pair]        = 9999.0
                continue
            path = paths_from_a[b.osm_node]
            rf   = aggregate_path_features(G, path)
            route_feats[pair] = rf
            base[pair]        = rf['base_time']

    # ── Step 3: Batch ML prediction for all pairs × all slots ─────────────
    print("  Batch ML prediction...")
    records    = []
    record_keys = []   # (pair, slot_idx) in same order as records

    for a in stops:
        for b in stops:
            if a.osm_node == b.osm_node:
                continue
            pair = (a.osm_node, b.osm_node)
            rf   = route_feats[pair]
            if rf is None:
                continue
            for si, tctx in enumerate(slot_ctxs):
                ctx = {
                    **tctx,
                    'origin_zone':str(a.zone_id),   # already set from Stop
                    'dest_zone':str(b.zone_id),   # already set from Stop
                    'weather_condition': weather,
                    'is_construction': '0',
                }
                records.append(build_features(rf, ctx))
                record_keys.append((pair, si))

    df     = pd.DataFrame(records)[ALL_FEATURES]
    pool   = Pool(df, cat_features=CAT_FEATURES)
    preds  = model.predict(pool)   # single batch call

    # ── Step 4: Assemble cost table ────────────────────────────────────────
    table: dict[tuple, list] = {}

    # init all pairs with fallback
    for a in stops:
        for b in stops:
            if a.osm_node == b.osm_node:
                continue
            pair = (a.osm_node, b.osm_node)
            table[pair] = [9999.0] * n_slots

    for (pair, si), mult in zip(record_keys, preds):
        bt           = base[pair]
        mult         = max(float(mult), 0.5)   # safety floor
        table[pair][si] = bt * mult

    print(f"  Done. {len(record_keys)} predictions ({len(stops)**2} pairs × {n_slots} slots).")
    return table, base, route_feats