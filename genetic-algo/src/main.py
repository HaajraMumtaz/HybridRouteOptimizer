import random
import networkx as nx
from datetime import datetime
from dataclasses import dataclass
from catboost import CatBoostRegressor

from routing     import precompute_edge_features
from precompute  import build_cost_table
from features    import ALL_SLOTS, TIME_SLOTS
from ga          import run_ga
from zones       import load_zoned_graph, snap_to_zone, debug_zone_plot
from link_parser import parse_maps_link

# ── config ───────────────────────────────────────────────────────────────────
ZONES_PATH      = "genetic-algo/data/lahore-zones.geojson"
MODEL_PATH      = "models/congestion_multiplier_v1.cbm"
WEATHER_OPTIONS = ["clear", "rain", "fog", "haze"]

@dataclass
class Stop:
    name:     str
    osm_node: int
    zone_id:  int = 15
    deadline: datetime | None = None


# ── helpers ───────────────────────────────────────────────────────────────────
def simulate_weather() -> str:
    return random.choices(WEATHER_OPTIONS, weights=[0.6, 0.2, 0.1, 0.1])[0]


def current_slot_index(now: datetime) -> int:
    for i, (_, (s, e)) in enumerate(TIME_SLOTS.items()):
        if s <= now.hour < e:
            return i
    return 5  # night fallback


def get_stops_from_user() -> list[Stop]:
    print("\nEnter stops via Google Maps links (max 20). Empty name to finish.\n")
    stops = []
    now = datetime.now()

    while len(stops) < 20:
        name = input(f"  [{len(stops)+1}] Stop name (Enter to finish): ").strip()
        if not name:
            break

        url = input(f"       Google Maps link for '{name}': ").strip()
        if not url:
            break

        coords = parse_maps_link(url)
        if coords is None:
            print("       ⚠ Could not extract coordinates. Try a full link.\n")
            continue

        lat, lon          = coords
        osm_node, zone_id = snap_to_zone(lat, lon)  # single call, no get_zone needed

        dl_str = input(f"       Deadline for '{name}' (HH:MM or Enter for none): ").strip()
        deadline = None
        if dl_str:
            try:
                t = datetime.strptime(dl_str, "%H:%M")
                deadline = now.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
            except ValueError:
                print("       ⚠ Invalid format, no deadline set.")

        stop = Stop(name=name, osm_node=osm_node, zone_id=zone_id, deadline=deadline)
        stops.append(stop)
        print(f"       ✓ ({lat:.5f}, {lon:.5f}) → node {osm_node} → zone {zone_id}\n")

    return stops


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    now = datetime.now()

    # 1. load enriched zoned graph (no spatial ops, just graphml)
    G = load_zoned_graph()

    # 2. stamp edge features once
    print("Precomputing edge features...")
    precompute_edge_features(G)

    # 3. debug plot — confirm zone assignments visually
    debug_zone_plot(ZONES_PATH)

    # 4. load model
    print("Loading model...")
    model = CatBoostRegressor()
    model.load_model(MODEL_PATH)

    # 5. collect stops
    stops = get_stops_from_user()
    if len(stops) < 2:
        print("Need at least 2 stops. Exiting.")
        return

    print(f"\n{len(stops)} stops loaded:")
    for s in stops:
        print(f"  {s.name:15s} node={s.osm_node}  zone={s.zone_id}")

    # 6. precompute cost table — all pairs × 6 slots, one batch predict
    weather = simulate_weather()
    print(f"\nWeather simulation: {weather}")
    print("Building cost table...")
    table, base, route_feats = build_cost_table(G, stops, model, weather, now)

    # 7. GA for every time slot
    print("\n── GA Results per Time Slot ─────────────────────────────────────")
    slot_results = {}
    for si, slot_name in enumerate(ALL_SLOTS):
        slot_hour  = list(TIME_SLOTS.values())[si][0]
        slot_start = now.replace(hour=slot_hour, minute=0, second=0, microsecond=0)

        best_route, best_cost = run_ga(
            stops       = stops,
            table       = table,
            start_time  = slot_start,
            pop_size    = 15,
            generations = 50,
            elite_k     = 4,
        )
        slot_results[si] = (best_route, best_cost)
        route_str = " → ".join(s.name for s in best_route)
        print(f"  [{slot_name:15s}]  {route_str}   ({best_cost:.1f} min)")

    # 8. detailed breakdown for current slot
    si_now      = current_slot_index(now)
    best_now, _ = slot_results[si_now]

    print(f"\n── Leg Detail (current: {ALL_SLOTS[si_now]}) ────────────────────")
    for a, b in zip(best_now[:-1], best_now[1:]):
        pair    = (a.osm_node, b.osm_node)
        bt      = base.get(pair, 9999.0)
        adj     = table.get(pair, [9999.0] * 6)[si_now]
        rf      = route_feats.get(pair) or {}
        dist_km = rf.get('distance_km', 0.0)
        mult    = (adj / bt) if bt > 0 else 1.0
        print(f"  {a.name:15s} → {b.name:15s}  "
              f"{dist_km:.2f} km  |  base {bt:.1f} min  |  "
              f"×{mult:.2f}  |  adjusted {adj:.1f} min")


if __name__ == "__main__":
    main()