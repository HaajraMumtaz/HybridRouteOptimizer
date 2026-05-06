import math
import networkx as nx
from collections import Counter

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────
ROAD_TYPE_MAP = {
    'motorway': 'highway',      'motorway_link': 'highway',
    'trunk': 'highway',         'trunk_link': 'highway',
    'primary': 'main_road',     'primary_link': 'main_road',
    'secondary': 'main_road',   'secondary_link': 'main_road',
    'tertiary': 'street',       'tertiary_link': 'street',
    'residential': 'street',    'living_street': 'street',
    'service': 'service_lane',
}

SPEED_DEFAULTS = {
    'highway': 100,
    'main_road': 60,
    'street': 30,
    'service_lane': 20,
}


# ─────────────────────────────────────────────────────────────────
# LAYER 0 — edge helpers (pure, no state)
# ─────────────────────────────────────────────────────────────────
def _road_type(hw) -> str:
    if isinstance(hw, list):
        hw = hw[0]
    return ROAD_TYPE_MAP.get(hw or '', 'street')


def _speed(data: dict) -> float:
    raw = data.get('maxspeed')
    if raw:
        try:
            return float(str(raw).split()[0])
        except (ValueError, AttributeError):
            pass
    rt = _road_type(data.get('highway', ''))
    return float(SPEED_DEFAULTS.get(rt, 30))


def _lanes(data: dict) -> float:
    try:
        return float(data.get('lanes', 1) or 1)
    except (ValueError, TypeError):
        return 1.0


def _base_time_minutes(length_m: float, speed_kmh: float) -> float:
    if speed_kmh <= 0:
        speed_kmh = 30.0
    return (length_m / 1000.0) / speed_kmh * 60.0


# ─────────────────────────────────────────────────────────────────
# LAYER 1 — precompute edge features (called ONCE at startup)
# ─────────────────────────────────────────────────────────────────
def precompute_edge_features(G) -> None:
    """
    Stamps G[u][v][k]['_f'] and G[u][v][k]['base_time'] on every edge.
    Call once after loading the graph. Mutates G in-place.
    """
    for u, v, k, data in G.edges(keys=True, data=True):
        length_m = float(data.get('length', 0) or 0)
        spd      = _speed(data)
        rt       = _road_type(data.get('highway', ''))

        # bearing for curvature — stored per-edge, aggregated at query time
        ux = G.nodes[u].get('x', 0); uy = G.nodes[u].get('y', 0)
        vx = G.nodes[v].get('x', 0); vy = G.nodes[v].get('y', 0)
        bearing = math.degrees(math.atan2(vx - ux, vy - uy)) % 360

        data['base_time'] = _base_time_minutes(length_m, spd)
        data['_f'] = {
            'distance_m':     length_m,
            'road_type':      rt,
            'speed_limit_kmh': spd,
            'num_lanes':      _lanes(data),
            'is_one_way':     '1' if data.get('oneway') else '0',
            'bearing':        bearing,   # used to compute curvature per path
        }


# ─────────────────────────────────────────────────────────────────
# LAYER 2 — routing (distance-only Dijkstra, single source → all)
# ─────────────────────────────────────────────────────────────────
def dijkstra_from(G, source_node: int) -> tuple[dict, dict]:
    """
    Returns (distances_in_minutes, predecessors) from source to ALL nodes.
    Uses precomputed base_time weight — no ML, no path storage overhead.
    """
    lengths, paths = nx.single_source_dijkstra(G, source_node, weight='base_time')
    return lengths, paths


# ─────────────────────────────────────────────────────────────────
# LAYER 3 — aggregate edge features along a path
# ─────────────────────────────────────────────────────────────────
def aggregate_path_features(G, path: list) -> dict:
    """
    Walk the path once, aggregate precomputed edge features.
    Returns a flat dict consumed by features.build_features().
    """
    total_dist   = 0.0
    total_time   = 0.0
    road_types   = []
    speeds       = []
    lanes_list   = []
    signal_count = 0
    one_way      = 0
    bearings     = []

    for u, v in zip(path[:-1], path[1:]):
        # pick lowest base_time parallel edge
        k    = min(G[u][v], key=lambda x: G[u][v][x].get('base_time', 9999))
        data = G[u][v][k]
        f    = data.get('_f', {})

        total_dist += f.get('distance_m', 0.0)
        total_time += data.get('base_time', 0.0)
        road_types.append(f.get('road_type', 'street'))
        speeds.append(f.get('speed_limit_kmh', 30.0))
        lanes_list.append(f.get('num_lanes', 1.0))
        bearings.append(f.get('bearing', 0.0))

        if f.get('is_one_way') == '1':
            one_way = 1
        if G.nodes[v].get('highway') == 'traffic_signals':
            signal_count += 1

    # curvature: mean absolute bearing change between consecutive edges
    curvature = 0.0
    if len(bearings) > 1:
        diffs = []
        for i in range(1, len(bearings)):
            d = abs(bearings[i] - bearings[i - 1])
            diffs.append(min(d, 360 - d))
        curvature = sum(diffs) / len(diffs)

    n = max(len(road_types), 1)
    dominant_rt = Counter(road_types).most_common(1)[0][0] if road_types else 'street'

    return {
        'base_time':       total_time,
        'distance_km':     max(total_dist / 1000.0, 0.01),
        'road_type':       dominant_rt,
        'has_signal':      '1' if signal_count > 0 else '0',
        'is_one_way':      str(one_way),
        'speed_limit_kmh': sum(speeds) / n,
        'num_lanes':       sum(lanes_list) / n,
        'road_curvature':  curvature,
    }