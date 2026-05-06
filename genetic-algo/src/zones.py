import networkx as nx
import geopandas as gpd
from shapely.geometry import Point

ZONED_GRAPH_PATH = "genetic-algo/data/lahore_graph_zoned.graphml"

_G_zoned: nx.MultiDiGraph | None = None

def load_zoned_graph():
    global _G_zoned
    print("📥 Loading zoned graph...")
    _G_zoned = nx.read_graphml(ZONED_GRAPH_PATH)
    _G_zoned = nx.relabel_nodes(_G_zoned, lambda x: int(x))
    for _, data in _G_zoned.nodes(data=True):
        data['x'] = float(data.get('x', 0))
        data['y'] = float(data.get('y', 0))
        data['zone_id'] = int(data.get('zone_id', 15))
    print(f"✅ Zoned graph loaded: {_G_zoned.number_of_nodes()} nodes")
    return _G_zoned


def snap_to_zone(lat: float, lon: float, default: int = 15) -> tuple[int, int]:
    """Snap (lat, lon) → nearest node → zone_id. Returns (osm_node, zone_id)."""
    if _G_zoned is None:
        raise RuntimeError("Call load_zoned_graph() first.")

    best_node, best_dist = None, float('inf')
    for node, data in _G_zoned.nodes(data=True):
        d = (data['x'] - lon) ** 2 + (data['y'] - lat) ** 2
        if d < best_dist:
            best_dist = d
            best_node = node

    zone = int(_G_zoned.nodes[best_node].get('zone_id', default))
    return best_node, zone


def debug_zone_plot(zones_geojson_path: str):
    """Load enriched graph + zones and plot node zone assignments."""
    import matplotlib.pyplot as plt
    import pandas as pd

    if _G_zoned is None:
        raise RuntimeError("Call load_zoned_graph() first.")

    zones_gdf = gpd.read_file(zones_geojson_path)
    zones_gdf["zoneId"] = pd.to_numeric(zones_gdf["zoneId"], errors="coerce")
    zones_gdf = zones_gdf.dropna(subset=["zoneId"])

    fig, ax = plt.subplots(figsize=(12, 12))
    zones_gdf.plot(ax=ax, alpha=0.3, edgecolor="black", column="zoneId", legend=True)

    xs, ys, colors = [], [], []
    for _, data in _G_zoned.nodes(data=True):
        xs.append(data['x'])
        ys.append(data['y'])
        colors.append(data['zone_id'])

    sc = ax.scatter(xs, ys, c=colors, s=1, cmap="tab20", alpha=0.5)
    plt.colorbar(sc, ax=ax, label="zone_id")
    plt.title("Node Zone Assignments (colored by zone)")
    plt.show()


def latlon_to_osm_node(G: nx.MultiDiGraph, lat: float, lon: float) -> int:
    """Snap (lat, lon) to nearest OSM node in graph."""
    best_node = None
    best_dist = float('inf')
    for node, data in G.nodes(data=True):
        nx_ = float(data.get('x', 0))   # lon
        ny  = float(data.get('y', 0))   # lat
        d   = (nx_ - lon) ** 2 + (ny - lat) ** 2
        if d < best_dist:
            best_dist = d
            best_node = node
    return int(best_node)
