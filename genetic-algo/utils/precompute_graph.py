import networkx as nx
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from collections import Counter

GRAPH_PATH  = "genetic-algo/data/lahore_graph.graphml"
ZONES_PATH  = "genetic-algo/data/lahore-zones.geojson"
OUTPUT_PATH = "genetic-algo/data/lahore_graph_zoned.graphml"

def main():
    # 1. load graph
    print("📥 Loading graph...")
    G = nx.read_graphml(GRAPH_PATH)
    G = nx.relabel_nodes(G, lambda x: int(x))
    for _, data in G.nodes(data=True):
        data['x'] = float(data.get('x', 0))
        data['y'] = float(data.get('y', 0))
    for _, _, _, data in G.edges(keys=True, data=True):
        if 'length' in data:
            data['length'] = float(data['length'])

    # 2. load + clean zones
    print("📥 Loading zones...")
    zones_gdf = gpd.read_file(ZONES_PATH)
    zones_gdf["zoneId"] = pd.to_numeric(zones_gdf["zoneId"], errors="coerce")
    zones_gdf = zones_gdf.dropna(subset=["zoneId"])
    zones_gdf = zones_gdf[~zones_gdf.geometry.is_empty]
    zones_gdf = zones_gdf[zones_gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
    zones_gdf["geometry"] = zones_gdf.geometry.buffer(0)
    zones_gdf = zones_gdf.drop_duplicates(subset=["zoneId"])
    zones_gdf["zoneId"] = zones_gdf["zoneId"].astype(int)
    zones_utm = zones_gdf.to_crs("EPSG:32643")
    print(f"✅ Zones loaded: {len(zones_gdf)}")

    # 3. build edge midpoints geodataframe
    print("🔄 Building edge midpoints...")
    rows = []
    for u, v, key, data in G.edges(keys=True, data=True):
        u_data, v_data = G.nodes[u], G.nodes[v]
        mid_lon = (float(u_data['x']) + float(v_data['x'])) / 2
        mid_lat = (float(u_data['y']) + float(v_data['y'])) / 2
        rows.append({"u": u, "v": v, "key": key, "geometry": Point(mid_lon, mid_lat)})

    edges_gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326").to_crs("EPSG:32643")

    # 4. primary spatial join
    print("🔗 Spatial join (intersection)...")
    joined = gpd.sjoin(edges_gdf, zones_utm[["zoneId", "geometry"]], how="left", predicate="intersects")

    # 5. nearest fallback
    missing = joined[joined["zoneId"].isna()]
    print(f"⚠️ Handling {len(missing)} missing edges via nearest neighbor...")
    if not missing.empty:
        nearest = gpd.sjoin_nearest(
            missing[["u", "v", "key", "geometry"]],
            zones_utm[["zoneId", "geometry"]],
            how="left"
        )
        joined.loc[missing.index, "zoneId"] = nearest["zoneId"].values

    # 6. write zone_id back to edges
    print("✍️ Writing zone_id to edges...")
    for _, row in joined.iterrows():
        z = row.get("zoneId")
        G[row["u"]][row["v"]][row["key"]]["zone_id"] = int(z) if pd.notna(z) else 15

    edge_assigned = sum(
        1 for _, _, d in G.edges(data=True) if d.get("zone_id", 15) != 15
    )
    print(f"✅ Edges assigned: {edge_assigned} / {G.number_of_edges()}")

    # 7. node zone = majority of incident edges, tie = first, none = 15
    print("🧭 Assigning node zones...")
    node_assigned = 0

    for node, ndata in G.nodes(data=True):
        edge_zones = []
        for _, _, edata in G.edges(node, data=True):
            z = edata.get('zone_id', 15)
            if z != 15:
                edge_zones.append(z)

        if edge_zones:
            ndata['zone_id'] = Counter(edge_zones).most_common(1)[0][0]  # tie = first
            node_assigned += 1
        else:
            ndata['zone_id'] = 15  # default

    print(f"✅ Nodes assigned: {node_assigned} / {G.number_of_nodes()}")
    print(f"  defaulted to zone 15: {G.number_of_nodes() - node_assigned}")

    # 8. save
    print(f"💾 Saving enriched graph to {OUTPUT_PATH}...")
    nx.write_graphml(G, OUTPUT_PATH)
    print("✅ Done.")

if __name__ == "__main__":
    main()