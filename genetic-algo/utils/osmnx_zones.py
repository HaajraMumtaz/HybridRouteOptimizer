import osmnx as ox
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import os
import sys



# ___0. load graph
city_name = "Lahore District, Punjab, Pakistan"
graph = ox.graph_from_place(city_name, network_type='drive')
# ── 1. Convert OSMnx graph to GeoDataFrames ──────────────────────────────────
nodes_gdf, edges_gdf = ox.graph_to_gdfs(graph, nodes=True, edges=True)

# ── 2. Load zone polygons ─────────────────────────────────────────────────────
file_path = os.path.join("genetic-algo", "data", "map.geojson")
print(file_path)
if not os.path.exists(file_path):
    print(f"❌ Error: File not found at {file_path}")
    sys.exit(1)  # Stops the script execution

try:
    zones = gpd.read_file(file_path)
    print("File loaded successfully.")
except Exception as e:
    print(f"❌ Error reading file: {e}")
    sys.exit(1)

# ── 3. Align CRS ──────────────────────────────────────────────────────────────
target_crs = edges_gdf.crs
nodes_gdf = nodes_gdf.to_crs(target_crs)
zones = zones.to_crs(target_crs)

# ── 4. Compute edge midpoints (vectorized via interpolate at 0.5 fraction) ────
midpoints = edges_gdf.geometry.interpolate(0.5, normalized=True)

# ── 5. Build midpoint GeoDataFrame (preserve MultiIndex u, v, key) ────────────
midpoints_gdf = gpd.GeoDataFrame(
    index=edges_gdf.index, geometry=midpoints, crs=target_crs
)

# ── 6. Spatial join: midpoints → zones ───────────────────────────────────────
joined = gpd.sjoin(
    midpoints_gdf,
    zones[["zoneId", "geometry"]],
    how="left",
    predicate="within",
)
# sjoin may produce duplicates if edges straddle boundaries; keep first match
joined = joined[~joined.index.duplicated(keep="first")]

# ── 7. Assign subzone_id (midpoint-based) back to edges ──────────────────────
edges_gdf["subzone_id"] = joined["zoneId"].reindex(edges_gdf.index)

# ── 8. Node-level zone assignment ─────────────────────────────────────────────
# Spatial join nodes → zones once, then map via index
nodes_joined = gpd.sjoin(
    nodes_gdf[["geometry"]],
    zones[["zoneId", "geometry"]],
    how="left",
    predicate="within",
)
nodes_joined = nodes_joined[~nodes_joined.index.duplicated(keep="first")]
node_zone: pd.Series = nodes_joined["zoneId"]  # indexed by osmid

# edges MultiIndex = (u, v, key); extract u/v node ids as arrays
u_ids = edges_gdf.index.get_level_values("u")
v_ids = edges_gdf.index.get_level_values("v")

node_zone_dict = node_zone.to_dict()
edges_gdf["subzone_u"] = u_ids.map(node_zone_dict)
edges_gdf["subzone_v"] = v_ids.map(node_zone_dict)

# ── 9. Boundary flag ──────────────────────────────────────────────────────────
# True when both endpoints are assigned AND belong to different zones
both_assigned = edges_gdf["subzone_u"].notna() & edges_gdf["subzone_v"].notna()
edges_gdf["is_boundary"] = both_assigned & (
    edges_gdf["subzone_u"] != edges_gdf["subzone_v"]
)

# ── 10. Handle missing assignments (already NaN from left-join; cast cleanly) ─
edges_gdf["subzone_id"] = edges_gdf["subzone_id"].where(
    edges_gdf["subzone_id"].notna(), other=pd.NA
)
edges_gdf["is_boundary"] = edges_gdf["is_boundary"].fillna(False)

# ── 11. Final output GeoDataFrame ─────────────────────────────────────────────
result = edges_gdf[["geometry", "subzone_id", "subzone_u", "subzone_v", "is_boundary"]]

result.to_file("genetic-algo/data/edges_with_zones.gpkg", layer="edges", driver="GPKG") 


print(result.head())
print(f"\nTotal edges : {len(result):,}")
print(f"Boundary edges: {result['is_boundary'].sum():,}")
print(f"Unassigned    : {result['subzone_id'].isna().sum():,}")