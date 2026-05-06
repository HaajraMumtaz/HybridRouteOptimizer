import osmnx as ox
import geopandas as gpd
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# -----------------------------
# 1. LOAD DATA
# -----------------------------
# -----------------------------
# 1. LOAD DATA (ROBUST VERSION)
# -----------------------------
print("📥 Loading graph...")
graph_path = os.path.join("genetic-algo", "data", "lahore_graph.graphml")
G = ox.load_graphml(graph_path)

print("📥 Loading zones...")
zones_path = os.path.join("genetic-algo", "data", "lahore-zones.geojson")
zones_gdf = gpd.read_file(zones_path)

# --- FIX START ---
# Check if zoneId column exists
if 'zoneId' not in zones_gdf.columns:
    print("❌ Error: 'zoneId' column not found in GeoJSON!")
    # Optional: use the index as zoneId if the column is missing
    zones_gdf['zoneId'] = zones_gdf.index

# Handle NaN values: Fill missing IDs with 0 or drop them
# Option A: Fill with 0
# Old line causing error
# zones_gdf['zoneId'] = zones_gdf['zoneId'].fillna(0).astype(int)

# New, robust solution
zones_gdf['zoneId'] = pd.to_numeric(zones_gdf['zoneId'], errors='coerce').fillna(0).astype(int)


# Option B: Or if you want to remove zones with no ID:
# zones_gdf = zones_gdf.dropna(subset=['zoneId'])
# zones_gdf['zoneId'] = zones_gdf['zoneId'].astype(int)
# --- FIX END ---

zones_gdf = zones_gdf.reset_index(drop=True)
zones_gdf = zones_gdf[['zoneId', 'geometry']]
# -----------------------------
# 2. PROJECT TO METRIC (UTM 43N)
# -----------------------------
print("🔄 Projecting to UTM Zone 43N (Metric)...")
metric_crs = "EPSG:32643" 
zones_gdf = zones_gdf.to_crs(metric_crs)


zones_gdf['geometry'] = zones_gdf.geometry.buffer(0)
print("🔄 Converting graph to edges...")
edges = ox.graph_to_gdfs(G, nodes=False).reset_index()
edges = edges.to_crs(metric_crs)

# -----------------------------
# 3. SPATIAL JOIN (INTERSECTION)
# -----------------------------
print("🔗 Performing primary spatial join (intersection)...")
# We use 'inner' or 'left' but specifically handle the duplicates
joined = gpd.sjoin(edges, zones_gdf, how="left", predicate="intersects")

# If an edge spans two zones, pick the first one to keep 1 row per edge
joined = (
    joined
    .sort_values(by="zoneId", na_position="last")  # keep valid zoneIds first
    .drop_duplicates(subset=['u', 'v', 'key'], keep='first')
)

# -----------------------------
# 4. NEAREST JOIN (THE FIX)
# -----------------------------
missing_mask = joined["zoneId"].isna()

if missing_mask.any():
    num_missing = missing_mask.sum()
    print(f"⚠️ Handling {num_missing} missing edges via nearest neighbor...")
    
    # Isolate missing edges
    missing_edges = joined[missing_mask].drop(columns=['index_right', 'zoneId'])
    
    # Find nearest zone within 2km (Increased distance for safety)
    nearest = gpd.sjoin_nearest(
        missing_edges,
        zones_gdf,
        how="left",
        max_distance=2000
    )
    
    # IMPORTANT: Deduplicate the nearest results. 
    # If one edge is near two zones, sjoin_nearest returns TWO rows.
    # This is what caused your ValueError.
    nearest = nearest.drop_duplicates(subset=['u', 'v', 'key'])
    
    # Create a mapping dictionary { (u, v, key): zoneId }
    nearest_map = nearest.set_index(['u', 'v', 'key'])['zoneId']
    
    # Update the main 'joined' dataframe using the map
    joined.set_index(['u', 'v', 'key'], inplace=True)
    joined['zoneId'] = joined['zoneId'].fillna(nearest_map)
    joined.reset_index(inplace=True)

# Final cleanup: fill remaining (far out) roads with 0
joined["zoneId"] = joined["zoneId"].fillna(0).astype(int)


# -----------------------------
# 5. WRITE BACK TO GRAPH
# -----------------------------
print("✍️ Assigning zone IDs back to graph edges...")

# Create a mapping dictionary { (u, v, key): zoneId }
zone_dict = joined.set_index(['u', 'v', 'key'])['zoneId'].to_dict()

# We use G.edges(keys=True, data=True) to ensure we get exactly 4 values:
# u (source), v (target), k (key), and d (dictionary of attributes)
for u, v, k, d in G.edges(keys=True, data=True):
    # Get the zone_id from our dictionary using the unique (u, v, k) tuple
    # If not found, default to 0
    d['zone_id'] = zone_dict.get((u, v, k), 0)

print("✅ Graph update complete.")
# -----------------------------
# 6. SUMMARY & VISUALIZATION
# -----------------------------
total_edges = len(edges)
assigned = (joined["zoneId"] != 0).sum()
unassigned = joined[joined["zoneId"] == 0]

print(f"\n📊 --- SUMMARY ---")
print(f"✅ Total edges: {total_edges}")
print(f"✅ Assigned:    {assigned}")
print(f"❌ Unassigned:  {len(unassigned)}")

print("\n📊 Generating debug plot...")
fig, ax = plt.subplots(figsize=(12, 12))

# Plot Zones
zones_gdf.plot(ax=ax, color='red', alpha=0.2, edgecolor='red', label='Zones')

# Plot Assigned Edges
assigned_gdf = joined[joined["zoneId"] != 0]
assigned_gdf.plot(ax=ax, color='blue', linewidth=0.5, alpha=0.5)

# Plot Unassigned (Highlight in Black)
if len(unassigned) > 0:
    unassigned.plot(ax=ax, color='black', linewidth=1.5, label='Unassigned')

legend_elements = [
    Patch(facecolor='red', alpha=0.3, label='Zone Boundaries'),
    Patch(facecolor='blue', alpha=0.5, label='Successfully Assigned'),
    Patch(facecolor='black', label='Still Unassigned')
]
ax.legend(handles=legend_elements)
plt.title(f"Lahore Zone Assignment (Unassigned: {len(unassigned)})")
plt.show()