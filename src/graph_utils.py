import osmnx as ox
import matplotlib.pyplot as plt
import os
# 1. Fetch data
city_name = "Lahore District, Punjab, Pakistan"
G = ox.graph_from_place(city_name, network_type='drive')

data_dir = os.path.join("genetic-algo", "data")
file_path = os.path.join(data_dir, "lahore_graph_new.graphml")

nodes, edges = ox.graph_to_gdfs(G)
city_boundary = ox.geocode_to_gdf(city_name)

ox.save_graphml(G, filepath=file_path)

print(f"✅ Graph saved successfully to: {os.path.abspath(file_path)}")
print(f"📊 Stats: {len(G.nodes())} nodes and {len(G.edges())} edges.")

# 2. Create the figure 
def printhandshake():
    plt.figure(figsize=(12, 12))

    # 3. Plot the boundary first (This creates the initial axes)
    city_boundary.plot(color='red', alpha=0.2)

    # 4. Plot the edges on the same axes using plt.gca()
    edges.plot(ax=plt.gca(), linewidth=0.5, color='black', alpha=0.5)

    # 5. Finalize the plot
    plt.title("Verification: Road Network vs. Exported Boundary")
    plt.axis('off') # Optional: cleans up the coordinate numbers
    plt.show()

def getBoundary():
    graph = ox.graph_from_place(city_name, network_type='drive')


    # Convert to GeoDataFrames (nodes and edges)
    nodes, edges = ox.graph_to_gdfs(graph)
    city_boundary = ox.geocode_to_gdf(city_name)

    city_boundary.to_file("lahore_boundary.geojson", driver='GeoJSON')
