def debug_zone_hit(G, zones_gdf):
    import random
    from shapely.geometry import Point

    print("\n🧪 ZONE HIT TEST\n")

    zones_gdf = zones_gdf.copy()

    # pick random nodes
    sample_nodes = list(G.nodes(data=True))[:20]

    hits = 0

    for n, d in sample_nodes:
        pt = Point(float(d["x"]), float(d["y"]))

        hit = zones_gdf[zones_gdf.geometry.contains(pt)]

        if not hit.empty:
            hits += 1
            print(f"✔ Node {n} → zone {hit.iloc[0].get('zoneId')}")
        else:
            print(f"❌ Node {n} → NO MATCH")

    print("\nHit rate:", hits, "/ 20")