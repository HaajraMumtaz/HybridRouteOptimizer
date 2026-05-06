import requests

def get_osrm_base_time(lat1, lon1, lat2, lon2) -> dict:
    """
    Uses OSRM public API to get real routed travel time and distance.
    No ML needed — this IS the ground truth base time.
    """
    url = (
        f"http://router.project-osrm.org/route/v1/driving/"
        f"{lon1},{lat1};{lon2},{lat2}"
        f"?overview=false&annotations=false"
    )
    r = requests.get(url, timeout=10)
    data = r.json()
    
    route = data['routes'][0]
    return {
        'base_time_minutes': route['duration'] / 60,
        'distance_km':       route['distance'] / 1000,
    }

# Example
result = get_osrm_base_time(31.5204, 74.3587, 31.4697, 74.2728)
print(result)
# → {'base_time_minutes': 18.4, 'distance_km': 11.2}