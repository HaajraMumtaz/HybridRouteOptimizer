import requests

def extract_coords_from_url(url):
    """
    Resolves shortened Google Maps URLs and extracts Lat/Long.
    """
    try:
        response = requests.get(url, allow_redirects=True, timeout=5)
        final_url = response.url
        
        #2.Extract from URL (usually in the @lat,long format or q=lat,long)
        # Example: .../@31.4826,74.3052,17z...
        if "@" in final_url:
            parts = final_url.split("@")[1].split(",")
            return float(parts[0]), float(parts[1])
        
        # Fallback for q= format
        elif "q=" in final_url:
            coords = final_url.split("q=")[1].split("&")[0].split(",")
            return float(coords[0]), float(coords[1])
            
    except Exception as e:
        print(f"Error extracting coordinates: {e}")
        return None