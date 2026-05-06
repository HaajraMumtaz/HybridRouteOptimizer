import re
import httpx   # pip install httpx

def parse_maps_link(url: str) -> tuple[float, float] | None:
    """
    Extract (lat, lon) from a Google Maps link.
    Handles:
      - https://maps.app.goo.gl/...  (short link → follow redirect)
      - https://www.google.com/maps/.../@lat,lon,...
      - https://www.google.com/maps/place/.../@lat,lon,...
    """
    # follow short links
    if "goo.gl" in url or "maps.app" in url:
        try:
            r = httpx.get(url, follow_redirects=True, timeout=10)
            url = str(r.url)
        except Exception:
            return None

    # @lat,lon pattern
    match = re.search(r'@(-?\d+\.\d+),(-?\d+\.\d+)', url)
    if match:
        return float(match.group(1)), float(match.group(2))

    # ?q=lat,lon pattern
    match = re.search(r'[?&]q=(-?\d+\.\d+),(-?\d+\.\d+)', url)
    if match:
        return float(match.group(1)), float(match.group(2))

    return None