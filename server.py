from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from catboost import CatBoostRegressor
import logging
import time
import uuid
from fastapi import Request
from genetic_algo.src.link_parser import parse_maps_link
from genetic_algo.src.main import Stop, simulate_weather
from genetic_algo.src.routing import precompute_edge_features
from genetic_algo.src.precompute import build_cost_table
from genetic_algo.src.ga import run_ga_top3
from genetic_algo.src.zones import load_zoned_graph, snap_to_zone

from maps_api.main import get_top_best
from pydantic import BaseModel, Field
# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="AXIS LHR — Route Optimizer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("axis-lhr")
# ── Lahore bounding box ───────────────────────────────────────────────────────
LHR_BOUNDS = {
    "min_lat": 31.20059, "max_lat": 31.71342,
    "min_lon": 74.00685, "max_lon": 74.65467,
}
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.time()

    logger.info(f"[{request_id}] START {request.method} {request.url}")

    try:
        response = await call_next(request)
        duration = time.time() - start_time

        logger.info(
            f"[{request_id}] END {request.method} {request.url} "
            f"status={response.status_code} duration={duration:.3f}s"
        )

        return response

    except Exception as e:
        duration = time.time() - start_time
        logger.exception(
            f"[{request_id}] ERROR {request.method} {request.url} "
            f"duration={duration:.3f}s error={str(e)}"
        )
        raise

# ── startup: load graph + model once ─────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global G, model

    logger.info("Loading graph...")
    G = load_zoned_graph()
    precompute_edge_features(G)
    logger.info("Graph loaded successfully.")

    logger.info("Loading CatBoost model...")
    model = CatBoostRegressor()
    model.load_model("models/congestion_multiplier_v1.cbm")
    logger.info("Model loaded successfully.")

    logger.info("Server ready.")


# ── schemas ───────────────────────────────────────────────────────────────────
class ResolveRequest(BaseModel):
    query: str                        # raw text or Google Maps URL


class ResolvedStop(BaseModel):
    name:    str
    lat:     float
    lon:     float
    zone_id: int
    valid:   bool                     # False = outside Lahore / unresolvable
    reason:  Optional[str] = None     # human-readable failure reason


class StopIn(BaseModel):
    name:     str
    lat:      float
    lon:      float
    zone_id:  int
    deadline: Optional[str] = None    # "HH:MM"


class OptimizeRequest(BaseModel):
    stops: list[StopIn]


class StopOut(BaseModel):
    name:     str
    lat:      float
    lon:      float
    zone_id:  int
    deadline: Optional[str] = None


class Top3Item(BaseModel):
    route:   list[str]
    ga_cost: float

class LegOut(BaseModel):
    from_:    str = Field(alias="from")
    to:       str
    distance: str
    duration: str
    steps:    list[str]

    class Config:
        populate_by_name = True
class OptimizeResponse(BaseModel):
    weather:     str
    google_mins: float
    final_score: float
    route:       list[StopOut]
    top3:        list[Top3Item]
    legs:        list[LegOut]   


# ── helpers ───────────────────────────────────────────────────────────────────
def _in_lahore(lat: float, lon: float) -> bool:
    return (
        LHR_BOUNDS["min_lat"] <= lat <= LHR_BOUNDS["max_lat"]
        and LHR_BOUNDS["min_lon"] <= lon <= LHR_BOUNDS["max_lon"]
    )


def _geocode_name(name: str) -> tuple[float, float] | None:
    """
    Google Geocoding API first; Nominatim OSM fallback for dev.
    """
    import os
    import requests as req

    key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if key:
        try:
            r    = req.get(
                "https://maps.googleapis.com/maps/api/geocode/json",
                params={"address": f"{name}, Lahore, Pakistan", "key": key},
                timeout=6,
            )
            data = r.json()
            if data.get("status") == "OK":
                loc = data["results"][0]["geometry"]["location"]
                return loc["lat"], loc["lng"]
        except Exception:
            pass

    # Nominatim fallback (rate-limited 1 req/s — fine for dev/testing)
    try:
        r       = req.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": f"{name}, Lahore", "format": "json", "limit": 1},
            headers={"User-Agent": "AXIS-LHR/1.0"},
            timeout=6,
        )
        results = r.json()
        if results:
            return float(results[0]["lat"]), float(results[0]["lon"])
    except Exception:
        pass

    return None


# ── page routes ───────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        name="input.html",
        context={"request": request},
        request=request
    )


@app.get("/results", response_class=HTMLResponse)
async def results_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="results.html",
        context={"request": request},
    )


# ── /api/resolve-stop ─────────────────────────────────────────────────────────
@app.post("/api/resolve-stop", response_model=ResolvedStop)
async def resolve_stop(body: ResolveRequest):
    """
    Called on every debounced keystroke from the input page.
    Accepts a raw stop name or a Google Maps URL.
    Returns coords + zone, or valid=False with a reason.
    """
    query = body.query.strip()
    logger.info(f"Resolving stop: {query}")
    if not query:
        raise HTTPException(status_code=400, detail="Empty query.")

    name,lat,lon = None, None,None


    logger.info(query)

    # 1. Google Maps URL → extract coords directly
    name,lat,lon = parse_maps_link(query)
    

    # 2. Free-text name → geocode
    if lat is None:
        result = _geocode_name(query)
        if result is None:
            return ResolvedStop(
                name="", lat=0.0, lon=0.0, zone_id=0,
                valid=False,
                reason="Could not geocode this location.",
            )
        lat, lon = result

    # 3. Boundary validation
    if not _in_lahore(lat, lon):
        return ResolvedStop(
            name=name, lat=lat, lon=lon, zone_id=0,
            valid=False,
            reason="Location is outside Lahore district bounds.",
        )

    # 4. Snap to OSM graph node + zone
    _, zone_id = snap_to_zone(lat, lon)

    return ResolvedStop(
        name=name, lat=lat, lon=lon,
        zone_id=zone_id, valid=True,
    )


# ── /api/optimize ─────────────────────────────────────────────────────────────
@app.post("/api/optimize", response_model=OptimizeResponse)
async def optimize(body: OptimizeRequest):
    """
    Receives already-resolved stops (lat/lon/zone_id from resolve-stop).
    Runs GA pipeline → Google Maps verification → returns best route.
    """
    logger.info(f"Optimize request received: {len(body.stops)} stops")
    if len(body.stops) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 stops.")

    now   = datetime.now()
    stops = []

    for s in body.stops:
        # re-snap is cheap and ensures graph node consistency
        osm_node, zone_id = snap_to_zone(s.lat, s.lon)
        logger.info(f"Stops snapped to graph: {[(s.name, s.zone_id) for s in stops]}")
        deadline = None
        if s.deadline:
            try:
                t        = datetime.strptime(s.deadline, "%H:%M")
                deadline = now.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
            except ValueError:
                pass

        stops.append(Stop(
            name=s.name, osm_node=osm_node, zone_id=zone_id,
            lat=s.lat, lon=s.lon, deadline=deadline,
        ))

    weather                        = simulate_weather()
    table, base, route_feats       = build_cost_table(G, stops, model, weather, now)
    top3                           = run_ga_top3(stops=stops, table=table, start_time=now, elite_k=4)

    logger.info("GA completed successfully")
    best_route, google_mins, score, legs = get_top_best(top3)

    return OptimizeResponse(
        weather     = weather,
        google_mins = round(google_mins, 1),
        final_score = round(score, 1),
        route       = [
            StopOut(
                name     = s.name,
                lat      = s.lat,
                lon      = s.lon,
                zone_id  = s.zone_id,
                deadline = s.deadline.strftime("%H:%M") if s.deadline else None,
            )
            for s in best_route
        ],
        top3 = [
            Top3Item(route=[s.name for s in route], ga_cost=round(cost, 1))
            for route, cost in top3
        ],
        legs = [
            LegOut(**{
                "from": leg["from"],
                "to":       leg["to"],
                "distance": leg["distance"],
                "duration": leg["duration"],
                "steps":    leg["steps"],
            })
            for leg in legs
        ],
    )