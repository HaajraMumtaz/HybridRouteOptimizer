"""
maps_api/main.py
Takes top 3 GA routes, validates with Google Maps, scores with weighted formula,
plots the best on a static map with turn-by-turn directions.
"""
# maps_api\main.py
import os
import re
import requests
import polyline
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.ndimage import gaussian_filter1d
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
# ── config ────────────────────────────────────────────────────────────────────
MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
ALPHA        = 0.7
BETA         = 0.3
OUTPUT_IMAGE = "best_route_map.png"
MAP_PADDING  = 0.005   # degrees of padding around bounding box


# ── Google Maps Directions API ────────────────────────────────────────────────
def _call_directions(stop_order: list) -> dict | None:
    origin      = f"{stop_order[0].lat},{stop_order[0].lon}"
    destination = f"{stop_order[-1].lat},{stop_order[-1].lon}"
    waypoints   = stop_order[1:-1]

    params = {
        "origin":         origin,
        "destination":    destination,
        "mode":           "driving",
        "key":            MAPS_API_KEY,
        "departure_time": "now",
        "traffic_model":  "best_guess",
    }
    if waypoints:
        wp_str = "|".join(f"{s.lat},{s.lon}" for s in waypoints)
        params["waypoints"] = f"optimize:false|{wp_str}"

    try:
        resp = requests.get(
            "https://maps.googleapis.com/maps/api/directions/json",
            params=params,
            timeout=10,
        )
        data = resp.json()
        if data.get("status") != "OK":
            print(f"  ⚠️  Maps API status: {data.get('status')} — {data.get('error_message','')}")
            return None
        return data
    except Exception as e:
        print(f"  ⚠️  Maps API error: {e}")
        return None


def _parse_directions(data: dict):
    """
    Returns:
        total_minutes,
        all_instructions,
        all_points,
        legs_out
    """

    legs = data["routes"][0]["legs"]

    total_mins = 0.0
    all_instructions = []
    all_points = []
    legs_out = []

    for leg in legs:

        duration_key = (
            "duration_in_traffic"
            if "duration_in_traffic" in leg
            else "duration"
        )

        leg_minutes = leg[duration_key]["value"] / 60.0
        total_mins += leg_minutes

        leg_steps = []

        for step in leg["steps"]:

            raw = step.get("html_instructions", "")

            text = re.sub(r"<[^>]+>", " ", raw).strip()
            text = re.sub(r"\s+", " ", text)

            leg_steps.append(text)
            all_instructions.append(text)

            pts = polyline.decode(step["polyline"]["points"])
            all_points.extend(pts)

        legs_out.append({
            "from": leg.get("start_address", "Unknown"),
            "to": leg.get("end_address", "Unknown"),
            "distance": leg["distance"]["text"],
            "duration": leg[duration_key]["text"],
            "steps": leg_steps
        })

    return (
        total_mins,
        all_instructions,
        all_points,
        legs_out
    )


# ── scoring ───────────────────────────────────────────────────────────────────
def _final_score(ga_time: float, google_time: float) -> float:
    return ALPHA * ga_time + BETA * google_time


# ── route smoothing ───────────────────────────────────────────────────────────
def _smooth_route(lats: list, lons: list, sigma: float = 2.0):
    """Apply Gaussian smoothing to route coordinates."""
    if len(lats) < 4:
        return lats, lons
    s_lats = gaussian_filter1d(np.array(lats), sigma=sigma).tolist()
    s_lons = gaussian_filter1d(np.array(lons), sigma=sigma).tolist()
    return s_lats, s_lons


# ── plot ──────────────────────────────────────────────────────────────────────
def _plot_route(
    stop_order:   list,
    route_points: list[tuple],
    title:        str,
    output_path:  str,
):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor("#f0f0f0")
    fig.patch.set_facecolor("#ffffff")

    if route_points:
        lats = [p[0] for p in route_points]
        lons = [p[1] for p in route_points]

        # smooth route
        s_lats, s_lons = _smooth_route(lats, lons, sigma=2.0)
        ax.plot(s_lons, s_lats, color="#1a73e8", linewidth=3.5,
                alpha=0.9, zorder=2, label="Route", solid_capstyle="round")

        # bounding box zoom with padding
        ax.set_xlim(min(lons) - MAP_PADDING, max(lons) + MAP_PADDING)
        ax.set_ylim(min(lats) - MAP_PADDING, max(lats) + MAP_PADDING)

    # stop markers
    colors = ["#34a853"] + ["#fbbc04"] * (len(stop_order) - 2) + ["#ea4335"]

    for i, stop in enumerate(stop_order):
        ax.scatter(stop.lon, stop.lat,
                   color=colors[i], s=120, zorder=5,
                   edgecolors="white", linewidths=1.5)
        ax.annotate(
            f"  {stop.name}",
            xy=(stop.lon, stop.lat),
            fontsize=9, fontweight="bold",
            color="#202124", zorder=6,
        )

    legend_elements = [
        mlines.Line2D([0], [0], color="#1a73e8", linewidth=3.5, label="Route"),
        mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor="#34a853",
                      markersize=10, label="Start"),
        mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor="#fbbc04",
                      markersize=10, label="Waypoint"),
        mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor="#ea4335",
                      markersize=10, label="End"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  🗺️  Map saved → {output_path}")


# ── main entry point ──────────────────────────────────────────────────────────
def get_top_best(top3: list[tuple[list, float]]) -> tuple[list, float, float]:
    """
    Takes top3 from GA: [(route, ga_cost), ...]
    Calls Maps API once per route, stores parsed result,
    scores with final_score = α*ga_time + β*google_time,
    plots winner, prints turn-by-turn.

    Returns (best_route, google_minutes, final_score)
    """
    print("\n── Google Maps Verification ─────────────────────────────────────")

    results = []

    for i, (route, ga_cost) in enumerate(top3):
        route_str = " → ".join(s.name for s in route)
        print(f"\n  Option {i+1}: {route_str}")
        print(f"    GA estimated : {ga_cost:.1f} min")

        data = _call_directions(route)

        if data is None:
            results.append({
                "route":       route,
                "ga_cost":     ga_cost,
                "google_mins": ga_cost,   # fallback to GA estimate
                "score":       ga_cost,
                "instructions": [],
                "points":      [],
                "legs":        [],
            })
  
            print(f"    Google Maps  : FAILED")
            continue

        # parse once, store everything
        google_mins, instructions, points,legs = _parse_directions(data)
        score = _final_score(ga_cost, google_mins)
        
        print(f"    Google Maps  : {google_mins:.1f} min")
        print(f"    Final score  : {ALPHA}×{ga_cost:.1f} + {BETA}×{google_mins:.1f} = {score:.1f}")

        results.append({
            "route":        route,
            "ga_cost":      ga_cost,
            "google_mins":  google_mins,
            "score":        score,
            "instructions": instructions,
            "points":       points,
            "legs":legs,
        })

    # ── pick winner by final score ────────────────────────────────────────────
    winner    = min(results, key=lambda x: x["score"])
    w_route   = winner["route"]
    route_str = " → ".join(s.name for s in w_route)

    print(f"\n  ✅ Best route  : {route_str}")
    print(f"     GA time     : {winner['ga_cost']:.1f} min")
    print(f"     Google time : {winner['google_mins']:.1f} min")
    print(f"     Final score : {winner['score']:.1f}")

    # ── turn-by-turn from stored result — no second API call ─────────────────
    instructions = winner["instructions"]
    points       = winner["points"]

    if instructions:
        print(f"\n── Turn-by-Turn Directions ──────────────────────────────────────")
        for j, step in enumerate(instructions, 1):
            print(f"  {j:>3}. {step}")

    if points:
        _plot_route(
            stop_order   = w_route,
            route_points = points,
            title        = (
                f"Best Route: {route_str}\n"
                f"GA: {winner['ga_cost']:.1f} min  |  "
                f"Google: {winner['google_mins']:.1f} min  |  "
                f"Score: {winner['score']:.1f}"
            ),
            output_path  = OUTPUT_IMAGE,
        )

    return w_route, winner["google_mins"], winner["score"],winner["legs"]