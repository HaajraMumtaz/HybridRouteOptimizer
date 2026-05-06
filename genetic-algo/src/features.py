from datetime import datetime

TIME_SLOTS = {
    'early_morning': (0,  7),
    'morning_peak':  (7,  10),
    'midday':        (10, 14),
    'afternoon':     (14, 17),
    'evening_peak':  (17, 21),
    'night':         (21, 24),
}
DAY_NAMES = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
ALL_SLOTS  = list(TIME_SLOTS.keys())   # 6 slots, fixed order

CAT_FEATURES = [
    "origin_zone","dest_zone","road_type","is_one_way","has_signal",
    "is_construction","weather_condition","day_of_week","time_slot",
    "is_weekend","is_holiday","day_type",
]
NUM_FEATURES = ["speed_limit_kmh","num_lanes","distance_km","road_curvature"]
ALL_FEATURES  = CAT_FEATURES + NUM_FEATURES


def get_time_context(now: datetime | None = None) -> dict:
    now = now or datetime.now()
    dow = DAY_NAMES[now.weekday()]
    is_weekend = '1' if now.weekday() >= 5 else '0'
    is_holiday = '0'   # ← wire in real holiday calendar here later
    for slot, (s, e) in TIME_SLOTS.items():
        if s <= now.hour < e:
            break
    else:
        slot = 'night'
    day_type = 'holiday' if is_holiday == '1' else ('weekend' if is_weekend == '1' else 'weekday')
    return {
        'day_of_week': dow,
        'time_slot':   slot,
        'is_weekend':  is_weekend,
        'is_holiday':  is_holiday,
        'day_type':    day_type,
    }


def slot_contexts(base_now: datetime | None = None) -> list[dict]:
    """Return one time-context dict per slot, anchored to today's date."""
    now  = base_now or datetime.now()
    ctxs = []
    for slot, (start_h, _) in TIME_SLOTS.items():
        fake = now.replace(hour=start_h, minute=0, second=0, microsecond=0)
        ctxs.append(get_time_context(fake))
    return ctxs    # length == 6, same order as ALL_SLOTS


def build_features(route_data: dict, context: dict) -> dict:
    """
    Combine route_data (from routing.aggregate_path_features)
    + context (origin_zone, dest_zone, weather, time keys, is_construction)
    → flat dict with ALL_FEATURES in correct types.
    """
    return {
        # categorical
        "origin_zone":       str(context.get('origin_zone', '15')),
        "dest_zone":         str(context.get('dest_zone',   '15')),
        "road_type":         route_data.get('road_type',         'street'),
        "is_one_way":        route_data.get('is_one_way',         '0'),
        "has_signal":        route_data.get('has_signal',         '0'),
        "is_construction":   context.get('is_construction',       '0'),
        "weather_condition": context.get('weather_condition',     'clear'),
        "day_of_week":       context.get('day_of_week',           'Monday'),
        "time_slot":         context.get('time_slot',             'midday'),
        "is_weekend":        context.get('is_weekend',            '0'),
        "is_holiday":        context.get('is_holiday',            '0'),
        "day_type":          context.get('day_type',              'weekday'),
        # numerical
        "speed_limit_kmh":   float(route_data.get('speed_limit_kmh', 30.0)),
        "num_lanes":         float(route_data.get('num_lanes',         1.0)),
        "distance_km":       float(route_data.get('distance_km',       0.1)),
        "road_curvature":    float(route_data.get('road_curvature',    0.0)),
    }