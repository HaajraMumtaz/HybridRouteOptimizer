results = []

for (u, v) in sample_pairs:

    osmnx_time = get_osmnx_time(u, v)

    features = {
        'road_type': ...,
        'city_zone': ...,
        'weather_condition': ...,
        'day_of_week': ...,
        'is_holiday': '0',
        'is_weekend': '0',
        'is_construction': '0'
    }

    model_time = model.predict(pd.DataFrame([features]))[0]

    results.append({
        'osmnx_time': osmnx_time,
        'model_time': model_time,
        'ratio': model_time / osmnx_time
    })