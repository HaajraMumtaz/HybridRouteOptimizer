import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split

# 1. Load 6,000 rows
df = pd.read_csv('data/raw/lahore_traffic_data.csv')

# We drop hour_of_day because Tier 2 (Peak Multipliers) handles that.
# We drop congestion/volume because those are unknown in the future.

features_to_drop = ['congestion_level', 'traffic_volume', 'hour_of_day', 'travel_time_minutes']
X = df.drop(columns=features_to_drop)
y = df['travel_time_minutes']

# 3. Identify Categorical Features
cat_features = ['road_type', 'city_zone', 'weather_condition', 'day_of_week', 'is_holiday']

# 4. Split data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize CatBoostRegressor
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='RMSE',
    random_seed=42,
    verbose=100 # Shows progress every 100 iterations
)

# Train the model
model.fit(
    X_train, y_train,
    cat_features=cat_features,
    eval_set=(X_test, y_test),
    early_stopping_rounds=50 # Stop if it stops improving
)

model.save_model('models/travel_time_model.cbm')
print("Model saved as travel_time_model.cbm")