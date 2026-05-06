import os
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ─── 1. LOAD ─────────────────────────────────────────────────────────────────
df = pd.read_csv('data/raw/lahore_traffic_dataset.csv')
print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} cols")
print(df.dtypes)
print(df.head(3).to_string())

# ─── 2. VALIDATE ─────────────────────────────────────────────────────────────
assert df['congestion_multiplier'].between(0.55, 3.5).all(), "Target out of range!"
assert (df['origin_zone'] != df['dest_zone']).all(), "Self-loops in data!"
assert df.isnull().sum().sum() == 0, "Null values found!"
print(f"\n✅ Validation passed")
print(f"Target range: {df['congestion_multiplier'].min():.3f} – {df['congestion_multiplier'].max():.3f}")
print(f"Target mean:  {df['congestion_multiplier'].mean():.3f}")
print(f"Target std:   {df['congestion_multiplier'].std():.3f}")

# ─── 3. FEATURES ─────────────────────────────────────────────────────────────
TARGET = 'congestion_multiplier'

CAT_FEATURES = [
    'origin_zone', 'dest_zone', 'road_type',
    'is_one_way', 'has_signal', 'is_construction',
    'weather_condition', 'day_of_week', 'time_slot',
    'is_weekend', 'is_holiday', 'day_type'
]

NUM_FEATURES = [
    'speed_limit_kmh', 'num_lanes', 'distance_km', 'road_curvature'
]

# Cast categoricals to str so CatBoost handles them correctly
for col in CAT_FEATURES:
    df[col] = df[col].astype(str)

X = df[CAT_FEATURES + NUM_FEATURES]
y = df[TARGET]

print(f"\nFeatures: {list(X.columns)}")
print(f"Cat: {CAT_FEATURES}")
print(f"Num: {NUM_FEATURES}")

# ─── 4. SPLIT ────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42
)

print(f"\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

train_pool = Pool(X_train, y_train, cat_features=CAT_FEATURES)
val_pool   = Pool(X_val,   y_val,   cat_features=CAT_FEATURES)
test_pool  = Pool(X_test,  y_test,  cat_features=CAT_FEATURES)

# ─── 5. TRAIN ────────────────────────────────────────────────────────────────
model = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.04,
    depth=8,
    l2_leaf_reg=3,
    min_data_in_leaf=10,
    loss_function='RMSE',
    eval_metric='RMSE',
    random_seed=42,
    verbose=200,
)

model.fit(
    train_pool,
    eval_set=val_pool,
    early_stopping_rounds=100,
)

# ─── 6. EVALUATE ─────────────────────────────────────────────────────────────
def evaluate(pool, y_true, split_name):
    preds = model.predict(pool)
    mae   = mean_absolute_error(y_true, preds)
    rmse  = np.sqrt(mean_squared_error(y_true, preds))
    r2    = r2_score(y_true, preds)
    mape  = np.mean(np.abs((y_true - preds) / y_true)) * 100
    print(f"\n── {split_name} ──")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")
    print(f"  MAPE : {mape:.2f}%")
    return preds

print("\n=== EVALUATION ===")
val_preds  = evaluate(val_pool,  y_val,  "Validation")
test_preds = evaluate(test_pool, y_test, "Test")

# ─── 7. SANITY CHECKS ────────────────────────────────────────────────────────
print("\n=== SANITY CHECKS ===")

# Check model learned zone personalities
test_df = X_test.copy()
test_df['actual'] = y_test.values
test_df['pred']   = test_preds

print("\nMean predicted multiplier by time_slot:")
print(test_df.groupby('time_slot')['pred'].mean().sort_values(ascending=False).to_string())

print("\nMean predicted multiplier by road_type:")
print(test_df.groupby('road_type')['pred'].mean().sort_values(ascending=False).to_string())

print("\nMean predicted multiplier by day_type:")
print(test_df.groupby('day_type')['pred'].mean().sort_values(ascending=False).to_string())

print("\nMean predicted multiplier by weather_condition:")
print(test_df.groupby('weather_condition')['pred'].mean().sort_values(ascending=False).to_string())

# ─── 8. FEATURE IMPORTANCE ───────────────────────────────────────────────────
fi = pd.Series(
    model.get_feature_importance(),
    index=CAT_FEATURES + NUM_FEATURES
).sort_values(ascending=False)

print("\n=== FEATURE IMPORTANCE ===")
print(fi.to_string())

# ─── 9. PLOTS ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Actual vs predicted
axes[0].scatter(y_test, test_preds, alpha=0.3, s=10)
axes[0].plot([0.55, 3.5], [0.55, 3.5], 'r--')
axes[0].set_xlabel('Actual')
axes[0].set_ylabel('Predicted')
axes[0].set_title('Actual vs Predicted')

# Residuals
residuals = y_test.values - test_preds
axes[1].hist(residuals, bins=50, edgecolor='black')
axes[1].axvline(0, color='red', linestyle='--')
axes[1].set_xlabel('Residual')
axes[1].set_title('Residual Distribution')

# Feature importance
fi.head(12).plot(kind='barh', ax=axes[2])
axes[2].invert_yaxis()
axes[2].set_title('Feature Importance')

plt.tight_layout()
plt.savefig('models/training_diagnostics.png', dpi=150)
plt.show()

# ─── 10. SAVE ────────────────────────────────────────────────────────────────
os.makedirs('models', exist_ok=True)
model.save_model('models/congestion_multiplier_v1.cbm')
print("\n✅ Model saved → models/congestion_multiplier_v1.cbm")

