import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# -----------------------------
# Utility function
# -----------------------------
def kgf_to_newton(thrust_kgf):
    return thrust_kgf * 9.81


# -----------------------------
# Load and clean dataset
# -----------------------------
df = pd.read_csv("converted_file.csv")

df.columns = df.columns.str.strip()

df = df.rename(columns={
    'Thrust (kgf)': 'Thrust',
    'Rotation speed (rpm)': 'RPM',
    'Voltage (V)': 'Voltage',
    'Current (A)': 'Current',
    'propeller size(diameter)': 'Diameter',
    'propeller size(pitch)': 'Pitch',
    'motor kv': 'KV'
})

df = df[
    ['RPM', 'Diameter', 'Pitch',
     'Voltage', 'Current', 'KV', 'Thrust']
]

df = df.apply(pd.to_numeric, errors='coerce')

df = df.dropna(
    subset=['RPM', 'Diameter', 'Voltage', 'KV', 'Thrust']
)

print("Cleaned dataset:", df.shape)


# ============================================================
# ELECTRICAL MODEL
# ============================================================

df_elec = df[
    ['Voltage', 'Current', 'KV', 'RPM']
].dropna()

X_elec = df_elec[
    ['Voltage', 'Current', 'KV']
]

y_elec = df_elec['RPM']

X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
    X_elec,
    y_elec,
    test_size=0.2,
    random_state=42
)

elec_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

elec_model.fit(X_train_e, y_train_e)

# Predictions
y_pred_e = elec_model.predict(X_test_e)

# Metrics
r2_e = r2_score(y_test_e, y_pred_e)
mae_e = mean_absolute_error(y_test_e, y_pred_e)
rmse_e = np.sqrt(
    mean_squared_error(y_test_e, y_pred_e)
)

print("\nElectrical Model")
print("R2:", r2_e)
print("MAE:", mae_e)
print("RMSE:", rmse_e)


# ============================================================
# AERODYNAMIC MODEL
# ============================================================

df['n'] = df['RPM'] / 60

df['D'] = df['Diameter'] * 0.0254

df['n2D4'] = (
    (df['n'] ** 2) *
    (df['D'] ** 4)
)

X_aero = df[
    ['RPM', 'Diameter', 'Pitch', 'n2D4']
]

y_aero = df['Thrust']

X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
    X_aero,
    y_aero,
    test_size=0.2,
    random_state=42
)

aero_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

aero_model.fit(X_train_a, y_train_a)

# Predictions
y_pred_a = aero_model.predict(X_test_a)

# Metrics
r2_a = r2_score(y_test_a, y_pred_a)

y_test_a_N = kgf_to_newton(y_test_a)
y_pred_a_N = kgf_to_newton(y_pred_a)

mae_a = mean_absolute_error(
    y_test_a_N,
    y_pred_a_N
)

rmse_a = np.sqrt(
    mean_squared_error(
        y_test_a_N,
        y_pred_a_N
    )
)

print("\nAerodynamic Model")
print("R2:", r2_a)
print("MAE (N):", mae_a)
print("RMSE (N):", rmse_a)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

elec_importance = pd.Series(
    elec_model.feature_importances_,
    index=X_elec.columns
)

aero_importance = pd.Series(
    aero_model.feature_importances_,
    index=X_aero.columns
)

importance_df = pd.DataFrame({
    'Electrical Model': elec_importance,
    'Aerodynamic Model': aero_importance
}).fillna(0)

print("\nFeature Importance")
print(importance_df)


# ============================================================
# SAVE MODELS
# ============================================================

joblib.dump(
    elec_model,
    "elec_model.pkl"
)

joblib.dump(
    aero_model,
    "aero_model.pkl"
)

# Save feature importance
importance_df.to_csv(
    "feature_importance.csv"
)

# Save metrics
metrics = pd.DataFrame({
    'Model': [
        'Electrical',
        'Aerodynamic'
    ],
    'R2': [
        r2_e,
        r2_a
    ],
    'MAE': [
        mae_e,
        mae_a
    ],
    'RMSE': [
        rmse_e,
        rmse_a
    ]
})

metrics.to_csv(
    "model_metrics.csv",
    index=False
)

print("\nModels saved successfully!")
