import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)


# ============================================================
# 1. UNIT CONVERSION
# ============================================================

def kgf_to_newton(thrust_kgf):
    return thrust_kgf * 9.81


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv("converted_file.csv")

print("Initial dataset shape:", df.shape)

# Remove extra spaces from column names
df.columns = df.columns.str.strip()


# ============================================================
# 3. RENAME COLUMNS
# ============================================================

df = df.rename(columns={
    "Thrust (kgf)": "Thrust",
    "Rotation speed (rpm)": "RPM",
    "Voltage (V)": "Voltage",
    "Current (A)": "Current",
    "propeller size(diameter)": "Diameter",
    "propeller size(pitch)": "Pitch",
    "motor kv": "KV"
})


# ============================================================
# 4. KEEP REQUIRED COLUMNS
# ============================================================

required_columns = [
    "RPM",
    "Diameter",
    "Pitch",
    "Voltage",
    "Current",
    "KV",
    "Thrust"
]

df = df[required_columns]


# ============================================================
# 5. CONVERT TO NUMERIC
# ============================================================

df = df.apply(
    pd.to_numeric,
    errors="coerce"
)


# ============================================================
# 6. REMOVE MISSING VALUES
# ============================================================

df = df.dropna(
    subset=[
        "RPM",
        "Diameter",
        "Pitch",
        "Voltage",
        "Current",
        "KV",
        "Thrust"
    ]
)

print("Dataset after cleaning:", df.shape)


# ============================================================
# 7. ELECTRICAL MODEL
#
# Voltage + Current + KV
#          ↓
#       RPM
# ============================================================

df_elec = df[
    [
        "Voltage",
        "Current",
        "KV",
        "RPM"
    ]
].dropna()


X_elec = df_elec[
    [
        "Voltage",
        "Current",
        "KV"
    ]
]

y_elec = df_elec["RPM"]


# Train-test split
X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
    X_elec,
    y_elec,
    test_size=0.20,
    random_state=42
)


# Random Forest Electrical Model
elec_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

elec_model.fit(
    X_train_e,
    y_train_e
)


# Predictions
y_pred_e = elec_model.predict(
    X_test_e
)


# Electrical metrics
r2_e = r2_score(
    y_test_e,
    y_pred_e
)

mae_e = mean_absolute_error(
    y_test_e,
    y_pred_e
)

rmse_e = np.sqrt(
    mean_squared_error(
        y_test_e,
        y_pred_e
    )
)


print("\n================================")
print("ELECTRICAL MODEL PERFORMANCE")
print("================================")

print(f"R²   : {r2_e:.4f}")
print(f"MAE  : {mae_e:.4f} RPM")
print(f"RMSE : {rmse_e:.4f} RPM")


# ============================================================
# 8. AERODYNAMIC MODEL
#
# RPM + Diameter + Pitch + n²D⁴
#              ↓
#            Thrust
# ============================================================

# Rotational speed in revolutions per second
df["n"] = df["RPM"] / 60.0


# Diameter from inch to metre
df["D"] = df["Diameter"] * 0.0254


# Aerodynamic parameter
df["n2D4"] = (
    (df["n"] ** 2)
    * (df["D"] ** 4)
)


# Features
X_aero = df[
    [
        "RPM",
        "Diameter",
        "Pitch",
        "n2D4"
    ]
]

# Target
y_aero = df["Thrust"]


# Train-test split
X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
    X_aero,
    y_aero,
    test_size=0.20,
    random_state=42
)


# Random Forest Aerodynamic Model
aero_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

aero_model.fit(
    X_train_a,
    y_train_a
)


# Predictions
y_pred_a = aero_model.predict(
    X_test_a
)


# R²
r2_a = r2_score(
    y_test_a,
    y_pred_a
)


# Convert kgf → N
y_test_a_N = kgf_to_newton(
    y_test_a
)

y_pred_a_N = kgf_to_newton(
    y_pred_a
)


# MAE
mae_a = mean_absolute_error(
    y_test_a_N,
    y_pred_a_N
)


# RMSE
rmse_a = np.sqrt(
    mean_squared_error(
        y_test_a_N,
        y_pred_a_N
    )
)


print("\n================================")
print("AERODYNAMIC MODEL PERFORMANCE")
print("================================")

print(f"R²   : {r2_a:.4f}")
print(f"MAE  : {mae_a:.4f} N")
print(f"RMSE : {rmse_a:.4f} N")


# ============================================================
# 9. FEATURE IMPORTANCE
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
    "Electrical Model": elec_importance,
    "Aerodynamic Model": aero_importance
})


importance_df = importance_df.fillna(0)


print("\n================================")
print("FEATURE IMPORTANCE")
print("================================")

print(importance_df)


# ============================================================
# 10. SAVE TRAINED MODELS
# ============================================================

joblib.dump(
    elec_model,
    "elec_model.pkl"
)

joblib.dump(
    aero_model,
    "aero_model.pkl"
)


# ============================================================
# 11. SAVE FEATURE IMPORTANCE
# ============================================================

importance_df.to_csv(
    "feature_importance.csv"
)


# ============================================================
# 12. SAVE MODEL METRICS
# ============================================================

metrics_df = pd.DataFrame({
    "Model": [
        "Electrical",
        "Aerodynamic"
    ],

    "R2": [
        r2_e,
        r2_a
    ],

    "MAE": [
        mae_e,
        mae_a
    ],

    "RMSE": [
        rmse_e,
        rmse_a
    ]
})


metrics_df.to_csv(
    "model_metrics.csv",
    index=False
)


# ============================================================
# 13. FINISHED
# ============================================================

print("\n================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("================================")

print("\nGenerated files:")

print("1. elec_model.pkl")
print("2. aero_model.pkl")
print("3. feature_importance.csv")
print("4. model_metrics.csv")
