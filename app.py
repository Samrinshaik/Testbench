import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Drone Propulsion Predictor",
    page_icon="🚁",
    layout="wide"
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    elec_model = joblib.load(
        "elec_model.pkl"
    )

    aero_model = joblib.load(
        "aero_model.pkl"
    )

    return elec_model, aero_model


elec_model, aero_model = load_models()


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def kgf_to_newton(thrust_kgf):

    return thrust_kgf * 9.81


def predict_thrust_system(
    voltage,
    current,
    kv,
    diameter,
    pitch
):

    # -----------------------------
    # Electrical model
    # -----------------------------

    elec_input = pd.DataFrame([{
        'Voltage': voltage,
        'Current': current,
        'KV': kv
    }])

    predicted_rpm = elec_model.predict(
        elec_input
    )[0]


    # -----------------------------
    # Aerodynamic model
    # -----------------------------

    n = predicted_rpm / 60

    D = diameter * 0.0254

    n2D4 = (
        (n ** 2) *
        (D ** 4)
    )

    aero_input = pd.DataFrame([{
        'RPM': predicted_rpm,
        'Diameter': diameter,
        'Pitch': pitch,
        'n2D4': n2D4
    }])

    predicted_thrust_kgf = aero_model.predict(
        aero_input
    )[0]

    predicted_thrust_N = kgf_to_newton(
        predicted_thrust_kgf
    )

    return predicted_rpm, predicted_thrust_N


def can_drone_fly(
    thrust_per_motor_kgf,
    drone_mass,
    num_motors
):

    total_thrust = (
        kgf_to_newton(thrust_per_motor_kgf)
        * num_motors
    )

    weight = drone_mass * 9.81

    return total_thrust > weight


# ============================================================
# TITLE
# ============================================================

st.title("🚁 Drone Propulsion Performance Predictor")

st.write(
    "Two-stage Random Forest model for predicting "
    "propeller RPM and thrust from electrical and "
    "propeller parameters."
)

st.divider()


# ============================================================
# SIDEBAR INPUTS
# ============================================================

st.sidebar.header("⚙️ System Inputs")

voltage = st.sidebar.number_input(
    "Voltage (V)",
    min_value=0.0,
    value=20.0,
    step=0.1
)

current = st.sidebar.number_input(
    "Current (A)",
    min_value=0.0,
    value=40.0,
    step=0.5
)

kv = st.sidebar.number_input(
    "Motor KV",
    min_value=0.0,
    value=1000.0,
    step=50.0
)

diameter = st.sidebar.number_input(
    "Propeller Diameter (inch)",
    min_value=0.0,
    value=20.0,
    step=0.5
)

pitch = st.sidebar.number_input(
    "Propeller Pitch (inch)",
    min_value=0.0,
    value=4.5,
    step=0.1
)


# ============================================================
# DRONE VALIDATION INPUTS
# ============================================================

st.sidebar.divider()

st.sidebar.header("🚁 Drone Validation")

drone_mass = st.sidebar.number_input(
    "Drone Mass (kg)",
    min_value=0.0,
    value=2.0,
    step=0.1
)

num_motors = st.sidebar.number_input(
    "Number of Motors",
    min_value=1,
    value=4,
    step=1
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

predict_button = st.sidebar.button(
    "Predict Performance",
    use_container_width=True
)


# ============================================================
# RESULTS
# ============================================================

if predict_button:

    rpm, thrust_N = predict_thrust_system(
        voltage=voltage,
        current=current,
        kv=kv,
        diameter=diameter,
        pitch=pitch
    )

    thrust_kgf = thrust_N / 9.81

    total_thrust_N = (
        thrust_N * num_motors
    )

    weight_N = drone_mass * 9.81

    flight_possible = (
        total_thrust_N > weight_N
    )


    # -----------------------------
    # Main results
    # -----------------------------

    st.subheader("Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Predicted RPM",
            f"{rpm:,.2f}"
        )

    with col2:

        st.metric(
            "Thrust per Motor",
            f"{thrust_N:,.2f} N"
        )

    with col3:

        st.metric(
            "Thrust per Motor",
            f"{thrust_kgf:,.2f} kgf"
        )


    st.divider()


    # -----------------------------
    # Drone validation
    # -----------------------------

    st.subheader("Drone Flight Validation")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Thrust",
            f"{total_thrust_N:,.2f} N"
        )

    with col2:

        st.metric(
            "Drone Weight",
            f"{weight_N:,.2f} N"
        )

    with col3:

        if flight_possible:

            st.success("FLIGHT CONDITION SATISFIED")

        else:

            st.error("FLIGHT CONDITION NOT SATISFIED")


    st.info(
        "The flight condition is evaluated by comparing "
        "predicted total thrust with the drone weight."
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.subheader("Model Architecture")

st.markdown(
    """
    **Stage 1 – Electrical Model**

    Voltage + Current + Motor KV → Predicted RPM

    **Stage 2 – Aerodynamic Model**

    Predicted RPM + Propeller Diameter + Pitch + n²D⁴
    → Predicted Thrust

    **Output**

    Predicted RPM + Thrust (N) + Drone flight validation
    """
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.divider()

st.subheader("Feature Importance")

try:

    importance_df = pd.read_csv(
        "feature_importance.csv",
        index_col=0
    )

    st.bar_chart(
        importance_df
    )

except FileNotFoundError:

    st.warning(
        "Feature importance file not found."
    )


# ============================================================
# MODEL METRICS
# ============================================================

st.subheader("Model Performance")

try:

    metrics = pd.read_csv(
        "model_metrics.csv"
    )

    st.dataframe(
        metrics,
        use_container_width=True,
        hide_index=True
    )

except FileNotFoundError:

    st.warning(
        "Model metrics file not found."
    )
