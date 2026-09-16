import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Drone Propulsion Predictor",
    page_icon="🚁",
    layout="wide"
)


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

@st.cache_resource
def load_models():

    electrical_model_path = (
        BASE_DIR / "elec_model.pkl"
    )

    aerodynamic_model_path = (
        BASE_DIR / "aero_model.pkl"
    )

    elec_model = joblib.load(
        electrical_model_path
    )

    aero_model = joblib.load(
        aerodynamic_model_path
    )

    return elec_model, aero_model


# Load models
elec_model, aero_model = load_models()


# ============================================================
# UNIT CONVERSION
# ============================================================

def kgf_to_newton(thrust_kgf):

    return thrust_kgf * 9.81


# ============================================================
# TWO-STAGE PREDICTION SYSTEM
# ============================================================

def predict_thrust_system(
    voltage,
    current,
    kv,
    diameter,
    pitch
):

    # ========================================================
    # STAGE 1 — ELECTRICAL MODEL
    # ========================================================

    electrical_input = pd.DataFrame([
        {
            "Voltage": voltage,
            "Current": current,
            "KV": kv
        }
    ])


    predicted_rpm = elec_model.predict(
        electrical_input
    )[0]


    # ========================================================
    # STAGE 2 — AERODYNAMIC MODEL
    # ========================================================

    # RPM → revolutions per second
    n = predicted_rpm / 60.0


    # Diameter inch → metre
    D = diameter * 0.0254


    # n²D⁴
    n2D4 = (
        (n ** 2)
        * (D ** 4)
    )


    aerodynamic_input = pd.DataFrame([
        {
            "RPM": predicted_rpm,
            "Diameter": diameter,
            "Pitch": pitch,
            "n2D4": n2D4
        }
    ])


    predicted_thrust_kgf = aero_model.predict(
        aerodynamic_input
    )[0]


    # kgf → Newton
    predicted_thrust_N = kgf_to_newton(
        predicted_thrust_kgf
    )


    return predicted_rpm, predicted_thrust_N


# ============================================================
# DRONE FLIGHT VALIDATION
# ============================================================

def drone_flight_validation(
    thrust_per_motor_N,
    drone_mass,
    number_of_motors
):

    total_thrust_N = (
        thrust_per_motor_N
        * number_of_motors
    )


    drone_weight_N = (
        drone_mass * 9.81
    )


    flight_possible = (
        total_thrust_N > drone_weight_N
    )


    return (
        total_thrust_N,
        drone_weight_N,
        flight_possible
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "🚁 Drone Propulsion Performance Predictor"
)

st.markdown(
    """
    ### Two-Stage Machine Learning Prediction System

    This application uses two Random Forest regression models:

    **Electrical Model**

    Voltage + Current + Motor KV → Predicted RPM

    **Aerodynamic Model**

    Predicted RPM + Propeller Diameter + Pitch + n²D⁴
    → Predicted Thrust
    """
)


st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Propulsion Parameters"
)


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
# DRONE PARAMETERS
# ============================================================

st.sidebar.divider()

st.sidebar.header(
    "🚁 Drone Parameters"
)


drone_mass = st.sidebar.number_input(
    "Drone Mass (kg)",
    min_value=0.0,
    value=2.0,
    step=0.1
)


number_of_motors = st.sidebar.number_input(
    "Number of Motors",
    min_value=1,
    value=4,
    step=1
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.sidebar.divider()

predict_button = st.sidebar.button(
    "🔮 Predict Performance",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    try:

        # Two-stage prediction
        predicted_rpm, predicted_thrust_N = (
            predict_thrust_system(
                voltage=voltage,
                current=current,
                kv=kv,
                diameter=diameter,
                pitch=pitch
            )
        )


        # Convert thrust back to kgf
        predicted_thrust_kgf = (
            predicted_thrust_N / 9.81
        )


        # Drone validation
        (
            total_thrust_N,
            drone_weight_N,
            flight_possible
        ) = drone_flight_validation(
            predicted_thrust_N,
            drone_mass,
            number_of_motors
        )


        # ====================================================
        # RESULTS
        # ====================================================

        st.header(
            "📊 Prediction Results"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Predicted RPM",
                f"{predicted_rpm:,.2f}"
            )


        with col2:

            st.metric(
                "Thrust / Motor",
                f"{predicted_thrust_N:,.2f} N"
            )


        with col3:

            st.metric(
                "Thrust / Motor",
                f"{predicted_thrust_kgf:,.2f} kgf"
            )


        st.divider()


        # ====================================================
        # DRONE VALIDATION
        # ====================================================

        st.header(
            "🚁 Drone Flight Validation"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Total Thrust",
                f"{total_thrust_N:,.2f} N"
            )


        with col2:

            st.metric(
                "Drone Weight",
                f"{drone_weight_N:,.2f} N"
            )


        with col3:

            if flight_possible:

                st.success(
                    "✅ FLIGHT CONDITION SATISFIED"
                )

            else:

                st.error(
                    "❌ FLIGHT CONDITION NOT SATISFIED"
                )


        # ====================================================
        # THRUST-TO-WEIGHT RATIO
        # ====================================================

        if drone_weight_N > 0:

            thrust_to_weight = (
                total_thrust_N
                / drone_weight_N
            )

            st.metric(
                "Thrust-to-Weight Ratio",
                f"{thrust_to_weight:.2f}"
            )


        st.info(
            """
            The flight condition is considered satisfied
            when the predicted total thrust is greater than
            the drone weight.
            """
        )


    except Exception as e:

        st.error(
            "Prediction failed."
        )

        st.exception(e)


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.header(
    "🧠 Model Architecture"
)

architecture_col1, architecture_col2 = st.columns(2)


with architecture_col1:

    st.subheader(
        "Electrical Model"
    )

    st.markdown(
        """
        **Inputs**

        • Voltage (V)  
        • Current (A)  
        • Motor KV  

        **Output**

        → Predicted RPM

        **Algorithm**

        Random Forest Regressor
        """
    )


with architecture_col2:

    st.subheader(
        "Aerodynamic Model"
    )

    st.markdown(
        """
        **Inputs**

        • Predicted RPM  
        • Propeller Diameter  
        • Propeller Pitch  
        • n²D⁴  

        **Output**

        → Predicted Thrust

        **Algorithm**

        Random Forest Regressor
        """
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.divider()

st.header(
    "📈 Feature Importance"
)

importance_file = (
    BASE_DIR / "feature_importance.csv"
)


if importance_file.exists():

    importance_df = pd.read_csv(
        importance_file,
        index_col=0
    )


    st.dataframe(
        importance_df,
        use_container_width=True
    )


    st.subheader(
        "Electrical Model"
    )

    electrical_importance = (
        importance_df["Electrical Model"]
        .sort_values(ascending=False)
    )

    st.bar_chart(
        electrical_importance
    )


    st.subheader(
        "Aerodynamic Model"
    )

    aerodynamic_importance = (
        importance_df["Aerodynamic Model"]
        .sort_values(ascending=False)
    )

    st.bar_chart(
        aerodynamic_importance
    )

else:

    st.warning(
        "feature_importance.csv not found."
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.divider()

st.header(
    "📋 Model Performance"
)

metrics_file = (
    BASE_DIR / "model_metrics.csv"
)


if metrics_file.exists():

    metrics_df = pd.read_csv(
        metrics_file
    )


    st.dataframe(
        metrics_df,
        use_container_width=True,
        hide_index=True
    )


else:

    st.warning(
        "model_metrics.csv not found."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Drone Propulsion Performance Prediction System | "
    "Random Forest Regression"
)
