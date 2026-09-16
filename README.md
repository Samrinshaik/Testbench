# Drone Propulsion Performance Predictor

A two-stage machine learning system for predicting drone propulsion performance.

## Model Architecture

### Stage 1: Electrical Model

Inputs:

- Voltage
- Current
- Motor KV

Output:

- Predicted RPM

### Stage 2: Aerodynamic Model

Inputs:

- Predicted RPM
- Propeller Diameter
- Propeller Pitch
- n²D⁴

Output:

- Predicted Thrust

## Machine Learning Algorithm

Random Forest Regression.

## Application

The Streamlit application predicts:

- RPM
- Thrust per motor
- Total drone thrust
- Drone weight
- Thrust-to-weight ratio
- Basic flight-condition validation

## Running the Application

```bash
pip install -r requirements.txt
streamlit run app.py
