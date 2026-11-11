"""
Wind Turbine Predictive Maintenance API
=======================================

This module exposes a REST API built with FastAPI to serve machine learning 
predictions for wind turbine operational status. It ingests real-time 
telemetry data from turbine sensors and predicts whether the turbine is 
operating normally, requires a warning inspection, or is at risk of a 
critical failure.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# ==============================================================================
# 1. APPLICATION & MODEL INITIALIZATION
# ==============================================================================
app = FastAPI(
    title="Wind Turbine Predictive Maintenance API",
    description="API for predicting wind turbine failures based on sensor telemetry.",
    version="1.0.0"
)

# Load the trained machine learning model globally on startup to minimize latency.
try:
    model = joblib.load('models/wind_turbine_model.pkl')
except Exception as e:
    model = None
    print(f"CRITICAL: Error loading the predictive model: {e}")


# ==============================================================================
# 2. DATA SCHEMAS
# ==============================================================================
class TurbineTelemetry(BaseModel):
    """
    Pydantic schema defining the expected JSON payload for the API.
    Includes 20 numeric sensor readings and 2 categorical metadata fields.
    """
    Gen_Avg_RPM: float
    Gen_Output_Voltage: float
    Gen_Coil_Temp: float
    Grid_Frequency_Hz: float
    Rotor_Blade_Angle_1: float
    Rotor_Blade_Angle_2: float
    Rotor_Wind_Speed: float
    Rotor_Shaft_Vibration: float
    Hydraulic_Oil_Pressure: float
    Hydraulic_Oil_Temp: float
    Hydraulic_Tank_Level: float
    Gearbox_Bearing_Temp: float
    Gearbox_Vibration_X: float
    Gearbox_Vibration_Y: float
    Amb_Ext_Temp: float
    Amb_Humidity: float
    Amb_Wind_Turbulence: float
    Tower_Oscillation: float
    Tower_Fatigue_Index: float
    System_Efficiency_Calc: float
    Installation_Region: str
    Turbine_Model: str


# ==============================================================================
# 3. ENDPOINTS
# ==============================================================================
@app.post("/predict")
def predict_status(data: TurbineTelemetry):
    """
    Predicts the operational status of a wind turbine.
    
    Args:
        data (TurbineTelemetry): JSON payload containing sensor readings.
        
    Returns:
        dict: A dictionary containing the numeric prediction code and the 
              human-readable status label.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded into memory.")
    
    # Convert the validated JSON payload into a Pandas DataFrame
    input_df = pd.DataFrame([data.model_dump()])

    # Apply manual One-Hot Encoding to categorical fields
    input_df_encoded = pd.get_dummies(input_df)

    # Retrieve the exact features the model saw during training
    expected_columns = model.feature_names_in_

    # Align the incoming payload with the training schema. 
    # Any missing categorical columns are automatically filled with 0.
    input_df_aligned = input_df_encoded.reindex(columns=expected_columns, fill_value=0)
    
    # Generate the prediction using the aligned DataFrame
    prediction = model.predict(input_df_aligned)[0]
    
    # Map the numerical output to business logic
    status_mapping = {0: "Normal", 1: "Warning", 2: "Failure"}
    result = status_mapping.get(prediction, "Unknown")
    
    return {
        "prediction_code": int(prediction),
        "status": result
    }