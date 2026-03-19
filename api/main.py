from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# ------------------------
# Create FastAPI app first
# ------------------------
app = FastAPI(
    title="Maternal Health Risk Prediction API",
    description="Predicts maternal health risk using ML",
    version="1.0"
)

# ------------------------
# Mount static files for favicon
# ------------------------
app.mount("/static", StaticFiles(directory="mlops/api/static"), name="static")

@app.get("/favicon.ico")
async def favicon():
    return FileResponse("mlops/api/static/favicon.ico")

# ------------------------
# Load ML artifacts
# ------------------------
model = joblib.load("mlops/model/rf_model.pkl")
label_encoder = joblib.load("mlops/model/label_encoder.pkl")

# ------------------------
# Input schema
# ------------------------
class PatientData(BaseModel):
    age: int
    systolic_bp: float
    diastolic_bp: float
    blood_sugar: float
    body_temp: float
    heart_rate: float

# ------------------------
# Root route
# ------------------------
@app.get("/")
def home():
    return {"message": "Maternal Health Risk API is running"}

# ------------------------
# Prediction route
# ------------------------
@app.post("/predict")
def predict_risk(data: PatientData):
    input_array = np.array([[
        data.age,
        data.systolic_bp,
        data.diastolic_bp,
        data.blood_sugar,
        data.body_temp,
        data.heart_rate
    ]])

    pred_encoded = model.predict(input_array)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    return {
        "risk_level": pred_label
    }