from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from src.predict import predict_data

app = FastAPI()

class F1Input(BaseModel):
    GP: str
    Driver: str
    LapNumber: int
    Compound: str

class F1Response(BaseModel):
    predicted_laptime: float

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}

@app.post("/predict", response_model=F1Response)
async def predict_laptime(f1_features: F1Input):
    try:
        features = [{
            "GP": f1_features.GP,
            "Driver": f1_features.Driver,
            "LapNumber": f1_features.LapNumber,
            "Compound": f1_features.Compound
        }]
        prediction = predict_data(features)
        return F1Response(predicted_laptime=float(prediction[0]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
