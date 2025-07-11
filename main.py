from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

# Load the trained model
model = joblib.load("credit_default_model.pkl")  # Rename your saved model to model.pkl

class CreditData(BaseModel):
    features: list  # 1D list of 23 feature values (same order as training)

@app.post("/predict")
def predict_default(data: CreditData):
    try:
        input_array = np.array(data.features).reshape(1, -1)
        prediction = model.predict(input_array)[0]
        return {"prediction": int(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
