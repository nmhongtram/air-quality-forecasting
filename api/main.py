from fastapi import FastAPI, Query, HTTPException
import torch
import numpy as np
import pandas as pd
from .data_collector import OpenAQProcessor
from model.models import LSTM, GRU, RNN
from model.config import Configuration
import os

app = FastAPI()

c = Configuration()

INPUT_SIZE = c.DATA_INPUT_SIZE
OFFSET = c.DATA_OFFSET

# Mapping tên mô hình với lớp tương ứng
MODEL_CLASSES = {
    "RNN": RNN,
    "LSTM": LSTM,
    "GRU": GRU
}


@app.get("/")
def root():
    return {"message": "Air quality prediction API is running."}


@app.get("/predict")
async def predict(
    ahead: int = 24,
    model_name: str = Query(default="LSTM")
):
    if ahead > 24:
        return {"error": "Model only supports prediction up to 24 hours ahead."}
    
    model_name = model_name.upper()
    if model_name not in MODEL_CLASSES:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model_name}")

    try:
        processor = OpenAQProcessor(API_KEY="94d98cbdd0c42919e1017aa7c619b0ec47fa75c988e232eafb8e8f8e01c3584e")
        df_raw = processor.get_data_for_prediction()
        df_processed = processor.preprocess(df_raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading or preprocessing data: {str(e)}")

    if len(df_processed) < INPUT_SIZE + OFFSET:
        raise HTTPException(status_code=400, detail="Not enough data to make prediction.")

    input_data = df_processed.iloc[-(INPUT_SIZE + OFFSET)+1:-OFFSET+1].values
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)

    model_class = MODEL_CLASSES[model_name]
    model_path = f"model/best-{model_name}.pth"

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model file not found: {model_path}")

    model = model_class(
        input_size=c.MODEL_INPUT_SIZE,
        hidden_size=c.MODEL_HIDDEN_SIZE,
        output_size=c.MODEL_OUTPUT_SIZE,
        ahead=c.MODEL_AHEAD,
        num_layers=c.MODEL_NUM_LAYERS,
    )

    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    with torch.no_grad():
        prediction = model(input_tensor).squeeze(0).numpy()[:ahead]

    prediction_df = pd.DataFrame(np.expm1(prediction), columns=["pm25", "co", "no2"])

    last_input_time = df_processed.index[-OFFSET]
    start_time = last_input_time + pd.Timedelta(hours=OFFSET + 1)

    prediction_df["datetimeFrom_local"] = pd.date_range(start=start_time, periods=ahead, freq="1h")

    return {
        "model": model_name,
        "ahead": ahead,
        "prediction": prediction_df.to_dict(orient="records")
    }

# uvicorn api.main:app --reload