from fastapi import FastAPI, Query, HTTPException
import torch
import numpy as np
import pandas as pd
from .data_collector import OpenAQProcessor
from model.models import LSTM, GRU, BiLSTM
from model.config import Configuration
import os

app = FastAPI()

c = Configuration()

# Static values
INPUT_SIZE = c.DATA_INPUT_SIZE
OFFSET = c.DATA_OFFSET
MODEL_INPUT_SIZE = c.MODEL_INPUT_SIZE
MODEL_HIDDEN_SIZE = c.MODEL_HIDDEN_SIZE
MODEL_OUTPUT_SIZE = c.MODEL_OUTPUT_SIZE
MODEL_NUM_LAYERS = c.MODEL_NUM_LAYERS

# Mapping tên mô hình với lớp tương ứng
MODEL_CLASSES = {
    "LSTM": LSTM,
    "GRU": GRU,
    "BiLSTM": BiLSTM,
}


@app.get("/")
def root():
    return {"message": "Air quality prediction API is running."}


@app.get("/predict")
def predict(
    ahead: int = Query(default=24, ge=1, le=72),
    model_name: str = Query(default="LSTM")
):
    model_name = model_name.upper()
    if model_name not in MODEL_CLASSES:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model_name}")

    try:
        processor = OpenAQProcessor(API_KEY="YOUR_API_KEY")
        df_raw = processor.get_data_for_prediction()
        df_processed = processor.preprocess(df_raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading or preprocessing data: {str(e)}")

    if len(df_processed) < INPUT_SIZE + OFFSET:
        raise HTTPException(status_code=400, detail="Not enough data to make prediction.")

    input_data = df_processed.iloc[-(INPUT_SIZE + OFFSET):-OFFSET].values
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)

    model_class = MODEL_CLASSES[model_name]
    model_path = f"model/best-{model_name}.pth"

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model file not found: {model_path}")

    model = model_class(
        input_size=MODEL_INPUT_SIZE,
        hidden_size=MODEL_HIDDEN_SIZE,
        output_size=MODEL_OUTPUT_SIZE,
        ahead=ahead,
        num_layers=MODEL_NUM_LAYERS,
    )

    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    with torch.no_grad():
        prediction = model(input_tensor).squeeze(0).numpy()

    prediction_df = pd.DataFrame(np.expm1(prediction), columns=["co", "pm25", "no2"])

    last_input_time = df_processed.index[-(OFFSET + 1)]
    start_time = last_input_time + pd.Timedelta(hours=OFFSET + 1)

    prediction_df["timestamp"] = pd.date_range(start=start_time, periods=ahead, freq="1h")

    return {
        "model": model_name,
        "ahead": ahead,
        "prediction": prediction_df.to_dict(orient="records")
    }
