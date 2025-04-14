import asyncio
import os
import time
import pickle
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException, WebSocket
from contextlib import asynccontextmanager
from typing import Dict, Any
from queue import Queue
from logging.handlers import QueueHandler

# Import models from external file
from models import Transaction, PredictionRequest, AVAILABLE_MODELS

# ======================================
# Logging Configuration
# ======================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()  # root logger
logger.setLevel(logging.INFO)

# Configure logging to send messages to a queue
uvicorn_logger = logging.getLogger("uvicorn")
uvicorn_logger.setLevel(logging.INFO)

# Create a queue for logging
log_queue = Queue()

# Create a queue handler
queue_handler = QueueHandler(log_queue)

logger.addHandler(queue_handler)

# Add the queue handler to the logger
uvicorn_logger.addHandler(queue_handler)

# ANSI escape codes for colored output
RED_BG = '\033[41m'  # Red background for fraud
GREEN_BG = '\033[42m'  # Green background for no fraud
RESET = '\033[0m'  # Reset color

# ======================================
# Global variables
# ======================================
MODELS = {}
SCALER = None
TRAINING_COLS = None
CATEGORICAL_MAPPINGS = {}

# ======================================
# Helper functions
# ======================================

def load_categorical_mappings():
    """Load categorical mappings from CSV file."""
    global CATEGORICAL_MAPPINGS
    mapping_path = "categorical_mappings.csv"
    mapping_df = pd.read_csv(mapping_path)
    for col in mapping_df["Column"].unique():
        col_mapping = mapping_df[mapping_df["Column"] == col]
        CATEGORICAL_MAPPINGS[col] = {
            row["Code"]: row["Value"] for _, row in col_mapping.iterrows()
        }

def load_models_and_scaler():
    """Load all ML and DNN models, scaler, and training columns."""
    global MODELS, SCALER, TRAINING_COLS
    model_dir = "models"

    # Load classic ML models
    for name in AVAILABLE_MODELS:
        model_path = os.path.join(model_dir, f"{name}_best_model.pkl")
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                MODELS[name] = pickle.load(f)

    # Load Neural Network model
    dnn_path = os.path.join(model_dir, "Default_Neural_Network_best_model.keras")
    if os.path.exists(dnn_path):
        MODELS["Neural Network"] = tf.keras.models.load_model(dnn_path)

    # Load scaler
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    with open(scaler_path, "rb") as f:
        SCALER = pickle.load(f)

    # Load training columns
    cols_path = os.path.join(model_dir, "training_cols.pkl")
    with open(cols_path, "rb") as f:
        TRAINING_COLS = pickle.load(f)

def preprocess_transaction(transaction: Transaction) -> np.ndarray:
    """Prepare transaction data for model inference."""
    df = pd.DataFrame({
        "Transaction_Amount": [transaction.ta],
        "Transaction_Type": [CATEGORICAL_MAPPINGS["Transaction_Type"][transaction.tt]],
        "Time_of_Transaction": [transaction.tm],
        "Device_Used": [CATEGORICAL_MAPPINGS["Device_Used"][transaction.du]],
        "Location": [CATEGORICAL_MAPPINGS["Location"][transaction.lc]],
        "Payment_Method": [CATEGORICAL_MAPPINGS["Payment_Method"][transaction.pm]],
        "User_ID": [transaction.ui],
        "Previous_Fraudulent_Transactions": [transaction.pf],
        "Account_Age": [transaction.aa],
        "Number_of_Transactions_Last_24H": [transaction.nt]
    })

    df_encoded = pd.get_dummies(df, columns=["Transaction_Type", "Device_Used", "Location", "Payment_Method"])
    df_encoded = df_encoded.reindex(columns=TRAINING_COLS, fill_value=0)

    return SCALER.transform(df_encoded)

# ======================================
# FastAPI setup (lifespan)
# ======================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_categorical_mappings()
    load_models_and_scaler()
    yield
    MODELS.clear()

app = FastAPI(lifespan=lifespan)

# ======================================
# Prediction Endpoint
# ======================================
@app.post("/predict")
async def predict_transaction(request: PredictionRequest) -> Dict[str, Any]:
    """Predict transaction fraud probability using selected models."""
    transaction_data = preprocess_transaction(request.transaction)
    predictions = {}

    for name in request.models:
        if name not in MODELS:
            logger.warning(f"Model '{name}' is valid but not loaded.")
            continue

        model = MODELS[name]
        start_time = time.time()

        if name == "Neural Network":
            prob = model.predict(transaction_data, verbose=0)[0, 0]
            threshold = 0.425
        else:
            prob = model.predict_proba(transaction_data)[:, 1][0]
            threshold = 0.5

        prediction = int(prob >= threshold)
        inference_time = (time.time() - start_time) * 1000  # ms

        predictions[name] = {
            "probability": float(prob),
            "prediction": prediction,
            "inference_time_ms": inference_time
        }

    if not predictions:
        raise HTTPException(status_code=400, detail="No valid or loaded models selected.")

        # Construct a single log message with color-coded probabilities
    log_parts = []
    for name, pred in predictions.items():
        prob = pred["probability"]
        #color = RED_BG if prob >= 0.5 else GREEN_BG
        #log_parts.append(f"{color}{name}: {prob:.3f}{RESET}")
        status = "FRAUD" if prob >= 0.5 else "OK"
        log_parts.append(f"{name}: {prob:.3f} | STATUS={status}")
    log_message = ", ".join(log_parts)

    # Log the result
    logger.info(log_message)
    
    return predictions

# ======================================
# WebSocket streaming logs
# ======================================
from starlette.websockets import WebSocketDisconnect

@app.websocket("/ws")
async def websocket_logs(websocket: WebSocket):
    """WebSocket endpoint for streaming all logs in batch."""
    await websocket.accept()
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    try:
        while True:
            logs_to_send = []

            while not log_queue.empty():
                log_record = log_queue.get()
                formatted_log = log_formatter.format(log_record)
                logs_to_send.append(formatted_log)

            if logs_to_send:
                await websocket.send_text("\n".join(logs_to_send))

            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.error(f"Unexpected WebSocket error: {str(e)}")
