import streamlit as st
import requests
import time
import random
import pandas as pd
import os
import uuid
from datetime import time as dt_time

from models import AVAILABLE_MODELS

# ----------------------------------------
# 🎯 CONFIGURATION AND CONSTANTS
# ----------------------------------------

API_URL = "http://127.0.0.1:8000/predict"
MAPPING_PATH = "categorical_mappings.csv"

# Custom thresholds per model (based on evaluation notebook)
FRAUD_THRESHOLDS = {
    "Neural Network": 0.425,
    # Default for others
}

# Store unique session IDs
session_dict = {}

# ----------------------------------------
# 📚 LOAD CATEGORICAL MAPPINGS
# ----------------------------------------

if not os.path.exists(MAPPING_PATH):
    st.error(f"Categorical mappings not found: {MAPPING_PATH}")
    st.stop()

mapping_df = pd.read_csv(MAPPING_PATH)
CATEGORICAL_MAPPINGS = {}
for col in mapping_df["Column"].unique():
    col_mapping = mapping_df[mapping_df["Column"] == col]
    CATEGORICAL_MAPPINGS[col] = {
        row["Value"]: row["Code"] for _, row in col_mapping.iterrows()
    }

    
# ----------------------------------------
# 🚀 HELPER FUNCTIONS
# ----------------------------------------

def generate_session_id():
    """Generate a unique session ID."""
    return str(uuid.uuid4())[:8]

def send_prediction_request(transaction, selected_models):
    """Send transaction data and selected models to FastAPI."""
    payload = {"transaction": transaction, "models": selected_models}
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return None

def display_predictions(predictions):
    """Display predictions in styled cards."""
    cols = st.columns(3)
    for i, (model, result) in enumerate(predictions.items()):
        prob = result["probability"]
        threshold = FRAUD_THRESHOLDS.get(model, 0.5)
        fraud = prob >= threshold
        bg_color = "#772e2c" if fraud else "#39511f"
        text_color = "#ffffff"
        with cols[i % 3]:
            st.markdown(
                f"""
                <div style="
                    padding:10px; margin:5px; border-radius:5px;
                    background-color:{bg_color}; color:{text_color}; text-align:center;">
                    <b>{model}</b><br>
                    Fraud Probability: {prob:.3f}<br>
                    Time: {result["inference_time_ms"]:.1f} ms
                </div>
                """, unsafe_allow_html=True)

def random_transaction_device():
    """Randomly select ATM or POS device name."""
    device_type = random.choice(["ATM", "POS"])
    return f"{device_type} #"

# ----------------------------------------
# 🖱️ MANUAL TEST MODE
# ----------------------------------------

def manual_test_mode(selected_models):
    """Manual mode to input transaction details."""
    st.subheader("🔍 Manual Transaction Check")

    col1, col2, col3 = st.columns(3)
    with col1:
        ta = st.slider("💰 Transaction Amount", 0, 100000, 500)
        du = st.selectbox("📱 Device Used", list(CATEGORICAL_MAPPINGS["Device_Used"]))
        aa = st.slider("📅 Account Age", 0, 200, 30)
        nt = st.slider("📈 Transactions Last 24H", 0, 50, 2)

    with col2:
        ui = st.slider("👤 User ID", 1, 4000, 1234)
        tt = st.selectbox("🏦 Transaction Type", list(CATEGORICAL_MAPPINGS["Transaction_Type"]))
        lc = st.selectbox("🌍 Location", list(CATEGORICAL_MAPPINGS["Location"]))

    with col3:
        tm = st.time_input("⏰ Time of Transaction", dt_time(12,0))
        tm_float = tm.hour + tm.minute/60.0
        pm = st.selectbox("💳 Payment Method", list(CATEGORICAL_MAPPINGS["Payment_Method"]))
        pf = st.slider("🕒 Previous Fraudulent Trans.", 0, 10, 0)

    transaction = {
        "ta": float(ta),
        "tt": CATEGORICAL_MAPPINGS["Transaction_Type"][tt],
        "tm": tm_float,
        "du": CATEGORICAL_MAPPINGS["Device_Used"][du],
        "lc": CATEGORICAL_MAPPINGS["Location"][lc],
        "pm": CATEGORICAL_MAPPINGS["Payment_Method"][pm],
        "ui": ui, "pf": pf, "aa": aa, "nt": nt
    }

    predictions = send_prediction_request(transaction, selected_models)
    if predictions:
        display_predictions(predictions)

# ----------------------------------------
# 🎬 SIMULATION MODE
# ----------------------------------------

def simulation_mode(selected_models):
    """Minimalist simulation mode."""
    st.subheader(f"🔄 {random_transaction_device()}{generate_session_id()} Live ")
    delay = st.slider("Delay (ms)", 50, 1000, (50,50), 50)
    placeholder = st.empty()

    while True:
        transaction = {
            "ta": round(random.uniform(1, 100000),2),
            "tt": random.choice(list(CATEGORICAL_MAPPINGS["Transaction_Type"].values())),
            "tm": round(random.uniform(0,24),2),
            "du": random.choice(list(CATEGORICAL_MAPPINGS["Device_Used"].values())),
            "lc": random.choice(list(CATEGORICAL_MAPPINGS["Location"].values())),
            "pm": random.choice(list(CATEGORICAL_MAPPINGS["Payment_Method"].values())),
            "ui": random.randint(1,4000),
            "pf": random.randint(0,5),
            "aa": random.randint(0,200),
            "nt": random.randint(0,50)
        }
        predictions = send_prediction_request(transaction, selected_models)

        with placeholder.container():
            display_predictions(predictions)

        time.sleep(random.uniform(*delay)/1000)

# ----------------------------------------
# 🎯 MAIN APPLICATION
# ----------------------------------------

def main():
    st.set_page_config("💳 Fraud Detection", layout="wide")
    st.title("💳 Fraud Detection Dashboard")

    # Sidebar for model selection
    st.sidebar.header("🔧 Model Selection")
    selected_models = st.sidebar.multiselect(
        "Choose models for inference:",
        AVAILABLE_MODELS,
        default=["Gradient Boosting", "HistGradientBoosting"] #"Neural Network",
    )

    mode = st.sidebar.radio("Select Mode:", ["Simulation", "Manual"])

    if mode == "Manual":
        manual_test_mode(selected_models)
    else:
        simulation_mode(selected_models)

if __name__ == "__main__":
    main()
