import pandas as pd
import requests
import json
import os

# Constants
API_URL = "http://localhost:8000/predict"
DATA_PATH = os.path.join("fraud-detection-dataset", "Fraud Detection Dataset.csv")
MAPPING_PATH = "categorical_mappings.csv"

# Global variable for categorical mappings
CATEGORICAL_MAPPINGS = {}


def load_categorical_mappings(file_path: str) -> dict:
    """Load categorical mappings from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Mapping file not found at: {file_path}")

    mapping_df = pd.read_csv(file_path)
    mappings = {}
    for column in mapping_df["Column"].unique():
        col_mapping = mapping_df[mapping_df["Column"] == column]
        mappings[column] = {row["Value"]: row["Code"] for _, row in col_mapping.iterrows()}
    print("Loaded categorical mappings:", mappings)
    return mappings


def sample_to_json(row: pd.Series) -> dict:
    """Convert a DataFrame row to a JSON-compatible dictionary."""
    return {
        "ta": float(row["Transaction_Amount"]) if pd.notna(row["Transaction_Amount"]) else 0.0,
        "tt": int(CATEGORICAL_MAPPINGS["Transaction_Type"].get(row["Transaction_Type"], 4)),
        "tm": float(row["Time_of_Transaction"]) if pd.notna(row["Time_of_Transaction"]) else 0.0,
        "du": int(CATEGORICAL_MAPPINGS["Device_Used"].get(row["Device_Used"], 4)),
        "lc": int(CATEGORICAL_MAPPINGS["Location"].get(row["Location"] if pd.notna(row["Location"]) else "Unknown", 8)),
        "pm": int(CATEGORICAL_MAPPINGS["Payment_Method"].get(row["Payment_Method"], 5)),
        "ui": int(row["User_ID"]) if pd.notna(row["User_ID"]) else 1,
        "pf": int(row["Previous_Fraudulent_Transactions"]) if pd.notna(row["Previous_Fraudulent_Transactions"]) else 0,
        "aa": int(row["Account_Age"]) if pd.notna(row["Account_Age"]) else 0,
        "nt": int(row["Number_of_Transactions_Last_24H"]) if pd.notna(row["Number_of_Transactions_Last_24H"]) else 0
    }


def send_request(sample_index: int, transaction: dict, actual_label: int) -> None:
    """Send a POST request to the API and print the response."""
    try:
        print(f"\nSending Sample {sample_index}: {json.dumps(transaction)}")
        response = requests.post(API_URL, json=transaction)
        response.raise_for_status()
        predictions = response.json()
        print(f"Sample {sample_index}:")
        print(f"Actual Fraudulent: {actual_label}")
        print("Input JSON:", json.dumps(transaction, indent=2))
        print("Predictions:", json.dumps(predictions, indent=2))
    except KeyError as e:
        print(f"\nSample {sample_index} KeyError: {e}")
    except requests.exceptions.RequestException as e:
        print(f"\nSample {sample_index} Error: {e}")
        print("Response Text:", e.response.text if e.response else "No response")


def main():
    """Main function to run the test script."""
    # Load data
    df = pd.read_csv(DATA_PATH)

    # Load categorical mappings
    global CATEGORICAL_MAPPINGS
    CATEGORICAL_MAPPINGS = load_categorical_mappings(MAPPING_PATH)

    # Select test samples
    fraud_samples = df[df["Fraudulent"] == 1].head(5)
    non_fraud_samples = df[df["Fraudulent"] == 0].head(5)
    test_samples = pd.concat([fraud_samples, non_fraud_samples])

    # Process each sample
    for index, row in test_samples.iterrows():
        transaction = sample_to_json(row)
        actual_label = row["Fraudulent"]
        send_request(index, transaction, actual_label)


if __name__ == "__main__":
    main()