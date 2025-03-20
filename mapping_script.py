import pandas as pd
import os

# Load dataset
data_path = os.path.join("fraud-detection-dataset", "Fraud Detection Dataset.csv")
df = pd.read_csv(data_path)

# Define categorical columns
categorical_cols = ["Transaction_Type", "Device_Used", "Location", "Payment_Method"]

# Extract unique values and create mappings
CATEGORICAL_MAPPINGS = {}
for col in categorical_cols:
    # Include NaN as 'Unknown' or similar
    unique_values = df[col].dropna().unique().tolist()
    mapping = {i: val for i, val in enumerate(unique_values)}
    # Add 'Unknown' for NaN
    if df[col].isna().any():
        mapping[max(mapping.keys()) + 1] = "Unknown"
    CATEGORICAL_MAPPINGS[col] = mapping

# Save mappings to CSV
mapping_data = []
for col, mapping in CATEGORICAL_MAPPINGS.items():
    for code, value in mapping.items():
        mapping_data.append({"Column": col, "Code": code, "Value": value})

mapping_df = pd.DataFrame(mapping_data)
mapping_df.to_csv("categorical_mappings.csv", index=False)
print("Saved mappings to categorical_mappings.csv")

# Print mappings for verification
for col, mapping in CATEGORICAL_MAPPINGS.items():
    print(f"{col}: {mapping}")