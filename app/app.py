import os
import json
import re

from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# ----------------------------------
# Paths
# ----------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_FILE = os.path.join(BASE_DIR, "data", "phones_2025.csv")
PROCESSED_DATA_FILE = os.path.join(BASE_DIR, "data", "processed.csv")
MODEL_FILE = os.path.join(BASE_DIR, "models", "best_model.pkl")
META_FILE = os.path.join(BASE_DIR, "models", "metadata.json")

# ----------------------------------
# Flask App
# ----------------------------------
app = Flask(__name__)

# ----------------------------------
# Load Model & Metadata
# ----------------------------------
print("📌 Loading model and metadata...")
model = joblib.load(MODEL_FILE)

with open(META_FILE, "r") as f:
    metadata = json.load(f)

# label index → readable label
label_mapping = {int(k): v for k, v in metadata["labels"].items()}

# INR price mapping
price_range_inr = metadata.get("price_ranges_inr", {})

# ----------------------------------
# Load Data & Prepare Dropdown Options
# ----------------------------------
print("📌 Preparing dropdown options...")

df_raw = pd.read_csv(RAW_DATA_FILE)
df_raw = df_raw.loc[:, ~df_raw.columns.str.contains('^Unnamed')]

# Extract brand same as preprocessing
df_raw["brand"] = df_raw["mobile_name"].str.split().str[0]
df_raw["brand"] = df_raw["brand"].fillna(df_raw["brand"].mode()[0])

# Fit label encoder same as preprocess
brand_encoder = LabelEncoder()
brand_encoder.fit(df_raw["brand"])

brand_options = sorted(df_raw["brand"].unique().tolist())

# Load processed dataset
df_proc = pd.read_csv(PROCESSED_DATA_FILE)

# ---- NEW: Realistic Dropdown Filter ----
def clean_dropdown(col, min_val=None, max_val=None):
    values = sorted(df_proc[col].dropna().unique().tolist())
    if min_val is not None:
        values = [v for v in values if v >= min_val]
    if max_val is not None:
        values = [v for v in values if v <= max_val]
    return values

# Modern phone spec ranges applied
ram_options = clean_dropdown("ram", min_val=4, max_val=24)  
storage_options = clean_dropdown("storage", min_val=64, max_val=1024)
battery_options = clean_dropdown("battery", min_val=3000, max_val=7000)
camera_options = clean_dropdown("camera_mp", min_val=8, max_val=200)
size_options = clean_dropdown("screen_size", min_val=5.5, max_val=7.5)

bool_options = [("1", "Yes"), ("0", "No")]

print("✅ Dropdown data ready.")

# ----------------------------------
# Helper: Build Feature Vector
# ----------------------------------
def build_feature_vector(form_data):

    values = {
        "brand": int(brand_encoder.transform([form_data.get("brand")])[0]),
        "ram": float(form_data.get("ram")),
        "storage": float(form_data.get("storage")),
        "battery": float(form_data.get("battery")),
        "camera_mp": float(form_data.get("camera_mp")),
        "screen_size": float(form_data.get("screen_size")),
        "is_5g": int(form_data.get("is_5g")),
        "has_fingerprint": int(form_data.get("has_fingerprint")),
        "has_nfc": int(form_data.get("has_nfc")),
    }

    # Optional realism logic:
    if values["ram"] <= 4:
        values["is_5g"] = 0
        values["has_nfc"] = 0

    return [values[col] for col in metadata["features"]]

# ----------------------------------
# Routes
# ----------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        brand_options=brand_options,
        ram_options=ram_options,
        storage_options=storage_options,
        battery_options=battery_options,
        camera_options=camera_options,
        size_options=size_options,
        bool_options=bool_options,
        prediction_label=None
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        feature_vector = build_feature_vector(request.form)
        prediction_index = int(model.predict([feature_vector])[0])

        readable_label = label_mapping.get(prediction_index, "Unknown")
        readable_price = price_range_inr.get(readable_label, "Unavailable")

        # Debug output
        print("\n🔍 DEBUG CHECK")
        print("➡ Predicted:", readable_label)
        print("➡ Available price keys:", list(price_range_inr.keys()))
        print("➡ Price matched:", readable_price)

        # Output text
        prediction_label = f"{readable_label} ({readable_price})"

    except Exception as e:
        print("❌ Error during prediction:", e)
        prediction_label = f"❌ Error: {str(e)}"

    return render_template(
        "index.html",
        brand_options=brand_options,
        ram_options=ram_options,
        storage_options=storage_options,
        battery_options=battery_options,
        camera_options=camera_options,
        size_options=size_options,
        bool_options=bool_options,
        prediction_label=prediction_label
    )

# ----------------------------------
# Run
# ----------------------------------
if __name__ == "__main__":
    app.run(debug=True)
