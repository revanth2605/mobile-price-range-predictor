import pandas as pd
import re
import json
import os
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# File Paths
# -------------------------------
INPUT_FILE = "data/phones_2025.csv"
OUTPUT_FILE = "data/processed.csv"
META_FILE = "models/metadata.json"

print("📌 Loading dataset...")
df = pd.read_csv(INPUT_FILE)

# Remove duplicate index column if present
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# -------------------------------
# Extract Brand
# -------------------------------
df["brand"] = df["mobile_name"].str.split().str[0]

# -------------------------------
# Clean Price → Integer
# -------------------------------
def clean_price(x):
    x = str(x)
    x = re.sub(r"[^\d]", "", x)
    return int(x) if x.isdigit() else None

df["price"] = df["price"].apply(clean_price)

print("\n✔ Price cleaned successfully.")

# -------------------------------
# Extract RAM & Storage
# -------------------------------
def extract_first_number(text):
    m = re.search(r"(\d+)", str(text))
    return int(m.group(1)) if m else None

df["ram"] = df["ram_and_storage"].apply(extract_first_number)

def extract_last_number(text):
    m = re.findall(r"(\d+)", str(text))
    return int(m[-1]) if m else None

df["storage"] = df["ram_and_storage"].apply(extract_last_number)

# -------------------------------
# Extract Other Features
# -------------------------------
def extract_battery(text):
    m = re.search(r"(\d+)\s*mAh", str(text))
    return int(m.group(1)) if m else None

df["battery"] = df["battery_and_charging_speed"].apply(extract_battery)

def extract_camera(text):
    m = re.search(r"(\d+)\s*MP", str(text))
    return int(m.group(1)) if m else None

df["camera_mp"] = df["rear_camera"].apply(extract_camera)

def extract_screen_size(text):
    m = re.search(r"(\d+(\.\d+)?)", str(text))
    return float(m.group(1)) if m else None

df["screen_size"] = df["display"].apply(extract_screen_size)

df["is_5g"] = df["5G|NFC|Fingerprint"].str.contains("5G", case=False, na=False).astype(int)
df["has_fingerprint"] = df["5G|NFC|Fingerprint"].str.contains("fingerprint", case=False, na=False).astype(int)
df["has_nfc"] = df["5G|NFC|Fingerprint"].str.contains("nfc", case=False, na=False).astype(int)

# -------------------------------
# Handle Missing Values
# -------------------------------
numeric_cols = ["ram", "storage", "battery", "camera_mp", "screen_size", "price"]
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

df["brand"] = df["brand"].fillna(df["brand"].mode()[0])

# -------------------------------
# Encode Brand
# -------------------------------
label_encoder_brand = LabelEncoder()
df["brand"] = label_encoder_brand.fit_transform(df["brand"])

# -------------------------------
# ⚡ Dynamic Price Bucketing
# -------------------------------
q1 = df["price"].quantile(0.25)
q2 = df["price"].quantile(0.50)
q3 = df["price"].quantile(0.75)

def price_to_range(price):
    if price < 10000:
        return "Budget"
    elif price < 20000:
        return "Mid-Range"
    elif price < 40000:
        return "Premium"
    else:
        return "Ultra Premium"

df["price_range_text"] = df["price"].apply(price_to_range)

# Create INR formatting for UI
price_ranges_inr = {
    "Budget": f"₹{int(df['price'].min())} - ₹{int(q1)}",
    "Mid-Range": f"₹{int(q1+1)} - ₹{int(q2)}",
    "Premium": f"₹{int(q2+1)} - ₹{int(q3)}",
    "Ultra Premium": f"₹{int(q3+1)} - ₹{int(df['price'].max())}"
}

print("\n📊 Dynamic Price Buckets:")
print(price_ranges_inr)

# -------------------------------
# Encode Target Label
# -------------------------------
label_encoder_y = LabelEncoder()
df["price_range"] = label_encoder_y.fit_transform(df["price_range_text"])

# -------------------------------
# Final Columns
# -------------------------------
final_cols = [
    "brand", "ram", "storage", "battery", "camera_mp", "screen_size",
    "is_5g", "has_fingerprint", "has_nfc", "price_range"
]

processed_df = df[final_cols]

# -------------------------------
# Save Processed Dataset
# -------------------------------
os.makedirs("models", exist_ok=True)
processed_df.to_csv(OUTPUT_FILE, index=False)

print("\n🎉 Preprocessing complete!")
print(f"📁 Saved → {OUTPUT_FILE}")

# -------------------------------
# SAVE METADATA JSON
# -------------------------------
metadata = {
    "brand_encoder_classes": list(label_encoder_brand.classes_),
    "labels": {i: v for i, v in enumerate(label_encoder_y.classes_)},
    "price_ranges_inr": price_ranges_inr
}

with open(META_FILE, "w") as f:
    json.dump(metadata, f, indent=4)

print("\n💾 Metadata stored in:", META_FILE)
