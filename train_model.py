import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree
import joblib
import os

# -----------------------------
# File Paths
# -----------------------------
DATA_FILE = "data/processed.csv"
MODEL_FILE = "models/price_model.pkl"
META_FILE = "models/metadata.json"

# Ensure model folder exists
os.makedirs("models", exist_ok=True)

print("📌 Loading processed dataset...")
df = pd.read_csv(DATA_FILE)

# -----------------------------
# Define features and target
# -----------------------------
feature_columns = [
    "brand", "ram", "storage", "battery", "camera_mp",
    "screen_size", "is_5g", "has_fingerprint", "has_nfc"
]

X = df[feature_columns]
y = df["price_range"]

# -----------------------------
# Train-test split
# -----------------------------
print("\n📂 Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# -----------------------------
# Train Decision Tree Model
# -----------------------------
print("\n🌲 Training Decision Tree Classifier...")
model = DecisionTreeClassifier(
    criterion="entropy", max_depth=10, random_state=42
)
model.fit(X_train, y_train)

# -----------------------------
# Evaluate Model
# -----------------------------
print("\n📊 Evaluating model...")
y_pred = model.predict(X_test)

print("\n🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n🧩 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# Save Model & Metadata
# -----------------------------
joblib.dump(model, MODEL_FILE)

metadata = {
    "features": feature_columns,
    "labels": {
        0: "Budget",
        1: "Mid-Range",
        2: "Premium",
        3: "Ultra Premium"
    }
}

with open(META_FILE, "w") as f:
    json.dump(metadata, f, indent=4)

print("\n💾 Model saved to:", MODEL_FILE)
print("📝 Metadata saved to:", META_FILE)

# Optional: display feature importance
print("\n🔥 Feature Importance:")
for feature, score in zip(feature_columns, model.feature_importances_):
    print(f"{feature}: {round(score, 4)}")
