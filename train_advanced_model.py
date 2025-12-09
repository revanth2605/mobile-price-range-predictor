import pandas as pd
import json
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Try importing XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    print("⚠ XGBoost not installed — skipping it.")
    HAS_XGB = False

# -----------------------------
# Load Data
# -----------------------------
DATA_FILE = "data/processed.csv"
MODEL_FILE = "models/best_model.pkl"
META_FILE = "models/metadata.json"

df = pd.read_csv(DATA_FILE)

feature_cols = [
    "brand", "ram", "storage", "battery", "camera_mp",
    "screen_size", "is_5g", "has_fingerprint", "has_nfc"
]

X = df[feature_cols]
y = df["price_range"]

# Train/Test Split
print("\n📌 Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# -----------------------------
# Model Candidates + Parameter Search
# -----------------------------
models = {
    "DecisionTree": (DecisionTreeClassifier(), {
        "model__max_depth": [5, 10, 15],
        "model__criterion": ["entropy", "gini"]
    }),
    "RandomForest": (RandomForestClassifier(), {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [10, 20, None]
    })
}

if HAS_XGB:
    models["XGBoost"] = (XGBClassifier(eval_metric="mlogloss"), {
        "model__n_estimators": [50, 100, 200],
        "model__learning_rate": [0.05, 0.1, 0.2]
    })

best_model = None
best_score = -1
best_name = ""

print("\n🔍 Running Grid Search on models...\n")

for name, (model, params) in models.items():
    print(f"⚙ Training: {name}")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    grid = GridSearchCV(pipeline, params, cv=3, scoring="f1_macro", n_jobs=-1)
    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)
    score = f1_score(y_test, y_pred, average="macro")

    print(f"➡ {name} F1 Score: {score:.4f}\n")

    if score > best_score:
        best_score = score
        best_model = grid
        best_name = name

# -----------------------------
# Final Evaluation
# -----------------------------
print("\n🏆 Best Model Selected:", best_name)
y_pred_final = best_model.predict(X_test)

print("\n🎯 Final Accuracy:", accuracy_score(y_test, y_pred_final))
print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred_final))
print("\n🧩 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_final))

# Save Model and Metadata
os.makedirs("models", exist_ok=True)

joblib.dump(best_model, MODEL_FILE)

metadata = {
    "best_model": best_name,
    "features": feature_cols,
    "labels": {
        0: "Budget",
        1: "Mid-Range",
        2: "Premium",
        3: "Ultra-Flagship"
    }
}

with open(META_FILE, "w") as f:
    json.dump(metadata, f, indent=4)

print(f"\n💾 Saved best model as: {MODEL_FILE}")
print(f"📝 Metadata stored at: {META_FILE}")

