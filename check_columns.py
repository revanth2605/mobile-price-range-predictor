import pandas as pd

df = pd.read_csv("data/phones_2025.csv")

print("\n📌 Available Columns in the Dataset:\n")
for col in df.columns:
    print("-", col)
