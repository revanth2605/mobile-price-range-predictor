import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

FILE_PATH = "data/phones_2025.csv"

print("📌 Loading dataset...")
df = pd.read_csv(FILE_PATH)

# ---- Fix: Clean price column ----
df['price'] = df['price'].astype(str)\
    .str.replace("₹", "", regex=False)\
    .str.replace(",", "", regex=False)\
    .str.extract("(\d+)")\
    .astype(float)

# Show first 20 rows
print("\n🔍 First 20 rows:")
print(df.head(20))

# Dataset summary
print("\n📊 Dataset summary:")
print(df.info())

# Missing values
print("\n⚠ Missing values:")
print(df.isnull().sum())

# Statistical summary
print("\n📈 Statistical summary:")
print(df.describe())

# Price distribution
print("\n💰 Price distribution summary:")
print(df['price'].describe())

# Histogram (optional visualization)
try:
    plt.figure(figsize=(8, 5))
    sns.histplot(df['price'], kde=True)
    plt.title("Price Distribution")
    plt.xlabel("Price (INR)")
    plt.ylabel("Frequency")
    plt.show()
except:
    print("\n(⚠ Plot skipped: no GUI available)")
