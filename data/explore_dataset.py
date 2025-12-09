import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# File path
FILE_PATH = "phones_2025.csv"


# Load dataset
print("📌 Loading dataset...")
df = pd.read_csv(FILE_PATH)

# Show first 20 rows
print("\n🔍 First 20 rows:")
print(df.head(20))

# Show dataset info
print("\n📊 Dataset summary (columns, types):")
print(df.info())

# Show missing values count
print("\n⚠ Missing values per column:")
print(df.isnull().sum())

# Show basic dataset description
print("\n📈 Statistical summary (numerical columns):")
print(df.describe())

# Value distribution of the price column
print("\n💰 Price value distribution:")
print(df['price'].describe())

# Optional: visualize price distribution (not mandatory but useful)
try:
    plt.figure(figsize=(8,5))
    sns.histplot(df['price'], kde=True)
    plt.title("Price Distribution")
    plt.xlabel("Price (₹)")
    plt.ylabel("Count")
    plt.show()
except:
    print("\n(⚠ Plot skipped — if running on non-GUI terminal.)")
