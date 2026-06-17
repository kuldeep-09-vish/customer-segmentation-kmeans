import pandas as pd
import numpy as np
import random

np.random.seed(42)

n = 10000

# -------------------------
# 1. Basic Columns
# -------------------------
customer_id = np.arange(1, n+1)

age = np.random.randint(18, 70, n)

income = np.random.normal(50000, 15000, n).astype(int)
income = np.abs(income)

gender = np.random.choice(["Male", "Female"], n)

city = np.random.choice(["Indore", "Bhopal", "Delhi", "Mumbai", "Pune"], n)

purchase_frequency = np.random.randint(1, 20, n)

last_purchase_days = np.random.randint(1, 365, n)

spending_score = np.random.randint(1, 100, n)

product_category = np.random.choice(
    ["Electronics", "Fashion", "Grocery", "Sports", "Beauty"], n
)

membership_status = np.random.choice(["Basic", "Silver", "Gold"], n)

# -------------------------
# 2. Create DataFrame
# -------------------------
df = pd.DataFrame({
    "CustomerID": customer_id,
    "Age": age,
    "Income": income,
    "Gender": gender,
    "City": city,
    "PurchaseFrequency": purchase_frequency,
    "LastPurchaseDays": last_purchase_days,
    "SpendingScore": spending_score,
    "ProductCategory": product_category,
    "MembershipStatus": membership_status
})

# -------------------------
# 3. Introduce Missing Values
# -------------------------
for col in ["Age", "Income", "SpendingScore"]:
    df.loc[df.sample(frac=0.05).index, col] = np.nan

# -------------------------
# 4. Introduce Duplicates
# -------------------------
df = pd.concat([df, df.sample(500)], ignore_index=True)

# -------------------------
# 5. Introduce Outliers
# -------------------------
outlier_indices = np.random.choice(df.index, 50)

df.loc[outlier_indices, "Income"] = df["Income"] * 10
df.loc[outlier_indices, "Age"] = np.random.randint(5, 120, 50)

# -------------------------
# 6. Shuffle dataset
# -------------------------
df = df.sample(frac=1).reset_index(drop=True)

# -------------------------
# 7. Save CSV
# -------------------------
df.to_csv("customer_dataset.csv", index=False)

print("Dataset created successfully!")
print(df.head())
print("Shape:", df.shape)