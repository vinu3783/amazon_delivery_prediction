"""Check data for issues before training"""

import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data/processed/feature_engineered_data.csv')

print("Data Shape:", df.shape)
print("\nColumn Types:")
print(df.dtypes.value_counts())

print("\nMissing Values:")
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    print(missing)
else:
    print("No missing values")

print("\nNumeric Columns:")
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(f"Count: {len(numeric_cols)}")
print(list(numeric_cols))

print("\nTarget Variable (Delivery_Time):")
if 'Delivery_Time' in df.columns:
    print(f"  Count: {df['Delivery_Time'].count()}")
    print(f"  Missing: {df['Delivery_Time'].isnull().sum()}")
    print(f"  Mean: {df['Delivery_Time'].mean():.2f}")
    print(f"  Min: {df['Delivery_Time'].min():.2f}")
    print(f"  Max: {df['Delivery_Time'].max():.2f}")
else:
    print("  NOT FOUND!")

print("\nColumns with >50% missing:")
high_missing = df.columns[df.isnull().sum() / len(df) > 0.5]
if len(high_missing) > 0:
    print(list(high_missing))
else:
    print("None")