from pathlib import Path

import pandas as pd


FEATURE_PATH = Path("data/processed/image_features.csv")


df = pd.read_csv(FEATURE_PATH)

feature_columns = [
    column
    for column in df.columns
    if column.startswith("feature_")
]

print("===== FEATURE DATASET =====")
print(f"Rows: {len(df)}")
print(f"Feature columns: {len(feature_columns)}")

print("\n===== EXPECTED SHAPE =====")
print("Expected rows: 9980")
print("Expected features: 60")

print("\n===== ACTUAL SHAPE =====")
print(df[feature_columns].shape)

print("\n===== SPLIT COUNTS =====")
print(df["split"].value_counts())

print("\n===== CLASS COUNTS =====")
print(df["class"].value_counts())

print("\n===== MISSING VALUES =====")
print(
    df[feature_columns]
    .isna()
    .sum()
    .sum()
)

print("\n===== FEATURE DATA TYPES =====")
print(
    df[feature_columns]
    .dtypes
    .value_counts()
)

print("\n===== FEATURE SUMMARY =====")
print(
    df[feature_columns]
    .describe()
    .T
    .head(10)
)