from pathlib import Path
import pandas as pd

METADATA_PATH = Path("data/processed/image_metadata.csv")

df = pd.read_csv(METADATA_PATH)

print("===== CLASS-WISE IMAGE COUNT =====")
print(df["class"].value_counts())

print("\n===== DIMENSIONS BY CLASS =====")
print(
    df.groupby("class")[["width", "height"]]
    .agg(["mean", "median", "min", "max"])
    .round(2)
)

print("\n===== FILE SIZE BY CLASS (KB) =====")
print(
    df.groupby("class")["file_size_kb"]
    .agg(["mean", "median", "min", "max"])
    .round(2)
)

print("\n===== IMAGE MODES BY CLASS =====")
print(
    pd.crosstab(
        df["class"],
        df["mode"]
    )
)

print("\n===== IMAGE FORMATS BY CLASS =====")
print(
    pd.crosstab(
        df["class"],
        df["format"]
    )
)

df["aspect_ratio"] = df["width"] / df["height"]

print("\n===== ASPECT RATIO BY CLASS =====")
print(
    df.groupby("class")["aspect_ratio"]
    .agg(["mean", "median", "min", "max"])
    .round(3)
)