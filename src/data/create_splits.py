from pathlib import Path
from collections import defaultdict
import hashlib

import pandas as pd
from sklearn.model_selection import train_test_split


DATASET_DIR = Path("data/raw/mediaeval/ITW-SM")
OUTPUT_PATH = Path("data/splits/image_splits.csv")

RANDOM_STATE = 42

CLASS_MAP = {
    "0_real": 0,
    "1_fake": 1,
}


def calculate_sha256(file_path):
    """Calculate the SHA-256 hash of a file."""
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


# ---------------------------------------------------------
# 1. Collect images and calculate hashes
# ---------------------------------------------------------

records = []

for folder_name, label in CLASS_MAP.items():

    folder = DATASET_DIR / folder_name

    for image_path in sorted(folder.iterdir()):

        if not image_path.is_file():
            continue

        file_hash = calculate_sha256(image_path)

        records.append(
            {
                "path": str(image_path),
                "label": label,
                "class": "synthetic" if label == 1 else "real",
                "sha256": file_hash,
            }
        )


df = pd.DataFrame(records)

print("===== ORIGINAL DATASET =====")
print(f"Total files: {len(df)}")

print("\nClass distribution:")
print(df["class"].value_counts())


# ---------------------------------------------------------
# 2. Remove exact duplicates logically
# ---------------------------------------------------------

duplicate_mask = df.duplicated(
    subset=["sha256"],
    keep="first",
)

duplicates = df[duplicate_mask].copy()

df_unique = df[~duplicate_mask].copy()

print("\n===== DUPLICATE REMOVAL =====")
print(f"Duplicate files excluded: {len(duplicates)}")
print(f"Unique images: {len(df_unique)}")


print("\nUnique class distribution:")
print(df_unique["class"].value_counts())


# ---------------------------------------------------------
# 3. Create train/test split
# ---------------------------------------------------------

train_df, temp_df = train_test_split(
    df_unique,
    test_size=0.30,
    stratify=df_unique["label"],
    random_state=RANDOM_STATE,
)


# ---------------------------------------------------------
# 4. Split remaining 30% into validation/test
# ---------------------------------------------------------

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=RANDOM_STATE,
)


# ---------------------------------------------------------
# 5. Assign split labels
# ---------------------------------------------------------

train_df = train_df.copy()
validation_df = validation_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"
validation_df["split"] = "validation"
test_df["split"] = "test"


# ---------------------------------------------------------
# 6. Combine all splits
# ---------------------------------------------------------

final_df = pd.concat(
    [
        train_df,
        validation_df,
        test_df,
    ],
    ignore_index=True,
)


# ---------------------------------------------------------
# 7. Sort for reproducibility
# ---------------------------------------------------------

final_df = final_df.sort_values(
    by=["split", "class", "path"]
).reset_index(drop=True)


# ---------------------------------------------------------
# 8. Save manifest
# ---------------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

final_df.to_csv(
    OUTPUT_PATH,
    index=False,
)


# ---------------------------------------------------------
# 9. Print final statistics
# ---------------------------------------------------------

print("\n===== FINAL SPLIT =====")

print(
    final_df["split"].value_counts()
)

print("\n===== SPLIT × CLASS =====")

print(
    pd.crosstab(
        final_df["split"],
        final_df["class"],
    )
)

print("\n===== CHECK FOR DUPLICATES ACROSS SPLITS =====")

split_hash_counts = (
    final_df.groupby("sha256")["split"]
    .nunique()
)

cross_split_duplicates = (
    split_hash_counts > 1
).sum()

print(
    f"Hashes appearing in multiple splits: "
    f"{cross_split_duplicates}"
)

print(f"\nSaved manifest to: {OUTPUT_PATH}")