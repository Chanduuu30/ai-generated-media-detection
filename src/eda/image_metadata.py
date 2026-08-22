from pathlib import Path
from PIL import Image
import pandas as pd

DATASET_DIR = Path("data/raw/mediaeval/ITW-SM")

CLASS_MAP = {
    "0_real": 0,
    "1_fake": 1,
}

records = []

for folder_name, label in CLASS_MAP.items():
    folder = DATASET_DIR / folder_name

    for image_path in sorted(folder.glob("*")):
        if not image_path.is_file():
            continue

        try:
            with Image.open(image_path) as image:
                width, height = image.size
                mode = image.mode
                image_format = image.format

            records.append({
                "path": str(image_path),
                "label": label,
                "class": "synthetic" if label == 1 else "real",
                "width": width,
                "height": height,
                "mode": mode,
                "format": image_format,
                "file_size_kb": image_path.stat().st_size / 1024,
            })

        except Exception as error:
            print(f"Skipping invalid image: {image_path}")
            print(f"Reason: {error}")

df = pd.DataFrame(records)

output_path = Path("data/processed/image_metadata.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output_path, index=False)

print("===== DATASET METADATA =====")
print(f"Valid images analyzed: {len(df)}")

print("\n===== CLASS COUNTS =====")
print(df["class"].value_counts())

print("\n===== IMAGE MODES =====")
print(df["mode"].value_counts())

print("\n===== IMAGE FORMATS =====")
print(df["format"].value_counts())

print("\n===== DIMENSION SUMMARY =====")
print(df[["width", "height"]].describe())

print("\n===== FILE SIZE SUMMARY (KB) =====")
print(df["file_size_kb"].describe())

print(f"\nMetadata saved to: {output_path}")