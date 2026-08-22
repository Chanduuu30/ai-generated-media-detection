from pathlib import Path
from PIL import Image

DATASET_DIR = Path("data/raw/mediaeval/ITW-SM")

classes = {
    "0_real": "Authentic",
    "1_fake": "Synthetic",
}

valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}

for folder, label in classes.items():
    folder_path = DATASET_DIR / folder

    total = 0
    valid = 0
    invalid = []

    for file_path in sorted(folder_path.rglob("*")):
        if not file_path.is_file():
            continue

        total += 1

        if file_path.suffix.lower() not in valid_extensions:
            invalid.append((file_path, "unsupported extension"))
            continue

        try:
            with Image.open(file_path) as image:
                image.verify()

            valid += 1

        except Exception as error:
            invalid.append((file_path, str(error)))

    print(f"\n===== {label} ({folder}) =====")
    print(f"Total files:   {total}")
    print(f"Valid images:  {valid}")
    print(f"Invalid:       {len(invalid)}")

    if invalid:
        print("\nInvalid files:")
        for file_path, reason in invalid:
            print(f"{file_path} -> {reason}")
