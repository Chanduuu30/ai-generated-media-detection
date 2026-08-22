from pathlib import Path
from collections import defaultdict
import hashlib

DATASET_DIR = Path("data/raw/mediaeval/ITW-SM")


def calculate_sha256(file_path):
    """Calculate SHA-256 hash of a file."""
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


hash_to_files = defaultdict(list)

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
}


for image_path in sorted(DATASET_DIR.rglob("*")):

    if not image_path.is_file():
        continue

    if image_path.suffix.lower() not in image_extensions:
        continue

    file_hash = calculate_sha256(image_path)
    hash_to_files[file_hash].append(image_path)


duplicate_groups = [
    files
    for files in hash_to_files.values()
    if len(files) > 1
]


total_duplicate_files = sum(
    len(files) - 1
    for files in duplicate_groups
)


print("===== DUPLICATE ANALYSIS =====")
print(f"Unique file hashes: {len(hash_to_files)}")
print(f"Duplicate groups: {len(duplicate_groups)}")
print(f"Duplicate files beyond originals: {total_duplicate_files}")


if duplicate_groups:

    print("\n===== DUPLICATE GROUPS =====")

    for group_number, files in enumerate(duplicate_groups, start=1):

        print(f"\nGroup {group_number}:")

        for file_path in files:
            print(f"  {file_path}")