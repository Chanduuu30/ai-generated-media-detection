from pathlib import Path
import hashlib

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEO_ROOT = PROJECT_ROOT / "data/raw/video"

OUTPUT_PATH = (
    PROJECT_ROOT /
    "data/processed/video/video_duplicates.csv"
)


def calculate_hash(path, chunk_size=1024 * 1024):

    hasher = hashlib.sha256()

    with open(path, "rb") as file:

        while True:

            chunk = file.read(chunk_size)

            if not chunk:
                break

            hasher.update(chunk)

    return hasher.hexdigest()


def main():

    video_paths = sorted(
        VIDEO_ROOT.glob("*/**/*.mp4")
    )

    print("===== VIDEO DUPLICATE ANALYSIS =====")

    print(
        f"Videos discovered: {len(video_paths)}"
    )

    records = []

    for index, video_path in enumerate(
        video_paths,
        start=1,
    ):

        file_hash = calculate_hash(
            video_path
        )

        records.append(
            {
                "path": str(
                    video_path.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "hash": file_hash,
                "size_bytes": video_path.stat().st_size,
            }
        )

        if index % 250 == 0:

            print(
                f"Processed "
                f"{index}/{len(video_paths)} videos"
            )

    df = pd.DataFrame(records)

    duplicate_counts = (
        df["hash"]
        .value_counts()
    )

    duplicate_hashes = (
        duplicate_counts[
            duplicate_counts > 1
        ]
    )

    duplicate_groups = len(
        duplicate_hashes
    )

    duplicate_files = int(
        duplicate_hashes.sum()
        - duplicate_groups
    )

    print("\n===== DUPLICATE SUMMARY =====")

    print(
        f"Unique file hashes: "
        f"{df['hash'].nunique()}"
    )

    print(
        f"Duplicate groups: "
        f"{duplicate_groups}"
    )

    print(
        f"Duplicate files beyond originals: "
        f"{duplicate_files}"
    )

    if duplicate_groups > 0:

        print(
            "\n===== DUPLICATE GROUPS ====="
        )

        group_number = 1

        for file_hash in duplicate_hashes.index:

            group = df[
                df["hash"] == file_hash
            ]

            print(
                f"\nGroup {group_number}:"
            )

            for path in group["path"]:

                print(
                    f"  {path}"
                )

            group_number += 1

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nHash information saved to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()