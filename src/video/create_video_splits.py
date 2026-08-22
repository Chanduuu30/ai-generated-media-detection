from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_PATH = (
    PROJECT_ROOT /
    "data/processed/video/video_metadata.csv"
)

TEST_LIST_PATH = (
    PROJECT_ROOT /
    "data/raw/video/List_of_testing_videos.txt"
)

OUTPUT_PATH = (
    PROJECT_ROOT /
    "data/splits/video/video_splits.csv"
)


RANDOM_STATE = 42


def load_official_test_list():

    test_paths = set()

    with open(TEST_LIST_PATH, "r") as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            # Format:
            # 1 YouTube-real/00170.mp4
            video_path = parts[-1]

            test_paths.add(
                video_path
            )

    return test_paths


def main():

    print("===== LOADING VIDEO METADATA =====")

    df = pd.read_csv(
        METADATA_PATH
    )

    print(
        f"Total videos: {len(df)}"
    )

    print("\n===== LOADING OFFICIAL TEST LIST =====")

    official_test_paths = (
        load_official_test_list()
    )

    print(
        f"Official test videos listed: "
        f"{len(official_test_paths)}"
    )

    # Convert project-relative path to
    # dataset-relative path.
    df["dataset_path"] = (
        df["path"]
        .str.replace(
            "data/raw/video/",
            "",
            regex=False,
        )
    )

    # Verify that every official test video
    # exists in our metadata.
    available_paths = set(
        df["dataset_path"]
    )

    missing_test_videos = (
        official_test_paths
        - available_paths
    )

    if missing_test_videos:

        print(
            "\nERROR: Official test videos "
            "missing from dataset:"
        )

        for path in sorted(
            missing_test_videos
        ):
            print(path)

        raise RuntimeError(
            "Some official test videos "
            "are missing."
        )

    # Assign official test videos.
    df["split"] = "train"

    df.loc[
        df["dataset_path"].isin(
            official_test_paths
        ),
        "split"
    ] = "test"

    test_df = df[
        df["split"] == "test"
    ].copy()

    remaining_df = df[
        df["split"] != "test"
    ].copy()

    print(
        "\n===== OFFICIAL TEST SET ====="
    )

    print(
        f"Test videos: {len(test_df)}"
    )

    print(
        "\nTest class distribution:"
    )

    print(
        test_df["class"].value_counts()
    )

    # Split remaining videos into:
    # 70% train
    # 15% validation
    #
    # Since the official test set is fixed,
    # the remaining data is split 82.35/17.65
    # to produce approximately 70/15 overall.

    train_df, validation_df = (
        train_test_split(
            remaining_df,
            test_size=0.17647,
            stratify=remaining_df["label"],
            random_state=RANDOM_STATE,
        )
    )

    train_df = train_df.copy()
    validation_df = validation_df.copy()

    train_df["split"] = "train"
    validation_df["split"] = "validation"

    final_df = pd.concat(
        [
            train_df,
            validation_df,
            test_df,
        ],
        ignore_index=True,
    )

    # Shuffle rows while keeping assignments fixed.
    final_df = final_df.sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    print(
        "\n===== FINAL SPLIT COUNTS ====="
    )

    print(
        final_df["split"].value_counts()
    )

    print(
        "\n===== SPLIT × CLASS ====="
    )

    print(
        pd.crosstab(
            final_df["split"],
            final_df["class"],
        )
    )

    print(
        "\n===== SOURCE × SPLIT ====="
    )

    print(
        pd.crosstab(
            final_df["source"],
            final_df["split"],
        )
    )

    # Check for duplicate paths across splits.
    print(
        "\n===== CHECK FOR PATH LEAKAGE ====="
    )

    path_split_counts = (
        final_df.groupby("path")["split"]
        .nunique()
    )

    duplicated_paths = (
        path_split_counts[
            path_split_counts > 1
        ]
    )

    print(
        "Paths appearing in multiple splits:",
        len(duplicated_paths),
    )

    # Check that official test videos are
    # exclusively in the test split.
    official_test_rows = final_df[
        final_df["dataset_path"].isin(
            official_test_paths
        )
    ]

    official_test_wrong_split = (
        official_test_rows[
            official_test_rows["split"] != "test"
        ]
    )

    print(
        "Official test videos outside test split:",
        len(official_test_wrong_split),
    )

    if len(duplicated_paths) > 0:
        raise RuntimeError(
            "Data leakage detected."
        )

    if len(official_test_wrong_split) > 0:
        raise RuntimeError(
            "Official test leakage detected."
        )

    # Save manifest.
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Keep dataset_path as useful metadata,
    # but the main path remains project-relative.
    final_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nVideo split manifest saved to:"
        f"\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()