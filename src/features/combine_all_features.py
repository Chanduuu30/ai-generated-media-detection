from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

COLOR_PATH = (
    PROJECT_ROOT /
    "data/processed/image_features.csv"
)

TEXTURE_PATH = (
    PROJECT_ROOT /
    "data/processed/texture_features.csv"
)

FREQUENCY_PATH = (
    PROJECT_ROOT /
    "data/processed/frequency_features.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT /
    "data/processed/all_image_features.csv"
)


def main():

    print("Loading feature datasets...")

    color_df = pd.read_csv(COLOR_PATH)
    texture_df = pd.read_csv(TEXTURE_PATH)
    frequency_df = pd.read_csv(FREQUENCY_PATH)

    print("\n===== INPUT SHAPES =====")
    print(f"Color:      {color_df.shape}")
    print(f"Texture:    {texture_df.shape}")
    print(f"Frequency:  {frequency_df.shape}")

    # Verify image ordering.
    if not color_df["path"].equals(texture_df["path"]):
        raise ValueError(
            "Color and texture image paths do not match."
        )

    if not color_df["path"].equals(frequency_df["path"]):
        raise ValueError(
            "Color and frequency image paths do not match."
        )

    # Verify labels.
    if not color_df["label"].equals(texture_df["label"]):
        raise ValueError(
            "Color and texture labels do not match."
        )

    if not color_df["label"].equals(frequency_df["label"]):
        raise ValueError(
            "Color and frequency labels do not match."
        )

    # Verify splits.
    if not color_df["split"].equals(texture_df["split"]):
        raise ValueError(
            "Color and texture splits do not match."
        )

    if not color_df["split"].equals(frequency_df["split"]):
        raise ValueError(
            "Color and frequency splits do not match."
        )

    texture_columns = [
        column
        for column in texture_df.columns
        if column.startswith("texture_feature_")
    ]

    frequency_columns = [
        column
        for column in frequency_df.columns
        if column.startswith("frequency_feature_")
    ]

    combined_df = pd.concat(
        [
            color_df,
            texture_df[texture_columns],
            frequency_df[frequency_columns],
        ],
        axis=1,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    feature_columns = [
        column
        for column in combined_df.columns
        if column.startswith("feature_")
        or column.startswith("texture_feature_")
        or column.startswith("frequency_feature_")
    ]

    print("\n===== COMBINATION COMPLETE =====")

    print(
        f"Total images: {len(combined_df)}"
    )

    print(
        f"Total features: {len(feature_columns)}"
    )

    print(
        f"Feature matrix shape: "
        f"{combined_df[feature_columns].shape}"
    )

    print("\n===== SPLIT COUNTS =====")
    print(
        combined_df["split"].value_counts()
    )

    print("\n===== CLASS COUNTS =====")
    print(
        combined_df["class"].value_counts()
    )

    print(
        f"\nSaved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()