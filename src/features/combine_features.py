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

OUTPUT_PATH = (
    PROJECT_ROOT /
    "data/processed/color_texture_features.csv"
)


def main():

    print("Loading color features...")
    color_df = pd.read_csv(COLOR_PATH)

    print("Loading texture features...")
    texture_df = pd.read_csv(TEXTURE_PATH)

    print("\n===== INPUT SHAPES =====")
    print(f"Color:   {color_df.shape}")
    print(f"Texture: {texture_df.shape}")

    # Verify that the images are in the same order.
    if not color_df["path"].equals(texture_df["path"]):
        raise ValueError(
            "Image paths do not match between "
            "color and texture datasets."
        )

    # Verify labels and splits.
    if not color_df["label"].equals(texture_df["label"]):
        raise ValueError(
            "Labels do not match between datasets."
        )

    if not color_df["split"].equals(texture_df["split"]):
        raise ValueError(
            "Splits do not match between datasets."
        )

    texture_columns = [
        column
        for column in texture_df.columns
        if column.startswith("texture_feature_")
    ]

    combined_df = pd.concat(
        [
            color_df,
            texture_df[texture_columns],
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
    