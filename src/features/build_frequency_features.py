from pathlib import Path

import pandas as pd

from frequency_features import extract_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_PATH = PROJECT_ROOT / "data/splits/image_splits.csv"
OUTPUT_PATH = PROJECT_ROOT / "data/processed/frequency_features.csv"
ERROR_PATH = PROJECT_ROOT / "data/processed/frequency_feature_errors.csv"


def main():

    print("Loading dataset split manifest...")

    df = pd.read_csv(SPLIT_PATH)

    print(f"Images to process: {len(df)}")

    feature_rows = []
    failed_images = []

    for index, row in df.iterrows():

        image_path = PROJECT_ROOT / row["path"]

        try:

            features = extract_features(image_path)

            feature_row = {
                "path": row["path"],
                "label": row["label"],
                "class": row["class"],
                "split": row["split"],
            }

            for feature_index, value in enumerate(features):
                feature_row[
                    f"frequency_feature_{feature_index}"
                ] = value

            feature_rows.append(feature_row)

        except Exception as error:

            failed_images.append(
                {
                    "path": row["path"],
                    "error": str(error),
                }
            )

        if (index + 1) % 500 == 0:
            print(
                f"Processed {index + 1}/{len(df)} images"
            )

    feature_df = pd.DataFrame(feature_rows)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n===== FREQUENCY FEATURE EXTRACTION COMPLETE =====")

    print(
        f"Successful images: {len(feature_df)}"
    )

    print(
        f"Failed images: {len(failed_images)}"
    )

    if len(feature_df) > 0:

        feature_columns = [
            column
            for column in feature_df.columns
            if column.startswith("frequency_feature_")
        ]

        print(
            f"Number of features: "
            f"{len(feature_columns)}"
        )

        print(
            f"Feature matrix shape: "
            f"{feature_df[feature_columns].shape}"
        )

        print("\n===== SPLIT COUNTS =====")

        print(
            feature_df["split"].value_counts()
        )

        print("\n===== CLASS COUNTS =====")

        print(
            feature_df["class"].value_counts()
        )

    if failed_images:

        pd.DataFrame(
            failed_images
        ).to_csv(
            ERROR_PATH,
            index=False,
        )

        print(
            f"\nFailed-image report saved to: "
            f"{ERROR_PATH}"
        )

    print(
        f"\nFeatures saved to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()