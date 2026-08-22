from pathlib import Path

import numpy as np
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FRAME_METADATA_PATH = (
    PROJECT_ROOT /
    "data/processed/video/frame_metadata.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT /
    "data/processed/video/"
    "video_all_frame_features.csv"
)

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2
RANDOM_STATE = 42


class FrameDataset(Dataset):

    def __init__(self, dataframe, transform):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_path = (
            PROJECT_ROOT /
            row["path"]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        image = self.transform(
            image
        )

        return image


def create_feature_extractor():

    print(
        "===== LOADING MOBILENETV3-SMALL ====="
    )

    weights = (
        models.MobileNet_V3_Small_Weights.DEFAULT
    )

    model = models.mobilenet_v3_small(
        weights=weights
    )

    feature_extractor = torch.nn.Sequential(
        model.features,
        model.avgpool,
        torch.nn.Flatten(),
    )

    feature_extractor.eval()

    for parameter in (
        feature_extractor.parameters()
    ):
        parameter.requires_grad = False

    return (
        feature_extractor,
        weights.transforms(),
    )


def extract_features(
    dataframe,
    model,
    transform,
    device,
):

    dataset = FrameDataset(
        dataframe,
        transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    all_features = []

    total = len(dataframe)

    print(
        "\n===== EXTRACTING FEATURES ====="
    )

    with torch.no_grad():

        for batch_number, images in enumerate(
            loader,
            start=1,
        ):

            images = images.to(device)

            outputs = model(images)

            outputs = (
                outputs.cpu()
                .numpy()
            )

            all_features.append(
                outputs
            )

            processed = min(
                batch_number * BATCH_SIZE,
                total,
            )

            if (
                processed % 1000 < BATCH_SIZE
                or processed == total
            ):
                print(
                    f"Processed "
                    f"{processed}/{total} frames"
                )

    return np.vstack(
        all_features
    )


def aggregate_video_features(
    dataframe,
    frame_features,
):

    print(
        "\n===== AGGREGATING FRAMES TO VIDEOS ====="
    )

    working_df = dataframe.copy()

    working_df["_feature_index"] = np.arange(
        len(working_df)
    )

    video_rows = []

    grouped = working_df.groupby(
        [
            "split",
            "class",
            "video_id",
        ],
        sort=False,
    )

    for (
        split,
        class_name,
        video_id,
    ), group in grouped:

        indices = (
            group["_feature_index"]
            .to_numpy()
        )

        features = frame_features[
            indices
        ]

        # Statistical aggregation preserves
        # information beyond a simple mean.
        mean_features = features.mean(
            axis=0
        )

        std_features = features.std(
            axis=0
        )

        max_features = features.max(
            axis=0
        )

        min_features = features.min(
            axis=0
        )

        combined = np.concatenate(
            [
                mean_features,
                std_features,
                max_features,
                min_features,
            ]
        )

        row = {
            "split": split,
            "class": class_name,
            "video_id": video_id,
            "frames_used": len(indices),
        }

        for i, value in enumerate(
            combined
        ):

            row[
                f"feature_{i}"
            ] = float(value)

        video_rows.append(
            row
        )

    return pd.DataFrame(
        video_rows
    )


def main():

    np.random.seed(
        RANDOM_STATE
    )

    torch.manual_seed(
        RANDOM_STATE
    )

    print(
        "===== ALL-FRAME VIDEO FEATURES ====="
    )

    print(
        "Loading frame metadata..."
    )

    df = pd.read_csv(
        FRAME_METADATA_PATH
    )

    print(
        f"Total frames: {len(df)}"
    )

    print(
        f"Total videos: "
        f"{df['video_id'].nunique()}"
    )

    print(
        "\n===== FRAMES BY SPLIT ====="
    )

    print(
        df["split"].value_counts()
    )

    print(
        "\n===== FRAMES BY CLASS ====="
    )

    print(
        df["class"].value_counts()
    )

    device = torch.device(
        "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    model, transform = (
        create_feature_extractor()
    )

    model = model.to(device)

    frame_features = extract_features(
        df,
        model,
        transform,
        device,
    )

    print(
        "\n===== FEATURE EXTRACTION COMPLETE ====="
    )

    print(
        f"Frame feature shape: "
        f"{frame_features.shape}"
    )

    video_features = (
        aggregate_video_features(
            df,
            frame_features,
        )
    )

    feature_columns = [
        column
        for column in video_features.columns
        if column.startswith(
            "feature_"
        )
    ]

    print(
        "\n===== VIDEO FEATURE DATASET ====="
    )

    print(
        f"Videos: "
        f"{len(video_features)}"
    )

    print(
        f"Features per video: "
        f"{len(feature_columns)}"
    )

    print(
        "\n===== VIDEO COUNTS ====="
    )

    print(
        video_features["split"]
        .value_counts()
    )

    print(
        "\n===== VIDEO CLASS COUNTS ====="
    )

    print(
        video_features["class"]
        .value_counts()
    )

    print(
        "\n===== SPLIT × CLASS ====="
    )

    print(
        pd.crosstab(
            video_features["split"],
            video_features["class"],
        )
    )

    print(
        "\n===== FRAMES USED PER VIDEO ====="
    )

    print(
        video_features[
            "frames_used"
        ].describe()
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    video_features.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\n===== COMPLETE ====="
    )

    print(
        f"Features saved to:"
        f"\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()