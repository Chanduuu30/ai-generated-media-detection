from pathlib import Path

import numpy as np
import pandas as pd
import torch

from PIL import Image

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FRAME_METADATA_PATH = (
    PROJECT_ROOT /
    "data/processed/video/frame_metadata.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT /
    "models/video"
)

FEATURES_PATH = (
    PROJECT_ROOT /
    "data/processed/video/video_cnn_features.csv"
)

RESULTS_PATH = (
    OUTPUT_DIR /
    "video_cpu_baseline_results.csv"
)

MODEL_PATH = (
    OUTPUT_DIR /
    "video_cpu_baseline.pkl"
)

IMAGE_SIZE = 224

# Only 3 frames per video.
FRAMES_PER_VIDEO = 3

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

        return image, index


def select_video_frames(df):

    """
    Select three representative frames
    from every video.

    We select approximately:
        early frame
        middle frame
        late frame
    """

    selected = []

    grouped = df.groupby(
        [
            "split",
            "class",
            "video_id",
        ],
        sort=False,
    )

    for _, group in grouped:

        group = group.sort_values(
            "filename"
        )

        if len(group) == 1:

            selected.append(
                group.iloc[0]
            )

        elif len(group) == 2:

            selected.extend(
                [
                    group.iloc[0],
                    group.iloc[1],
                ]
            )

        else:

            indices = [
                0,
                len(group) // 2,
                len(group) - 1,
            ]

            for index in indices:

                selected.append(
                    group.iloc[index]
                )

    return pd.DataFrame(
        selected
    ).reset_index(
        drop=True
    )


def create_model():

    print(
        "\n===== LOADING MOBILENETV3-SMALL ====="
    )

    weights = (
        models.MobileNet_V3_Small_Weights.DEFAULT
    )

    model = models.mobilenet_v3_small(
        weights=weights
    )

    # Remove the final classification layer.
    # The output becomes a feature vector.
    feature_extractor = torch.nn.Sequential(
        model.features,
        model.avgpool,
        torch.nn.Flatten(),
    )

    return feature_extractor


def extract_frame_features(
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

    features = []

    model.eval()

    with torch.no_grad():

        for batch_index, (
            images,
            indices,
        ) in enumerate(loader):

            images = images.to(
                device
            )

            outputs = model(
                images
            )

            outputs = (
                outputs.cpu()
                .numpy()
            )

            features.append(
                outputs
            )

            processed = (
                min(
                    (batch_index + 1)
                    * BATCH_SIZE,
                    len(dataframe),
                )
            )

            if processed % 1000 < BATCH_SIZE:

                print(
                    f"Processed "
                    f"{processed}/"
                    f"{len(dataframe)} frames"
                )

    return np.vstack(
        features
    )


def aggregate_video_features(
    dataframe,
    frame_features,
):

    rows = []

    # Add temporary feature index.
    temp_df = dataframe.copy()

    temp_df["_feature_index"] = np.arange(
        len(temp_df)
    )

    grouped = temp_df.groupby(
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

        video_feature = (
            frame_features[indices]
            .mean(axis=0)
        )

        rows.append(
            {
                "split": split,
                "class": class_name,
                "video_id": video_id,
                "feature": video_feature,
            }
        )

    return rows


def build_feature_dataframe(video_rows):

    feature_matrix = np.vstack(
        [
            row["feature"]
            for row in video_rows
        ]
    )

    metadata = pd.DataFrame(
        [
            {
                "split": row["split"],
                "class": row["class"],
                "video_id": row["video_id"],
            }
            for row in video_rows
        ]
    )

    feature_columns = [
        f"feature_{i}"
        for i in range(
            feature_matrix.shape[1]
        )
    ]

    feature_df = pd.DataFrame(
        feature_matrix,
        columns=feature_columns,
    )

    return pd.concat(
        [
            metadata.reset_index(drop=True),
            feature_df,
        ],
        axis=1,
    )


def evaluate_model(
    model,
    X,
    y,
    split_name,
):

    predictions = model.predict(
        X
    )

    probabilities = model.predict_proba(
        X
    )[:, 1]

    accuracy = accuracy_score(
        y,
        predictions,
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y,
        probabilities,
    )

    matrix = confusion_matrix(
        y,
        predictions,
    )

    print(
        f"\n===== {split_name.upper()} RESULTS ====="
    )

    print(
        f"Accuracy:  {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1-score:  {f1:.4f}"
    )

    print(
        f"ROC-AUC:   {roc_auc:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(matrix)

    return {
        "split": split_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


def main():

    np.random.seed(
        RANDOM_STATE
    )

    torch.manual_seed(
        RANDOM_STATE
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "===== CPU VIDEO BASELINE ====="
    )

    print(
        "Loading frame metadata..."
    )

    df = pd.read_csv(
        FRAME_METADATA_PATH
    )

    print(
        f"Total frames available: "
        f"{len(df)}"
    )

    print(
        f"Total videos represented: "
        f"{df['video_id'].nunique()}"
    )

    # ------------------------------------------------
    # Select 3 frames from every video.
    # ------------------------------------------------

    selected_df = select_video_frames(
        df
    )

    print(
        "\n===== FRAME SAMPLING ====="
    )

    print(
        f"Selected frames: "
        f"{len(selected_df)}"
    )

    print(
        f"Videos represented: "
        f"{selected_df['video_id'].nunique()}"
    )

    print(
        "Frames per video target: 3"
    )

    print(
        "\n===== VIDEOS BY SPLIT ====="
    )

    print(
        selected_df.groupby(
            "split"
        )["video_id"].nunique()
    )

    # ------------------------------------------------
    # CPU device.
    # ------------------------------------------------

    device = torch.device(
        "cpu"
    )

    print(
        "\nDevice: CPU"
    )

    print(
        "GPU is intentionally not used."
    )

    # ------------------------------------------------
    # ImageNet preprocessing.
    # ------------------------------------------------

    weights = (
        models.MobileNet_V3_Small_Weights.DEFAULT
    )

    transform = (
        weights.transforms()
    )

    # ------------------------------------------------
    # Load frozen CNN.
    # ------------------------------------------------

    feature_extractor = create_model()

    feature_extractor = (
        feature_extractor.to(device)
    )

    feature_extractor.eval()

    for parameter in (
        feature_extractor.parameters()
    ):

        parameter.requires_grad = False

    # ------------------------------------------------
    # Extract CNN features.
    # ------------------------------------------------

    print(
        "\n===== EXTRACTING FRAME FEATURES ====="
    )

    frame_features = (
        extract_frame_features(
            selected_df,
            feature_extractor,
            transform,
            device,
        )
    )

    print(
        "\nFrame feature matrix shape:"
    )

    print(
        frame_features.shape
    )

    # ------------------------------------------------
    # Aggregate frames → video.
    # ------------------------------------------------

    print(
        "\n===== AGGREGATING TO VIDEO LEVEL ====="
    )

    video_rows = (
        aggregate_video_features(
            selected_df,
            frame_features,
        )
    )

    video_features = (
        build_feature_dataframe(
            video_rows
        )
    )

    print(
        f"Video feature matrix: "
        f"{video_features.shape}"
    )

    print(
        "\nVideo counts:"
    )

    print(
        video_features["split"]
        .value_counts()
    )

    print(
        "\nVideo class counts:"
    )

    print(
        video_features["class"]
        .value_counts()
    )

    # ------------------------------------------------
    # Save video features.
    # ------------------------------------------------

    video_features.to_csv(
        FEATURES_PATH,
        index=False,
    )

    print(
        f"\nVideo features saved to:"
        f"\n{FEATURES_PATH}"
    )

    # ------------------------------------------------
    # Prepare train / validation data.
    # ------------------------------------------------

    feature_columns = [
        column
        for column in video_features.columns
        if column.startswith(
            "feature_"
        )
    ]

    train_df = video_features[
        video_features["split"] == "train"
    ]

    validation_df = video_features[
        video_features["split"] == "validation"
    ]

    X_train = train_df[
        feature_columns
    ].to_numpy()

    X_validation = validation_df[
        feature_columns
    ].to_numpy()

    y_train = (
        train_df["class"]
        .map(
            {
                "real": 0,
                "synthetic": 1,
            }
        )
        .to_numpy()
    )

    y_validation = (
        validation_df["class"]
        .map(
            {
                "real": 0,
                "synthetic": 1,
            }
        )
        .to_numpy()
    )

    print(
        "\n===== CLASSIFIER DATA ====="
    )

    print(
        f"Training videos: "
        f"{len(X_train)}"
    )

    print(
        f"Validation videos: "
        f"{len(X_validation)}"
    )

    print(
        f"Number of CNN features: "
        f"{len(feature_columns)}"
    )

    # ------------------------------------------------
    # Class-balanced logistic regression.
    # ------------------------------------------------

    print(
        "\n===== TRAINING LOGISTIC REGRESSION ====="
    )

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    classifier.fit(
        X_train,
        y_train,
    )

    print(
        "Training complete."
    )

    # ------------------------------------------------
    # Validation evaluation.
    # ------------------------------------------------

    results = []

    validation_result = evaluate_model(
        classifier,
        X_validation,
        y_validation,
        "validation",
    )

    results.append(
        validation_result
    )

    # ------------------------------------------------
    # Save classifier.
    # ------------------------------------------------

    import joblib

    joblib.dump(
        classifier,
        MODEL_PATH,
    )

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    print(
        "\n===== CPU VIDEO BASELINE COMPLETE ====="
    )

    print(
        f"Model saved to:"
        f"\n{MODEL_PATH}"
    )

    print(
        f"Results saved to:"
        f"\n{RESULTS_PATH}"
    )

    print(
        "\nThe official test set was NOT used."
    )


if __name__ == "__main__":
    main()