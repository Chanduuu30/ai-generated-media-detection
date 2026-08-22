from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_FEATURES_PATH = (
    PROJECT_ROOT /
    "data/processed/all_image_features.csv"
)

VIDEO_FEATURES_PATH = (
    PROJECT_ROOT /
    "data/processed/video/"
    "video_all_frame_features.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = (
    MODEL_DIR /
    "multimodal_validation_model.pkl"
)

RESULTS_PATH = (
    MODEL_DIR /
    "multimodal_validation_results.csv"
)


def load_features(path):

    df = pd.read_csv(path)

    feature_columns = [
        column
        for column in df.columns
        if column.startswith("feature_")
    ]

    return df, feature_columns


def main():

    print(
        "===== MULTIMODAL VALIDATION EXPERIMENT ====="
    )

    print(
        "\nLoading image features..."
    )

    image_df, image_features = load_features(
        IMAGE_FEATURES_PATH
    )

    print(
        f"Image features: "
        f"{len(image_features)}"
    )

    print(
        "Loading video features..."
    )

    video_df, video_features = load_features(
        VIDEO_FEATURES_PATH
    )

    print(
        f"Video features: "
        f"{len(video_features)}"
    )

    # --------------------------------------------------
    # Keep only train and validation data.
    # The official test split is deliberately excluded.
    # --------------------------------------------------

    image_df = image_df[
        image_df["split"].isin(
            ["train", "validation"]
        )
    ].copy()

    video_df = video_df[
        video_df["split"].isin(
            ["train", "validation"]
        )
    ].copy()

    # --------------------------------------------------
    # Build video lookup.
    #
    # Image data contains image-level samples, while
    # video data contains video-level samples.
    #
    # Therefore we use the video feature distribution
    # separately rather than pretending image IDs and
    # video IDs are directly aligned.
    # --------------------------------------------------

    print(
        "\n===== DATASET INFORMATION ====="
    )

    print(
        f"Image rows: {len(image_df)}"
    )

    print(
        f"Video rows: {len(video_df)}"
    )

    print(
        "\nImage split counts:"
    )

    print(
        image_df["split"].value_counts()
    )

    print(
        "\nVideo split counts:"
    )

    print(
        video_df["split"].value_counts()
    )

    # --------------------------------------------------
    # Train separate models and combine their prediction
    # probabilities.
    #
    # This is safer than incorrectly joining unrelated
    # image and video samples.
    # --------------------------------------------------

    print(
        "\n===== TRAINING IMAGE MODEL ====="
    )

    image_train = image_df[
        image_df["split"] == "train"
    ]

    image_validation = image_df[
        image_df["split"] == "validation"
    ]

    X_image_train = image_train[
        image_features
    ].to_numpy()

    X_image_validation = image_validation[
        image_features
    ].to_numpy()

    y_image_train = (
        image_train["class"]
        .map(
            {
                "real": 0,
                "synthetic": 1,
            }
        )
        .to_numpy()
    )

    y_image_validation = (
        image_validation["class"]
        .map(
            {
                "real": 0,
                "synthetic": 1,
            }
        )
        .to_numpy()
    )

    image_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        max_features="sqrt",
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )

    image_model.fit(
        X_image_train,
        y_image_train,
    )

    image_probabilities = (
        image_model.predict_proba(
            X_image_validation
        )[:, 1]
    )

    print(
        "Image model trained."
    )

    print(
        "\n===== TRAINING VIDEO MODEL ====="
    )

    video_train = video_df[
        video_df["split"] == "train"
    ]

    video_validation = video_df[
        video_df["split"] == "validation"
    ]

    X_video_train = video_train[
        video_features
    ].to_numpy()

    X_video_validation = video_validation[
        video_features
    ].to_numpy()

    y_video_train = (
        video_train["class"]
        .map(
            {
                "real": 0,
                "synthetic": 1,
            }
        )
        .to_numpy()
    )

    y_video_validation = (
        video_validation["class"]
        .map(
            {
                "real": 0,
                "synthetic": 1,
            }
        )
        .to_numpy()
    )

    video_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        max_features="sqrt",
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )

    video_model.fit(
        X_video_train,
        y_video_train,
    )

    video_probabilities = (
        video_model.predict_proba(
            X_video_validation
        )[:, 1]
    )

    print(
        "Video model trained."
    )

    # --------------------------------------------------
    # The image and video validation sets contain
    # different numbers of samples.
    #
    # Therefore we evaluate their probability outputs
    # separately first and then create a normalized
    # multimodal score using their mean validation
    # performance.
    #
    # This experiment measures whether the two modalities
    # provide complementary predictive information.
    # --------------------------------------------------

    image_auc = roc_auc_score(
        y_image_validation,
        image_probabilities,
    )

    video_auc = roc_auc_score(
        y_video_validation,
        video_probabilities,
    )

    print(
        "\n===== INDIVIDUAL MODALITY ROC-AUC ====="
    )

    print(
        f"Image ROC-AUC: "
        f"{image_auc:.4f}"
    )

    print(
        f"Video ROC-AUC: "
        f"{video_auc:.4f}"
    )

    # --------------------------------------------------
    # Since the image and video validation examples are
    # not paired, evaluate a weighted ensemble through
    # repeated stratified score distributions.
    #
    # To keep the experiment statistically valid, use
    # the same class counts from each modality and create
    # a class-balanced ensemble score.
    # --------------------------------------------------

    image_real = image_probabilities[
        y_image_validation == 0
    ]

    image_fake = image_probabilities[
        y_image_validation == 1
    ]

    video_real = video_probabilities[
        y_video_validation == 0
    ]

    video_fake = video_probabilities[
        y_video_validation == 1
    ]

    sample_count = min(
        len(image_real),
        len(video_real),
        len(image_fake),
        len(video_fake),
    )

    rng = np.random.default_rng(
        42
    )

    image_real = rng.choice(
        image_real,
        sample_count,
        replace=False,
    )

    image_fake = rng.choice(
        image_fake,
        sample_count,
        replace=False,
    )

    video_real = rng.choice(
        video_real,
        sample_count,
        replace=False,
    )

    video_fake = rng.choice(
        video_fake,
        sample_count,
        replace=False,
    )

    image_scores = np.concatenate(
        [
            image_real,
            image_fake,
        ]
    )

    video_scores = np.concatenate(
        [
            video_real,
            video_fake,
        ]
    )

    y_true = np.concatenate(
        [
            np.zeros(sample_count),
            np.ones(sample_count),
        ]
    )

    # Test several ensemble weights.
    weights = [
        0.25,
        0.50,
        0.75,
    ]

    rows = []

    print(
        "\n===== MULTIMODAL ENSEMBLE ====="
    )

    for image_weight in weights:

        video_weight = (
            1.0 - image_weight
        )

        ensemble_scores = (
            image_weight * image_scores
            +
            video_weight * video_scores
        )

        ensemble_predictions = (
            ensemble_scores >= 0.5
        ).astype(int)

        accuracy = accuracy_score(
            y_true,
            ensemble_predictions,
        )

        precision = precision_score(
            y_true,
            ensemble_predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            ensemble_predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            ensemble_predictions,
            zero_division=0,
        )

        roc_auc = roc_auc_score(
            y_true,
            ensemble_scores,
        )

        matrix = confusion_matrix(
            y_true,
            ensemble_predictions,
        )

        print(
            f"\nImage weight: "
            f"{image_weight:.2f}"
        )

        print(
            f"Video weight: "
            f"{video_weight:.2f}"
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
            "Confusion Matrix:"
        )

        print(matrix)

        rows.append(
            {
                "image_weight": image_weight,
                "video_weight": video_weight,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc_auc,
            }
        )

    results = pd.DataFrame(
        rows
    )

    best_row = results.loc[
        results["roc_auc"].idxmax()
    ]

    print(
        "\n===== BEST MULTIMODAL RESULT ====="
    )

    print(
        best_row.to_string()
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "image_model": image_model,
            "video_model": video_model,
            "best_image_weight": float(
                best_row["image_weight"]
            ),
            "best_video_weight": float(
                best_row["video_weight"]
            ),
        },
        MODEL_PATH,
    )

    results.to_csv(
        RESULTS_PATH,
        index=False,
    )

    print(
        "\n===== EXPERIMENT COMPLETE ====="
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
        "\nOfficial test set was NOT used."
    )


if __name__ == "__main__":
    main()