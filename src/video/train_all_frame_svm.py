from pathlib import Path

import joblib
import pandas as pd

from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_PATH = (
    PROJECT_ROOT /
    "data/processed/video/"
    "video_all_frame_features.csv"
)

MODEL_DIR = PROJECT_ROOT / "models/video"

MODEL_PATH = (
    MODEL_DIR /
    "video_all_frame_svm.pkl"
)

RESULTS_PATH = (
    MODEL_DIR /
    "video_all_frame_svm_results.csv"
)


def main():

    print(
        "===== ALL-FRAME VIDEO SVM ====="
    )

    print(
        "Loading video features..."
    )

    df = pd.read_csv(
        FEATURES_PATH
    )

    feature_columns = [
        column
        for column in df.columns
        if column.startswith("feature_")
    ]

    train_df = df[
        df["split"] == "train"
    ].copy()

    validation_df = df[
        df["split"] == "validation"
    ].copy()

    print(
        "\n===== DATA SHAPES ====="
    )

    print(
        f"Training videos: "
        f"{len(train_df)}"
    )

    print(
        f"Validation videos: "
        f"{len(validation_df)}"
    )

    print(
        f"Number of features: "
        f"{len(feature_columns)}"
    )

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
        "\n===== TRAINING SVM ====="
    )

    model = SVC(
        kernel="linear",
        C=1.0,
        class_weight="balanced",
        probability=True,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Training complete."
    )

    predictions = model.predict(
        X_validation
    )

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    accuracy = accuracy_score(
        y_validation,
        predictions,
    )

    precision = precision_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_validation,
        probabilities,
    )

    matrix = confusion_matrix(
        y_validation,
        predictions,
    )

    print(
        "\n===== VALIDATION RESULTS ====="
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

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    results = pd.DataFrame(
        [
            {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc_auc,
            }
        ]
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