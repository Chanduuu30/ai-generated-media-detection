from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_PATH = (
    PROJECT_ROOT
    / "data/processed/video/video_all_frame_features.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models/video/video_all_frame_svm.pkl"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "models/video/final_video_test_results.csv"
)


def main():

    print("===== FINAL VIDEO TEST EVALUATION =====")

    print("\nLoading all-frame video features...")

    df = pd.read_csv(FEATURES_PATH)

    feature_columns = [
        column
        for column in df.columns
        if column.startswith("feature_")
    ]

    test_df = df[
        df["split"] == "test"
    ].copy()

    print("\n===== TEST DATA =====")

    print(f"Test videos: {len(test_df)}")
    print(f"Number of features: {len(feature_columns)}")

    print("\n===== TEST CLASS DISTRIBUTION =====")

    print(
        test_df["class"].value_counts()
    )

    X_test = test_df[
        feature_columns
    ].to_numpy()

    y_test = (
        test_df["class"]
        .map(
            {
                "real": 0,
                "synthetic": 1,
            }
        )
        .to_numpy()
    )

    print("\n===== LOADING SELECTED VIDEO MODEL =====")

    model = joblib.load(MODEL_PATH)

    print("Model loaded successfully.")

    print("\n===== EVALUATING ON OFFICIAL TEST SET =====")

    predictions = model.predict(X_test)

    # SVC was trained with probability=True,
    # so predict_proba() is available.
    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    print("\n===== FINAL VIDEO TEST RESULTS =====")

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

    print("\nConfusion Matrix:")

    print(matrix)

    print("\n===== CLASSIFICATION REPORT =====")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Real",
                "Synthetic",
            ],
            zero_division=0,
        )
    )

    results = pd.DataFrame(
        [
            {
                "model": "MobileNetV3-Small + All-Frame SVM",
                "test_videos": len(test_df),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc_auc,
            }
        ]
    )

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        RESULTS_PATH,
        index=False,
    )

    print(
        f"Results saved to:\n{RESULTS_PATH}"
    )

    print(
        "\n===== FINAL VIDEO EVALUATION COMPLETE ====="
    )

    print(
        "The official test set was used ONLY for final evaluation."
    )


if __name__ == "__main__":
    main()