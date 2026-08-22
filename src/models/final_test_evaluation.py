from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
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

FEATURE_PATH = (
    PROJECT_ROOT /
    "data/processed/all_image_features.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = (
    MODEL_DIR /
    "final_random_forest.pkl"
)

RESULTS_PATH = (
    PROJECT_ROOT /
    "models/final_test_results.csv"
)


def main():

    print("Loading all image features...")

    df = pd.read_csv(FEATURE_PATH)

    feature_columns = [
        column
        for column in df.columns
        if column.startswith("feature_")
        or column.startswith("texture_feature_")
        or column.startswith("frequency_feature_")
    ]

    train_df = df[
        df["split"] == "train"
    ].copy()

    test_df = df[
        df["split"] == "test"
    ].copy()

    X_train = train_df[
        feature_columns
    ]

    y_train = train_df[
        "label"
    ]

    X_test = test_df[
        feature_columns
    ]

    y_test = test_df[
        "label"
    ]

    print("\n===== DATA SHAPES =====")

    print(
        f"Training data: {X_train.shape}"
    )

    print(
        f"Test data:     {X_test.shape}"
    )

    print(
        f"Number of features: {len(feature_columns)}"
    )

    print("\n===== TEST CLASS DISTRIBUTION =====")

    print(
        test_df["class"].value_counts()
    )

    print(
        "\n===== TRAINING FINAL MODEL ====="
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        max_features="sqrt",
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    print("Training complete.")

    print(
        "\n===== EVALUATING ON HELD-OUT TEST SET ====="
    )

    predictions = model.predict(
        X_test
    )

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
    )

    recall = recall_score(
        y_test,
        predictions,
    )

    f1 = f1_score(
        y_test,
        predictions,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    print("\n===== FINAL TEST RESULTS =====")

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
        )
    )

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
                "model": "Random Forest",
                "features": "color_texture_frequency",
                "num_features": len(feature_columns),
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
        f"\nModel saved to: {MODEL_PATH}"
    )

    print(
        f"Results saved to: {RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()