from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_PATH = PROJECT_ROOT / "data/processed/image_features.csv"


def evaluate_model(model, X, y, dataset_name):
    """Evaluate a trained model."""

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions)
    recall = recall_score(y, predictions)
    f1 = f1_score(y, predictions)
    roc_auc = roc_auc_score(y, probabilities)

    matrix = confusion_matrix(y, predictions)

    print(f"\n===== {dataset_name.upper()} RESULTS =====")

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    return {
        "dataset": dataset_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


def main():

    print("Loading feature dataset...")

    df = pd.read_csv(FEATURE_PATH)

    feature_columns = [
        column
        for column in df.columns
        if column.startswith("feature_")
    ]

    X_train = df.loc[
        df["split"] == "train",
        feature_columns,
    ]

    y_train = df.loc[
        df["split"] == "train",
        "label",
    ]

    X_validation = df.loc[
        df["split"] == "validation",
        feature_columns,
    ]

    y_validation = df.loc[
        df["split"] == "validation",
        "label",
    ]

    print("\n===== DATA SHAPES =====")

    print(f"Training features:    {X_train.shape}")
    print(f"Validation features:  {X_validation.shape}")

    print("\n===== TRAINING LOGISTIC REGRESSION =====")

    model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(
        X_train,
        y_train,
    )

    print("Training complete.")

    validation_results = evaluate_model(
        model,
        X_validation,
        y_validation,
        "validation",
    )

    print("\n===== EXPERIMENT SUMMARY =====")

    for metric, value in validation_results.items():

        if metric == "dataset":
            continue

        print(
            f"{metric}: {value:.4f}"
        )


if __name__ == "__main__":
    main()
    