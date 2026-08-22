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

FEATURE_PATH = (
    PROJECT_ROOT /
    "data/processed/texture_features.csv"
)


def evaluate_model(model, X, y):
    """Evaluate the trained model."""

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions)
    recall = recall_score(y, predictions)
    f1 = f1_score(y, predictions)
    roc_auc = roc_auc_score(y, probabilities)

    matrix = confusion_matrix(y, predictions)

    print("\n===== VALIDATION RESULTS =====")

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)


def main():

    print("Loading texture feature dataset...")

    df = pd.read_csv(FEATURE_PATH)

    feature_columns = [
        column
        for column in df.columns
        if column.startswith("texture_feature_")
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

    print(
        f"Training features:   {X_train.shape}"
    )

    print(
        f"Validation features: {X_validation.shape}"
    )

    print(
        "\n===== TRAINING LOGISTIC REGRESSION "
        "WITH TEXTURE FEATURES ====="
    )

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

    evaluate_model(
        model,
        X_validation,
        y_validation,
    )


if __name__ == "__main__":
    main()