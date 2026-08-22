from pathlib import Path

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

FEATURE_PATH = (
    PROJECT_ROOT /
    "data/processed/color_texture_features.csv"
)


def main():

    print("Loading combined feature dataset...")

    df = pd.read_csv(FEATURE_PATH)

    feature_columns = [
        column
        for column in df.columns
        if column.startswith("feature_")
        or column.startswith("texture_feature_")
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
        f"Number of features: {len(feature_columns)}"
    )

    print(
        "\n===== TRAINING RANDOM FOREST "
        "WITH COLOR + TEXTURE ====="
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    print("Training complete.")

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
    )

    recall = recall_score(
        y_validation,
        predictions,
    )

    f1 = f1_score(
        y_validation,
        predictions,
    )

    roc_auc = roc_auc_score(
        y_validation,
        probabilities,
    )

    matrix = confusion_matrix(
        y_validation,
        predictions,
    )

    print("\n===== VALIDATION RESULTS =====")

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)


if __name__ == "__main__":
    main()