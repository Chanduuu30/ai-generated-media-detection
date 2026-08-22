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
    "data/processed/all_image_features.csv"
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
    ]

    validation_df = df[
        df["split"] == "validation"
    ]

    X_train = train_df[
        feature_columns
    ]

    y_train = train_df[
        "label"
    ]

    X_validation = validation_df[
        feature_columns
    ]

    y_validation = validation_df[
        "label"
    ]

    print("\n===== DATA SHAPES =====")

    print(
        f"Training features:   {X_train.shape}"
    )

    print(
        f"Validation features: {X_validation.shape}"
    )

    print(
        "\n===== TRAINING TUNED RANDOM FOREST ====="
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

    print("\n===== FINAL VALIDATION RESULTS =====")

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)


if __name__ == "__main__":
    main()