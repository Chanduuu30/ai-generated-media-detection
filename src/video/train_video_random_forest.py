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
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_PATH = (
    PROJECT_ROOT /
    "data/processed/video/video_cnn_features.csv"
)

MODEL_DIR = PROJECT_ROOT / "models/video"

MODEL_PATH = (
    MODEL_DIR /
    "video_random_forest.pkl"
)

RESULTS_PATH = (
    MODEL_DIR /
    "video_random_forest_results.csv"
)

RANDOM_STATE = 42


def evaluate_model(model, X, y):

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

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

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


def main():

    print(
        "===== VIDEO RANDOM FOREST ====="
    )

    print(
        "Loading video CNN features..."
    )

    df = pd.read_csv(
        FEATURES_PATH
    )

    feature_columns = [
        column
        for column in df.columns
        if column.startswith(
            "feature_"
        )
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
        "\n===== TRAINING RANDOM FOREST ====="
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Training complete."
    )

    results = evaluate_model(
        model,
        X_validation,
        y_validation,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    results_df = pd.DataFrame(
        [results]
    )

    results_df.to_csv(
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