from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_PATH = (
    PROJECT_ROOT /
    "data/processed/all_image_features.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT /
    "models/random_forest_tuning_results.csv"
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

    X_train = train_df[
        feature_columns
    ]

    y_train = train_df[
        "label"
    ]

    print("\n===== TRAINING DATA =====")

    print(
        f"Training samples: {X_train.shape[0]}"
    )

    print(
        f"Number of features: {X_train.shape[1]}"
    )

    print(
        "\n===== CONTROLLED RANDOM FOREST TUNING ====="
    )

    model = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
    )

    parameter_grid = {
    "n_estimators": [200, 300],
    "max_depth": [20, None],
    "min_samples_split": [2],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt"],
}
    search = GridSearchCV(
        estimator=model,
        param_grid=parameter_grid,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        verbose=2,
        return_train_score=True,
    )

    print(
        "\nStarting 3-fold grid search..."
    )

    search.fit(
        X_train,
        y_train,
    )

    print(
        "\n===== TUNING COMPLETE ====="
    )

    print(
        f"Best CV ROC-AUC: "
        f"{search.best_score_:.4f}"
    )

    print(
        "\n===== BEST PARAMETERS ====="
    )

    for parameter, value in (
        search.best_params_.items()
    ):
        print(
            f"{parameter}: {value}"
        )

    results = pd.DataFrame(
        search.cv_results_
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nTuning results saved to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()