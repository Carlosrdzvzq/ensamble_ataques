import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.impute import SimpleImputer  # type: ignore
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix  # type: ignore


def _separar_x_y(meta_df: pd.DataFrame, y: pd.Series):
    x = meta_df.copy()
    y = y.copy()

    x = x.replace([np.inf, -np.inf], np.nan)

    return x, y


def _crear_modelo(random_state: int = 42):
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    min_samples_leaf=5,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return model


def train(
    meta_train: pd.DataFrame,
    y_meta_train: pd.Series,
    random_state: int = 42,
):
    x_train, y_train = _separar_x_y(meta_train, y_meta_train)

    model = _crear_modelo(random_state=random_state)
    model.fit(x_train, y_train)

    result = {
        "model_name": "MetaModel_RandomForest",
        "feature_columns": list(x_train.columns),
        "model": model,
    }

    return result


def test(
    model,
    meta_test: pd.DataFrame,
    y_meta_test: pd.Series,
):
    x_test, y_test = _separar_x_y(meta_test, y_meta_test)
    y_pred = model.predict(x_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    }

    result = {
        "model_name": "MetaModel_RandomForest",
        "metrics": metrics,
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "feature_columns": list(x_test.columns),
        "model": model,
        "pred_test": y_pred,
    }

    return result