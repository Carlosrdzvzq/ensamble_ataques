import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.impute import SimpleImputer  # type: ignore
from sklearn.model_selection import StratifiedKFold  # type: ignore
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix  # type: ignore


def _separar_x_y(df: pd.DataFrame, target_col: str = "Label"):
    x = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    return x, y


def _crear_modelo(random_state: int = 42):
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    random_state=random_state,
                    max_iter=1000,
                    learning_rate=0.1,
                ),
            ),
        ]
    )
    return model


def train(
    df: pd.DataFrame,
    target_col: str = "Label",
    n_splits: int = 5,
    random_state: int = 42,
):
    x, y = _separar_x_y(df, target_col)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    accuracies = []
    macro_f1s = []
    weighted_f1s = []

    classes = np.sort(y.unique())
    n_classes = len(classes)

    # Aquí se guardarán las predicciones out-of-fold
    oof_pred = np.zeros(len(df), dtype=int)
    # Predicciones OOF de probabilidad
    oof_proba = np.zeros((len(df), n_classes), dtype=float)

    for train_idx, val_idx in skf.split(x, y):
        x_train_fold = x.iloc[train_idx]
        x_val_fold = x.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]

        model = _crear_modelo(random_state=random_state)
        model.fit(x_train_fold, y_train_fold)

        y_pred = model.predict(x_val_fold)
        y_proba = model.predict_proba(x_val_fold)

        # Guardar predicciones y probabilidades del fold en sus posiciones originales
        oof_pred[val_idx] = y_pred
        oof_proba[val_idx] = y_proba

        accuracies.append(accuracy_score(y_val_fold, y_pred))
        macro_f1s.append(f1_score(y_val_fold, y_pred, average="macro", zero_division=0))
        weighted_f1s.append(f1_score(y_val_fold, y_pred, average="weighted", zero_division=0))

    final_model = _crear_modelo(random_state=random_state)
    final_model.fit(x, y)

    metrics = {
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),

        "macro_f1_mean": float(np.mean(macro_f1s)),
        "macro_f1_std": float(np.std(macro_f1s)),

        "weighted_f1_mean": float(np.mean(weighted_f1s)),
        "weighted_f1_std": float(np.std(weighted_f1s)),
    }

    result = {
        "model_name": "HistGradientBoostingClassifier",
        "metrics": metrics,
        "feature_columns": list(x.columns),
        "model": final_model,
        "classes": classes.tolist(),
        "oof_pred_train": oof_pred,
        "oof_proba_train": oof_proba,
    }

    return result


def test(model, df: pd.DataFrame, target_col: str = "Label"):
    x_test, y_test = _separar_x_y(df, target_col)

    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    }

    result = {
        "model_name": "HistGradientBoostingClassifier",
        "metrics": metrics,
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "feature_columns": list(x_test.columns),
        "model": model,
        "pred_test": y_pred,
        "proba_test": y_proba,
    }

    return result