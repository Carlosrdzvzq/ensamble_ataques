import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix  # type: ignore


def _separar_x_y(df: pd.DataFrame, target_col: str = "Label"):
    x = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    x = x.replace([np.inf, -np.inf], np.nan)

    return x, y


def test_soft_voting(
    model_rf,
    model_hgb,
    model_lr,
    df: pd.DataFrame,
    target_col: str = "Label",
):
    x_test, y_test = _separar_x_y(df, target_col)

    proba_rf = model_rf.predict_proba(x_test)
    proba_hgb = model_hgb.predict_proba(x_test)
    proba_lr = model_lr.predict_proba(x_test)

    proba_final = (proba_rf + proba_hgb + proba_lr) / 3.0
    y_pred = np.argmax(proba_final, axis=1)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    }

    result = {
        "model_name": "SoftVoting_RF_HGB",
        "metrics": metrics,
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "pred_test": y_pred,
    }

    return result