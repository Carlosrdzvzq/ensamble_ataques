import pandas as pd  # type: ignore
import numpy as np   # type: ignore
from Modelos.logistic_regression_model import train as train_lr, test as test_lr
from Modelos.random_forest_model import train as train_rf, test as test_rf
from Modelos.hist_gradient_boosting_model import train as train_hgb, test as test_hgb
from Modelos.meta_model import train as train_meta, test as test_meta
from Modelos.meta_model_rf import train as train_meta_rf, test as test_meta_rf
from Modelos.soft_voting_model import test_soft_voting

# ENTRENAMIENTO Y EVALUACIÓN INDIVIDUAL
def comprobacion_individual(train_df: pd.DataFrame, test_df: pd.DataFrame):

    resultado_lr_train = train_lr(train_df)
    resultado_lr_test = test_lr(resultado_lr_train["model"], test_df)

    resultado_rf_train = train_rf(train_df)
    resultado_rf_test = test_rf(resultado_rf_train["model"], test_df)

    resultado_hgb_train = train_hgb(train_df)
    resultado_hgb_test = test_hgb(resultado_hgb_train["model"], test_df)

    print("Resultados individuales con 5-fold en train y test\n")
    print("LR CV:", resultado_lr_train["metrics"])
    print("LR TEST:", resultado_lr_test["metrics"])
    print("RF CV:", resultado_rf_train["metrics"])
    print("RF TEST:", resultado_rf_test["metrics"])
    print("HGB CV:", resultado_hgb_train["metrics"])
    print("HGB TEST:", resultado_hgb_test["metrics"])

    return {
        "lr_train": resultado_lr_train, "lr_test": resultado_lr_test,
        "rf_train": resultado_rf_train, "rf_test": resultado_rf_test,
        "hgb_train": resultado_hgb_train, "hgb_test": resultado_hgb_test,
    }

# LÓGICA DE ENSAMBLES (SOFT VOTING)
def ejecutar_soft_voting(resultados: dict, test_df: pd.DataFrame):
    resultado_soft_voting = test_soft_voting(
        resultados["rf_test"]["model"],
        resultados["hgb_test"]["model"],
        resultados["lr_test"]["model"],
        test_df
    )
    print("\nSOFT VOTING RF + HGB + LR")
    print("SOFT VOTING TEST:", resultado_soft_voting["metrics"])
    return resultado_soft_voting

# LÓGICA DE ENSAMBLES (STACKING)
def construir_meta_datasets_stacking(resultados_modelos: dict, train_df: pd.DataFrame, test_df: pd.DataFrame):
    # Nota: Se usan las OOF (Out-of-Fold) para el Meta-Train para evitar Overfitting
    classes = resultados_modelos["lr_train"]["classes"]
    meta_train_dict = {}
    meta_test_dict = {}

    for i, clase in enumerate(classes):
        # Probabilidades OOF para entrenamiento
        meta_train_dict[f"lr_prob_{clase}"] = resultados_modelos["lr_train"]["oof_proba_train"][:, i]
        meta_train_dict[f"rf_prob_{clase}"] = resultados_modelos["rf_train"]["oof_proba_train"][:, i]
        meta_train_dict[f"hgb_prob_{clase}"] = resultados_modelos["hgb_train"]["oof_proba_train"][:, i]

        # Probabilidades reales para test
        meta_test_dict[f"lr_prob_{clase}"] = resultados_modelos["lr_test"]["proba_test"][:, i]
        meta_test_dict[f"rf_prob_{clase}"] = resultados_modelos["rf_test"]["proba_test"][:, i]
        meta_test_dict[f"hgb_prob_{clase}"] = resultados_modelos["hgb_test"]["proba_test"][:, i]

    meta_train = pd.DataFrame(meta_train_dict)
    meta_test = pd.DataFrame(meta_test_dict)
    
    return meta_train, meta_test, train_df["Label"].copy(), test_df["Label"].copy()

def ejecutar_stacking(meta_train, meta_test, y_meta_train, y_meta_test):
    # Usando el Meta-Modelo basado en Random Forest (Experimento B)
    resultado_meta_train = train_meta_rf(meta_train, y_meta_train)
    resultado_meta_test = test_meta_rf(resultado_meta_train["model"], meta_test, y_meta_test)

    print("\nSTACKING META-MODEL (RF)")
    print("META TEST:", resultado_meta_test["metrics"])
    print(resultado_meta_test["classification_report"])
    return resultado_meta_test

# --- MAIN: ORQUESTACIÓN ---

def main():
    # 1. Leer Datos
    train_df = pd.read_csv("train_insdn_cic.csv", encoding="utf-8-sig")
    test_df = pd.read_csv("test_insdn_cic.csv", encoding="utf-8-sig")

    # 2. Comprobación Individual
    resultados = comprobacion_individual(train_df, test_df)

    # 3. Soft Voting
    ejecutar_soft_voting(resultados, test_df)

    # 4. Stacking
    m_train, m_test, y_tr, y_te = construir_meta_datasets_stacking(resultados, train_df, test_df)
    ejecutar_stacking(m_train, m_test, y_tr, y_te)

    print("\n--- Fin del proceso ---")

if __name__ == "__main__":
    main()