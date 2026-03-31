import pandas as pd  # type: ignore
import numpy as np   # type: ignore
from Modelos.logistic_regression_model import train as train_lr, test as test_lr
from Modelos.random_forest_model import train as train_rf, test as test_rf
from Modelos.hist_gradient_boosting_model import train as train_hgb, test as test_hgb
# from Modelos.stacking_logistic_regression_model import train as train_meta, test as test_meta
from Modelos.stacking_random_forest_model import train as train_meta_rf, test as test_meta_rf
from Modelos.soft_voting_model import test_soft_voting


def imprimir_metricas_cv(nombre, metrics):
    print(f"{nombre} CV:")
    print(f"  Accuracy:   {metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f}")
    print(f"  Macro F1:   {metrics['macro_f1_mean']:.4f} ± {metrics['macro_f1_std']:.4f}")
    print(f"  Weighted F1:{metrics['weighted_f1_mean']:.4f} ± {metrics['weighted_f1_std']:.4f}")
    print()


# ENTRENAMIENTO Y EVALUACIÓN INDIVIDUAL
def comprobacion_individual(train_df: pd.DataFrame, test_df: pd.DataFrame):

    resultado_lr_train = train_lr(train_df)
    resultado_lr_test = test_lr(resultado_lr_train["model"], test_df)

    resultado_rf_train = train_rf(train_df)
    resultado_rf_test = test_rf(resultado_rf_train["model"], test_df)

    resultado_hgb_train = train_hgb(train_df)
    resultado_hgb_test = test_hgb(resultado_hgb_train["model"], test_df)

    
    
    

    print("Resultados individuales con 5-fold en train y test\n")
    imprimir_metricas_cv("LR", resultado_lr_train["metrics"])
    print("LR TEST:", resultado_lr_test["metrics"])
    imprimir_metricas_cv("RF", resultado_rf_train["metrics"])
    print("RF TEST:", resultado_rf_test["metrics"])
    imprimir_metricas_cv("HGB", resultado_hgb_train["metrics"])
    print("HGB TEST:", resultado_hgb_test["metrics"])

    return {
        "lr_train": resultado_lr_train, "lr_test": resultado_lr_test,
        "rf_train": resultado_rf_train, "rf_test": resultado_rf_test,
        "hgb_train": resultado_hgb_train, "hgb_test": resultado_hgb_test,
    }


# LÓGICA DE ENSAMBLES (STACKING)
def construir_meta_datasets_stacking(resultados_modelos: dict, train_df: pd.DataFrame, test_df: pd.DataFrame):
    clases_lr = list(resultados_modelos["lr_train"]["classes"])
    clases_rf = list(resultados_modelos["rf_train"]["classes"])
    clases_hgb = list(resultados_modelos["hgb_train"]["classes"])

    if clases_lr != clases_rf or clases_lr != clases_hgb:
        raise ValueError(
            f"El orden de clases no coincide:\n"
            f"LR: {clases_lr}\nRF: {clases_rf}\nHGB: {clases_hgb}"
        )

    classes = clases_lr
    meta_train_dict = {}
    meta_test_dict = {}

    lr_oof = resultados_modelos["lr_train"]["oof_proba_train"]
    rf_oof = resultados_modelos["rf_train"]["oof_proba_train"]
    hgb_oof = resultados_modelos["hgb_train"]["oof_proba_train"]

    lr_test = resultados_modelos["lr_test"]["proba_test"]
    rf_test = resultados_modelos["rf_test"]["proba_test"]
    hgb_test = resultados_modelos["hgb_test"]["proba_test"]

    for i, clase in enumerate(classes):
        meta_train_dict[f"lr_prob_{clase}"] = lr_oof[:, i]
        meta_train_dict[f"rf_prob_{clase}"] = rf_oof[:, i]
        meta_train_dict[f"hgb_prob_{clase}"] = hgb_oof[:, i]

        meta_test_dict[f"lr_prob_{clase}"] = lr_test[:, i]
        meta_test_dict[f"rf_prob_{clase}"] = rf_test[:, i]
        meta_test_dict[f"hgb_prob_{clase}"] = hgb_test[:, i]

    meta_train_dict["lr_conf_max"] = lr_oof.max(axis=1)
    meta_train_dict["rf_conf_max"] = rf_oof.max(axis=1)
    meta_train_dict["hgb_conf_max"] = hgb_oof.max(axis=1)

    meta_test_dict["lr_conf_max"] = lr_test.max(axis=1)
    meta_test_dict["rf_conf_max"] = rf_test.max(axis=1)
    meta_test_dict["hgb_conf_max"] = hgb_test.max(axis=1)

    meta_train_dict["lr_pred_class"] = lr_oof.argmax(axis=1)
    meta_train_dict["rf_pred_class"] = rf_oof.argmax(axis=1)
    meta_train_dict["hgb_pred_class"] = hgb_oof.argmax(axis=1)

    meta_test_dict["lr_pred_class"] = lr_test.argmax(axis=1)
    meta_test_dict["rf_pred_class"] = rf_test.argmax(axis=1)
    meta_test_dict["hgb_pred_class"] = hgb_test.argmax(axis=1)

    meta_train = pd.DataFrame(meta_train_dict)
    meta_test = pd.DataFrame(meta_test_dict)

    return meta_train, meta_test, train_df["Label"].copy(), test_df["Label"].copy()


def ejecutar_stacking(meta_train, meta_test, y_meta_train, y_meta_test):
    # Usando el Meta-Modelo basado en Random Forest (Experimento B)
    resultado_meta_train = train_meta_rf(meta_train, y_meta_train)
    resultado_meta_test = test_meta_rf(resultado_meta_train["model"], meta_test, y_meta_test)

    print("\nSTACKING META-MODEL (RF)")
    print("STACKING TEST:", resultado_meta_test["metrics"])
    print(resultado_meta_test["classification_report"])
    print("Confusion Matrix:")
    print(np.array(resultado_meta_test["confusion_matrix"]))
    return resultado_meta_test


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
    print(resultado_soft_voting["classification_report"])
    print("Confusion Matrix:")
    print(np.array(resultado_soft_voting["confusion_matrix"]))
    return resultado_soft_voting


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