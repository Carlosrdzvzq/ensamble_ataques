from pathlib import Path
import pandas as pd #type: ignore
import numpy as np #type: ignore
import json

# ============================================================
# RUTAS
# Estructura esperada:
#
# Proyecto/
# └── Datasets/
#     ├── Datasets unidos (p1)/
#     │   ├── CIC_train_unido.csv
#     │   ├── CIC_test_unido.csv
#     │   └── InSDN_unido.csv
#     └── Datasets procesado (p2)/
#         └── procesar_datasets.py
# ============================================================

OUTPUT_DIR = Path(__file__).resolve().parent
UNIDOS_DIR = (OUTPUT_DIR.parent) / "Datasets unidos (p1)"

# Entradas
CIC_TRAIN_INPUT = UNIDOS_DIR / "CIC_train_unido.csv"
CIC_TEST_INPUT = UNIDOS_DIR / "CIC_test_unido.csv"
INSDN_INPUT = UNIDOS_DIR / "INSDN_unido.csv"

# Salidas
TRAIN_INSDN_CIC_OUTPUT = OUTPUT_DIR / "train_insdn_cic.csv"
TEST_INSDN_CIC_OUTPUT = OUTPUT_DIR / "test_insdn_cic.csv"
LABELS_MAP_OUTPUT = OUTPUT_DIR / "list_labels.json"

# Umbral para mandar clases raras a AnotherAttack
MIN_SAMPLES_PER_CLASS = 1000

# Umbral para balancear train
MAX_CLASS = 3000

# ------------------------------------------------------------
# Columnas a eliminar
# ------------------------------------------------------------
DROP_COLS_INSDN = [
    "Flow ID",
    "Src IP",
    "Src Port",
    "Dst IP",
    "Dst Port",
    "Timestamp",
]


# ------------------------------------------------------------
# Renombrado CIC -> formato InSDN
# ------------------------------------------------------------
CIC_TO_INSDN_RENAME = {
    "Total Fwd Packets": "Tot Fwd Pkts",
    "Total Backward Packets": "Tot Bwd Pkts",
    "Fwd Packets Length Total": "TotLen Fwd Pkts",
    "Bwd Packets Length Total": "TotLen Bwd Pkts",
    "Fwd Packet Length Max": "Fwd Pkt Len Max",
    "Fwd Packet Length Min": "Fwd Pkt Len Min",
    "Fwd Packet Length Mean": "Fwd Pkt Len Mean",
    "Fwd Packet Length Std": "Fwd Pkt Len Std",
    "Bwd Packet Length Max": "Bwd Pkt Len Max",
    "Bwd Packet Length Min": "Bwd Pkt Len Min",
    "Bwd Packet Length Mean": "Bwd Pkt Len Mean",
    "Bwd Packet Length Std": "Bwd Pkt Len Std",
    "Flow Bytes/s": "Flow Byts/s",
    "Flow Packets/s": "Flow Pkts/s",
    "Fwd IAT Total": "Fwd IAT Tot",
    "Bwd IAT Total": "Bwd IAT Tot",
    "Fwd Header Length": "Fwd Header Len",
    "Bwd Header Length": "Bwd Header Len",
    "Fwd Packets/s": "Fwd Pkts/s",
    "Bwd Packets/s": "Bwd Pkts/s",
    "Packet Length Min": "Pkt Len Min",
    "Packet Length Max": "Pkt Len Max",
    "Packet Length Mean": "Pkt Len Mean",
    "Packet Length Std": "Pkt Len Std",
    "Packet Length Variance": "Pkt Len Var",
    "FIN Flag Count": "FIN Flag Cnt",
    "SYN Flag Count": "SYN Flag Cnt",
    "RST Flag Count": "RST Flag Cnt",
    "PSH Flag Count": "PSH Flag Cnt",
    "ACK Flag Count": "ACK Flag Cnt",
    "URG Flag Count": "URG Flag Cnt",
    "ECE Flag Count": "ECE Flag Cnt",
    "Avg Packet Size": "Pkt Size Avg",
    "Avg Fwd Segment Size": "Fwd Seg Size Avg",
    "Avg Bwd Segment Size": "Bwd Seg Size Avg",
    "Fwd Avg Bytes/Bulk": "Fwd Byts/b Avg",
    "Fwd Avg Packets/Bulk": "Fwd Pkts/b Avg",
    "Fwd Avg Bulk Rate": "Fwd Blk Rate Avg",
    "Bwd Avg Bytes/Bulk": "Bwd Byts/b Avg",
    "Bwd Avg Packets/Bulk": "Bwd Pkts/b Avg",
    "Bwd Avg Bulk Rate": "Bwd Blk Rate Avg",
    "Subflow Fwd Packets": "Subflow Fwd Pkts",
    "Subflow Fwd Bytes": "Subflow Fwd Byts",
    "Subflow Bwd Packets": "Subflow Bwd Pkts",
    "Subflow Bwd Bytes": "Subflow Bwd Byts",
    "Init Fwd Win Bytes": "Init Fwd Win Byts",
    "Init Bwd Win Bytes": "Init Bwd Win Byts",
    "Fwd Act Data Packets": "Fwd Act Data Pkts",
}

# ------------------------------------------------------------
# Normalización de etiquetas
# Mantener etiquetas unificando equivalencias
# ------------------------------------------------------------
LABEL_NORMALIZATION = {
    # Benignos
    "benign": "Benign",
    "normal": "Benign",

    # CIC: unificar equivalencias obvias
    "drdos_dns": "DNS",
    "drdos_ldap": "LDAP",
    "ldap": "LDAP",
    "drdos_mssql": "MSSQL",
    "mssql": "MSSQL",
    "drdos_netbios": "NetBIOS",
    "netbios": "NetBIOS",
    "drdos_ntp": "NTP",
    "drdos_snmp": "SNMP",
    "drdos_udp": "UDP",
    "udp": "UDP",
    "udp-lag": "UDPLag",
    "udplag": "UDPLag",
    "syn": "Syn",
    "tftp": "TFTP",
    "portmap": "Portmap",
    "webddos": "WebDDoS",

    # InSDN: mantener nombres, solo normalizar escritura
    "ddos": "DDoS",
    "dos": "DoS",
    "probe": "Probe",
    "bfa": "BFA",
    "u2r": "U2R",
    "web-attack": "Web-Attack",
    "botnet": "BOTNET",
}


def leer_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")
    return pd.read_csv(path, low_memory=False, encoding="utf-8-sig")


def eliminar_columnas_no_deseadas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Eliminar columnas no deseadas
    cols_to_drop = [c for c in DROP_COLS_INSDN if c in df.columns]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    return df

def normalizar_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Label" not in df.columns:
        raise ValueError("El DataFrame no contiene la columna 'Label'.")

    def normalizar_valor(label):
        if pd.isna(label):
            return label

        label_str = str(label).strip()
        label_key = label_str.lower()

        return LABEL_NORMALIZATION.get(label_key, label_str)

    df["Label"] = df["Label"].apply(normalizar_valor)

    return df

def normalizar_columnas_cic_a_insdn(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rename_map = {col: CIC_TO_INSDN_RENAME[col] for col in df.columns if col in CIC_TO_INSDN_RENAME}
    df = df.rename(columns=rename_map)

    print(f"Se renombraron {len(rename_map)} columnas.")
    return df

def agrupar_rares_desde_train_en_another_attack(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.copy()
    test_df = test_df.copy()

    for nombre, df in {
        "train_df": train_df,
        "test_df": test_df,
    }.items():
        if "Label" not in df.columns:
            raise ValueError(f"El DataFrame '{nombre}' no contiene la columna 'Label'.")

    # Conteo solo en train
    label_counts_train = train_df["Label"].value_counts()

    # Labels raros en train
    rare_labels = set(label_counts_train[label_counts_train < MIN_SAMPLES_PER_CLASS].index.tolist())

    # Labels que aparecen en test pero no existen en train
    train_labels = set(train_df["Label"].dropna().unique())
    test_labels = set(test_df["Label"].dropna().unique())
    unseen_in_train = test_labels - train_labels

    # Todo lo que debe irse a Another_attack
    labels_to_group = sorted(rare_labels.union(unseen_in_train))

    train_df["Label"] = train_df["Label"].replace(labels_to_group, "Another_attack")
    test_df["Label"] = test_df["Label"].replace(labels_to_group, "Another_attack")

    print(f"Labels agrupados en 'Another_attack': {labels_to_group}")
    print(f"Total de labels agrupados: {len(labels_to_group)}")

    return train_df, test_df


def transformar_labels_a_numericos(
    train_insdn_cic: pd.DataFrame,
    test_insdn_cic: pd.DataFrame
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_insdn_cic_df = train_insdn_cic.copy()
    test_insdn_cic_df = test_insdn_cic.copy()

    for nombre, df in {
        "train_insdn_cic_df": train_insdn_cic_df,
        "test_insdn_cic_df": test_insdn_cic_df
    }.items():
        if "Label" not in df.columns:
            raise ValueError(f"El DataFrame '{nombre}' no contiene la columna 'Label'.")

    # Obtener todos los labels únicos de los 3 datasets
    all_labels = pd.concat(
        [
            train_insdn_cic_df["Label"],
            test_insdn_cic_df["Label"]
        ],
        ignore_index=True,
    ).dropna()

    unique_labels = sorted(all_labels.astype(str).unique())

    # Crear diccionario label -> número
    list_labels = {label: idx for idx, label in enumerate(unique_labels)}

    # Transformar columna Label en cada dataset
    train_insdn_cic_df["Label"] = train_insdn_cic_df["Label"].map(list_labels)
    test_insdn_cic_df["Label"] = test_insdn_cic_df["Label"].map(list_labels)

    print("Mapeo de labels a numéricos:")
    print(list_labels)

    return list_labels, train_insdn_cic_df, test_insdn_cic_df

def dividir_insdn_train_test(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    if df.empty:
        raise ValueError("El DataFrame de InSDN está vacío.")

    # Mezclar filas
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Punto de corte 60/40
    split_index = int(len(df) * train_ratio)

    insdn_train_df = df.iloc[:split_index].copy()
    insdn_test_df = df.iloc[split_index:].copy()

    print(f"InSDN train: {len(insdn_train_df)} registros")
    print(f"InSDN test: {len(insdn_test_df)} registros")

    return insdn_train_df, insdn_test_df

def balancear_train(
    df: pd.DataFrame,
    label_col: str = "Label",
    max_class: int = MAX_CLASS,
    random_state: int = 42
) -> pd.DataFrame:
    df = df.copy()

    if label_col not in df.columns:
        raise ValueError(f"El DataFrame no contiene la columna '{label_col}'.")

    if df.empty:
        raise ValueError("El DataFrame está vacío.")

    partes_balanceadas = []

    conteo_original = df[label_col].value_counts().sort_index()
    print("Conteo original por clase:")
    print(conteo_original.to_dict())

    for clase, grupo in df.groupby(label_col):
        if len(grupo) > max_class:
            grupo_balanceado = grupo.sample(n=max_class, random_state=random_state)
        else:
            grupo_balanceado = grupo

        partes_balanceadas.append(grupo_balanceado)

    df_balanceado = pd.concat(partes_balanceadas, ignore_index=True)

    # Mezclar resultado final
    df_balanceado = df_balanceado.sample(frac=1, random_state=random_state).reset_index(drop=True)

    conteo_final = df_balanceado[label_col].value_counts().sort_index()
    print("Conteo balanceado por clase:")
    print(conteo_final.to_dict())

    return df_balanceado

def guardar_list_labels(list_labels: dict, output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(list_labels, f, ensure_ascii=False, indent=4)

    print(f"Mapa de labels guardado en: {output_path}")

def main():
    # Leer los 3 datasets
    print("Se leen los 3 datasets P1.")
    cic_train_df = leer_csv(CIC_TRAIN_INPUT)
    cic_test_df = leer_csv(CIC_TEST_INPUT)
    insdn_df = leer_csv(INSDN_INPUT)

    
    # Eliminar columnas no deseadas en INSDN
    print("\nSe eliminan columnas no sedeadas P2.")
    insdn_df = eliminar_columnas_no_deseadas(insdn_df)


    # Normalizar labels en todos los datasets
    print("\nSe normalizan los labels de los datasets P3.")
    cic_train_df = normalizar_labels(cic_train_df)
    cic_test_df = normalizar_labels(cic_test_df)
    insdn_df = normalizar_labels(insdn_df)


    # Normaliza columnas de cic a insdn
    print("\nSe normalizan las columnas de cic a insdn P4.")
    cic_train_df = normalizar_columnas_cic_a_insdn(cic_train_df ) 
    cic_test_df = normalizar_columnas_cic_a_insdn(cic_test_df)


    # Dividir InSDN en 60 train/40 test
    print("\nSe divide InSDN en 60 train / 40 test.")
    insdn_train_df, insdn_test_df = dividir_insdn_train_test(insdn_df)


    # Unificación de datasets para entrenamiento y testeo
    print("\nSe unifican datasets finales.")
    train_insdn_cic = pd.concat([cic_train_df, insdn_train_df], ignore_index=True)
    test_insdn_cic = pd.concat([cic_test_df, insdn_test_df], ignore_index=True)
    print(f"train_insdn_cic: {len(train_insdn_cic)} registros")
    print(f"test_insdn_cic: {len(test_insdn_cic)} registros")


    # Reemplaza labels raros < MIN_SAMPLES_PER_CLASS por nuevo label Another_attack con relación en train
    print("\nSe reemplazan labels raros < MIN por Another_attack P5.")
    train_insdn_cic, test_insdn_cic = agrupar_rares_desde_train_en_another_attack(
        train_insdn_cic, test_insdn_cic
    )


    # Reemplaza los labels por sus versiones numericas
    print("\nSe reemplazan labels por version numerica.")
    list_labels, train_insdn_cic, test_insdn_cic = transformar_labels_a_numericos( 
        train_insdn_cic, test_insdn_cic
    )


    # Banalceo de train 
    print("\nSe balancea train.")
    train_insdn_cic_balanced = balancear_train(train_insdn_cic)


    # Guardar resultados
    train_insdn_cic_balanced.to_csv(TRAIN_INSDN_CIC_OUTPUT, index=False, encoding="utf-8-sig")
    test_insdn_cic.to_csv(TEST_INSDN_CIC_OUTPUT, index=False, encoding="utf-8-sig")


    # Resultados finales
    print("\nObtenemos los siguientes resultados finales.")
    conteo = train_insdn_cic_balanced['Label'].value_counts().sort_index()
    print("Conteo train:")
    print(conteo.to_dict())
    conteo = test_insdn_cic['Label'].value_counts().sort_index()
    print("Conteo test:")
    print(conteo.to_dict())


    # Guardar mapeo de labels
    guardar_list_labels(list_labels, LABELS_MAP_OUTPUT)


    print("Datasets procesados correctamente.")
    print(f"train_insdn_cic_balanced guardado en: {TRAIN_INSDN_CIC_OUTPUT}")
    print(f"test_insdn_cic guardado en: {TEST_INSDN_CIC_OUTPUT}")


if __name__ == "__main__":
    main()
