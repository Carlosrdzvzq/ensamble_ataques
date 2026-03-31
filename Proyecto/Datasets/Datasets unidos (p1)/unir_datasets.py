from pathlib import Path
import pandas as pd #type: ignore


# ------------------------------------------------------------
# ESTRUCTURA ESPERADA
# PROYECTO MINERIA/
# ├── CIC-DDoS2019/
# ├── InSDN/
# └── Proyecto/
#     └── Datasets/
#         └── Datasets unidos (p1)
#               └── unir_datasets.py
# ------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent.parent

CIC_DIR = BASE_DIR / "CIC-DDoS2019"
INSDN_DIR = BASE_DIR / "InSDN"

OUTPUT_CIC_TRAIN = SCRIPT_DIR / "CIC_train_unido.csv"
OUTPUT_CIC_TEST = SCRIPT_DIR / "CIC_test_unido.csv"

OUTPUT_INSDN = SCRIPT_DIR / "INSDN_unido.csv"

def obtener_archivos_parquet(carpeta: Path, tipo: str):
    """
    tipo = 'training' o 'testing'
    """
    archivos = sorted(
        [
            p for p in carpeta.iterdir()
            if p.is_file()
            and p.suffix.lower() == ".parquet"
            and tipo.lower() in p.stem.lower()
        ]
    )
    return archivos

def obtener_archivos_csv(carpeta: Path):

    archivos = sorted(
        [
            p for p in carpeta.iterdir()
            if p.is_file()
            and p.suffix.lower() == ".csv"
        ]
    )
    return archivos


def unir_parquets_a_csv(archivos, output_csv: Path):
    if not archivos:
        print(f"No se encontraron archivos para generar: {output_csv.name}")
        return

    dataframes = []

    print(f"\nGenerando {output_csv.name} ...")
    for archivo in archivos:
        print(f"Leyendo: {archivo.name}")
        df = pd.read_parquet(archivo)
        dataframes.append(df)

    df_final = pd.concat(dataframes, ignore_index=True)

    print(f"Guardando: {output_csv}")
    df_final.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"Archivo generado correctamente: {output_csv.name}")
    print(f"Filas: {len(df_final)}")
    print(f"Columnas: {len(df_final.columns)}")


def unir_csvs(archivos, output_csv: Path):
    if not archivos:
        print(f"No se encontraron archivos para generar: {output_csv.name}")
        return

    dataframes = []

    print(f"\nGenerando {output_csv.name} ...")
    for archivo in archivos:
        print(f"Leyendo: {archivo.name}")
        df = pd.read_csv(archivo)
        dataframes.append(df)

    df_final = pd.concat(dataframes, ignore_index=True)

    print(f"Guardando: {output_csv}")
    df_final.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"Archivo generado correctamente: {output_csv.name}")
    print(f"Filas: {len(df_final)}")
    print(f"Columnas: {len(df_final.columns)}")


def main():
    if not CIC_DIR.exists():
        raise FileNotFoundError(f"No existe la carpeta: {CIC_DIR}")

    archivos_training = obtener_archivos_parquet(CIC_DIR, "training")
    archivos_testing = obtener_archivos_parquet(CIC_DIR, "testing")

    archivos_insdn = obtener_archivos_csv(INSDN_DIR)

    print("\nArchivos insdn encontrados:")
    for a in archivos_insdn:
        print(f" - {a.name}")

    print("\nArchivos cic training encontrados:")
    for a in archivos_training:
        print(f" - {a.name}")

    print("\nArchivos cic testing encontrados:")
    for a in archivos_testing:
        print(f" - {a.name}")

    unir_csvs(
        archivos_insdn,
        OUTPUT_INSDN
    )

    unir_parquets_a_csv(
        archivos_training,
        OUTPUT_CIC_TRAIN 
    )

    unir_parquets_a_csv(
        archivos_testing,
        OUTPUT_CIC_TEST
    )


if __name__ == "__main__":
    main()