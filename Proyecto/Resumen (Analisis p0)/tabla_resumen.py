from pathlib import Path
import pandas as pd #type: ignore

# ---------------------------------------------------------------------
# RUTAS
# Asume esta estructura:
# PROYECTO MINERIA/
# ├── CIC-DDoS2019/
# ├── InSDN/
# └── Proyecto/
#     └── Resumen (Analisis p0)
#          └── tabla_resumen.py
# ---------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent

CIC_DIR = BASE_DIR / "CIC-DDoS2019"
INSDN_DIR = BASE_DIR / "InSDN"

OUTPUT_CIC = SCRIPT_DIR / "tabla_resumen_CIC.csv"
OUTPUT_INSDN = SCRIPT_DIR / "tabla_resumen_InSDN.csv"

# Consideraremos como "benignos" tanto BENIGN/benign como Normal
BENIGN_LABELS = {"benign", "normal"}


def leer_columna_label(path: Path) -> pd.Series:
    """
    Lee únicamente la columna Label del archivo.
    """
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path, columns=["Label"])
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path, usecols=["Label"], encoding="utf-8-sig", low_memory=False)
    else:
        raise ValueError(f"Formato no soportado: {path}")

    return df["Label"].astype(str).str.strip()


def formatear_porcentaje(cantidad: int, total: int) -> str:
    """
    Devuelve el porcentaje respecto al total.
    Ejemplo: 50% o 54.73%
    """
    if total == 0:
        return "0%"

    porcentaje = (cantidad / total) * 100

    if porcentaje.is_integer():
        return f"{int(porcentaje)}%"

    return f"{porcentaje:.2f}%"


def resumir_archivo(path: Path) -> dict:
    """
    Genera una fila resumen para un archivo.
    """
    labels = leer_columna_label(path)
    conteos = labels.value_counts(dropna=False)
    total_registros = int(len(labels))

    benign_count = 0
    total_ataques = 0
    ataques = []

    for label, cantidad in conteos.items():
        cantidad = int(cantidad)
        porcentaje = formatear_porcentaje(cantidad, total_registros)

        if str(label).strip().lower() in BENIGN_LABELS:
            benign_count += cantidad
        else:
            total_ataques += cantidad
            ataques.append(f"{label}({cantidad} - {porcentaje})")

    benign_str = f"{benign_count} ({formatear_porcentaje(benign_count, total_registros)})"
    porcentaje_ataques = formatear_porcentaje(total_ataques, total_registros)

    return {
        "NombreArchivo": path.stem,
        "CantidadRegistros": total_registros,
        "Ataques": ", ".join(ataques) if ataques else "-",
        "Benign": benign_str,
        "TotalAtaques": total_ataques,
        "%Ataques": porcentaje_ataques,
    }


def construir_tabla(directorio: Path, extension: str) -> pd.DataFrame:
    """
    Construye la tabla resumen de todos los archivos de un directorio.
    """
    if not directorio.exists():
        raise FileNotFoundError(f"No existe el directorio: {directorio}")

    archivos = sorted(
        [p for p in directorio.iterdir() if p.is_file() and p.suffix.lower() == extension.lower()]
    )

    filas = [resumir_archivo(path) for path in archivos]
    return pd.DataFrame(
        filas,
        columns=[
            "NombreArchivo",
            "CantidadRegistros",
            "Ataques",
            "Benign",
            "TotalAtaques",
            "%Ataques",
        ],
    )


def main():
    # Tabla CIC
    tabla_cic = construir_tabla(CIC_DIR, ".parquet")

    # Tabla InSDN
    tabla_insdn = construir_tabla(INSDN_DIR, ".csv")

    # Mostrar en consola
    print("\n" + "=" * 130)
    print("TABLA RESUMEN - CIC-DDoS2019")
    print("=" * 130)
    print(tabla_cic.to_string(index=False))

    print("\n" + "=" * 130)
    print("TABLA RESUMEN - InSDN")
    print("=" * 130)
    print(tabla_insdn.to_string(index=False))

    # Guardar a CSV
    tabla_cic.to_csv(OUTPUT_CIC, index=False, encoding="utf-8-sig")
    tabla_insdn.to_csv(OUTPUT_INSDN, index=False, encoding="utf-8-sig")

    print("\nArchivos generados:")
    print(f"- {OUTPUT_CIC}")
    print(f"- {OUTPUT_INSDN}")


if __name__ == "__main__":
    main()