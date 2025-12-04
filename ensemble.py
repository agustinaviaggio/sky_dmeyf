import pandas as pd
import numpy as np
import subprocess
from pathlib import Path
from datetime import datetime

# Configuración
BUCKET_NAME = "gs://sra_electron_bukito3/"
ENVIOS = 11000

# Ubicación de los archivos generados por Script 1
DIR_PRED = Path("preds_por_estudio")

# Detectar automáticamente los 3 archivos más recientes por estudio
def cargar_pred(study):
    files = sorted(DIR_PRED.glob(f"preds_{study}_*.csv"))
    if not files:
        raise FileNotFoundError(f"No se encontró archivo para {study}")
    print(f"Cargando {files[-1]}")
    return pd.read_csv(files[-1])

def main():
    fecha = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Cargar predicciones completas
    df_2511 = cargar_pred("2511_2")
    df_2611 = cargar_pred("2611_2")
    df_2711 = cargar_pred("2711_2")

    # Asegurar orden consistente
    df_2511 = df_2511.sort_values("numero_de_cliente").reset_index(drop=True)
    df_2611 = df_2611.sort_values("numero_de_cliente").reset_index(drop=True)
    df_2711 = df_2711.sort_values("numero_de_cliente").reset_index(drop=True)

    # Verificar que los clientes coinciden EXACTAMENTE
    if not (df_2511["numero_de_cliente"].equals(df_2611["numero_de_cliente"]) and
            df_2511["numero_de_cliente"].equals(df_2711["numero_de_cliente"])):
        raise ValueError("ERROR: No coinciden los número_de_cliente entre estudios.")

    print("✓ Los clientes están perfectamente alineados.")

    # Ensemble: promedio simple
    prob_final = (
        df_2511["prob"] +
        df_2611["prob"] +
        df_2711["prob"]
    ) / 3.0

    # Seleccionar top N envíos
    idx_top = np.argsort(prob_final)[::-1][:ENVIOS]
    clientes_top = df_2511.loc[idx_top, "numero_de_cliente"]

    # Crear submission
    df_sub = pd.DataFrame({"numero_de_cliente": clientes_top})

    Path("submissions_ensemble").mkdir(exist_ok=True)
    out = Path("submissions_ensemble") / f"submission_ensemble_top{ENVIOS}_{fecha}.csv"
    df_sub.to_csv(out, index=False)

    print(f"✓ Generado: {out}")

    # Subir al bucket
    gcs_path = f"{BUCKET_NAME}submissions/{out.name}"
    subprocess.run(["gsutil", "cp", str(out), gcs_path])

    print(f"✓ Subido a: {gcs_path}")
    print("✓ Ensemble final completado.")

if __name__ == "__main__":
    main()
