import duckdb
import logging
import numpy as np
import pandas as pd
import subprocess
import tempfile
import lightgbm as lgb
import os
import gc
import yaml
import json
from pathlib import Path
from datetime import datetime

ESTUDIOS = ['2511_2', '2611_2', '2711_2']
BUCKET_NAME = "gs://sra_electron_bukito3/"
MES_PREDICCION = 202109
ENVIOS_FIJOS = 11000

FEATURES_FROM_MODEL = {}

os.makedirs("logs", exist_ok=True)
fecha = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"logs/log_prediccion_por_estudio_{fecha}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def refrescar_credenciales_gcs():
    from google.auth import default
    from google.auth.transport.requests import Request
    credentials, project = default()
    credentials.refresh(Request())
    os.environ['CLOUDSDK_AUTH_ACCESS_TOKEN'] = credentials.token
    return credentials.token

def cargar_features_por_estudio(study_name, bucket_name):
    refrescar_credenciales_gcs()
    gcs_pattern = f"{bucket_name}resultados/metadata_features_{study_name}_*.json"
    result = subprocess.run(['gsutil', 'ls', gcs_pattern],
                            capture_output=True, text=True)
    if result.returncode != 0:
        return None

    archivos = result.stdout.strip().split('\n')
    if not archivos or not archivos[0]:
        return None

    archivo = sorted(archivos)[-1]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        tmp_path = tmp.name

    subprocess.run(['gsutil', 'cp', archivo, tmp_path],
                   check=True, capture_output=True)

    with open(tmp_path, 'r') as f:
        metadata = json.load(f)
    os.unlink(tmp_path)
    return metadata['features']

def descargar_modelos_estudio(study_name, bucket_name):
    refrescar_credenciales_gcs()
    local_dir = Path.home() / f"modelos_temp_{study_name}"
    local_dir.mkdir(exist_ok=True)
    gcs_pattern = f"{bucket_name}modelos_finales/{study_name}_seed_*.txt"
    subprocess.run(['gsutil', '-m', 'cp', gcs_pattern, str(local_dir)],
                   check=True, capture_output=True)
    modelos = [lgb.Booster(model_file=str(p)) for p in sorted(local_dir.glob("*.txt"))]
    import shutil
    shutil.rmtree(local_dir)
    return modelos

def cargar_datos_prediccion(mes_prediccion):
    conf_file = Path(f"~/sky_dmeyf/{ESTUDIOS[0]}/conf.yaml").expanduser()
    data_path = yaml.safe_load(open(conf_file))['configuracion']['DATA_PATH_OPT']
    conn = duckdb.connect(database=':memory:')
    conn.execute("INSTALL httpfs;")
    conn.execute("LOAD httpfs;")

    token = refrescar_credenciales_gcs()
    conn.execute(f"CREATE SECRET (TYPE GCS, PROVIDER config, BEARER_TOKEN '{token}')")

    conn.execute("""
        CREATE TABLE datos AS
        SELECT * FROM read_parquet('{}')
    """.format(data_path))

    data = conn.execute(
        f"SELECT * FROM datos WHERE foto_mes = {mes_prediccion}"
    ).fetchnumpy()

    columnas_prohibidas = {'target_binario', 'target_ternario', 'foto_mes'}
    feature_cols = [c for c in data.keys() if c not in columnas_prohibidas]
    X = np.column_stack([data[c] for c in feature_cols])

    if 'numero_de_cliente' in data:
        numeros_cliente = data['numero_de_cliente']
    else:
        numeros_cliente = np.arange(X.shape[0])

    conn.close()
    return X, numeros_cliente, feature_cols

def obtener_features_de_modelo(modelo, feature_cols):
    columnas_prohibidas = {'target_binario', 'target_ternario', 'foto_mes'}
    feat = [f for f in modelo.feature_name() if f not in columnas_prohibidas]
    return feat

def predecir_estudio(modelos, X, feature_cols, study_name):
    features = cargar_features_por_estudio(study_name, BUCKET_NAME)
    if features is None:
        features = obtener_features_de_modelo(modelos[0], feature_cols)

    indices = [feature_cols.index(f) for f in features]
    Xf = X[:, indices]

    preds = [m.predict(Xf) for m in modelos]
    preds_mean = np.mean(preds, axis=0)
    return preds_mean

def generar_submission_simple(prob, clientes, nombre):
    idx = np.argsort(prob)[::-1][:ENVIOS_FIJOS]
    df = pd.DataFrame({"numero_de_cliente": clientes[idx]})

    outdir = Path("submissions_por_estudio")
    outdir.mkdir(exist_ok=True)
    outfile = outdir / nombre
    df.to_csv(outfile, index=False)

    subprocess.run(['gsutil', 'cp', str(outfile), f"{BUCKET_NAME}submissions/{nombre}"])
    return outfile

def main():
    X, clientes, feature_cols = cargar_datos_prediccion(MES_PREDICCION)

    for study in ESTUDIOS:
        logger.info(f"====== Estudio {study} ======")
        modelos = descargar_modelos_estudio(study, BUCKET_NAME)
        pred = predecir_estudio(modelos, X, feature_cols, study)

        nombre = f"submission_{study}_top{ENVIOS_FIJOS}_{fecha}.csv"
        generar_submission_simple(pred, clientes, nombre)

        del modelos
        gc.collect()

    logger.info("Listo: 3 CSV generados.")

if __name__ == "__main__":
    main()
