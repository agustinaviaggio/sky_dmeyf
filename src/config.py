import yaml
import os
import logging

logger = logging.getLogger(__name__)

# Ruta del archivo de configuracion
PATH_CONFIG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "conf.yaml")

try:
    with open(PATH_CONFIG, "r") as f:
        _cfgGeneral = yaml.safe_load(f)
        _cfg = _cfgGeneral["experimento_colaborativo"]

        STUDY_NAME = _cfgGeneral.get("STUDY_NAME", "test")
        DATA_PATH = _cfg.get("DATA_PATH", "../data/competencia_crudo.csv")
        OUTPUT_PATH = _cfg.get("OUTPUT_PATH", "../data/competencia_02_fe.csv")
        SQL_TABLE_NAME = 'tabla_features' # Este valor es fijo en el código
        SEMILLA = _cfg.get("SEMILLA", [42])
        MES_TRAIN = _cfg.get("MES_TRAIN", "202102")
        MES_VALIDACION = _cfg.get("MES_VALIDACION", "202103")
        MES_TEST = _cfg.get("MES_TEST", "202104")
        GANANCIA_ACIERTO = _cfg.get("GANANCIA_ACIERTO", 780000)
        COSTO_ESTIMULO = _cfg.get("COSTO_ESTIMULO", 20000)

except Exception as e:
    logger.error(f"Error al cargar el archivo de configuracion: {e}")
    raise