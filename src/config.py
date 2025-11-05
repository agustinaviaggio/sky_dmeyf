import yaml
import os
import logging

logger = logging.getLogger(__name__)

# Ruta del archivo de configuracion
PATH_CONFIG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "conf.yaml")

try:
    with open(PATH_CONFIG, "r") as f:
        _cfgGeneral = yaml.safe_load(f)
        _cfg = _cfgGeneral["competencia02"]

        STUDY_NAME = _cfgGeneral("STUDY_NAME")
        DATA_PATH = _cfg("DATA_PATH")
        OUTPUT_PATH = _cfg("OUTPUT_PATH")
        SQL_TABLE_NAME = ('SQL_TABLE_NAME')
        SEMILLA = _cfg("SEMILLA")
        MESES_TRAIN = _cfg("MESES_TRAIN")
        MES_VALIDACION = _cfg("MES_VALIDACION")
        MES_TEST = _cfg("MES_TEST")
        GANANCIA_ACIERTO = _cfg("GANANCIA_ACIERTO")
        COSTO_ESTIMULO = _cfg("COSTO_ESTIMULO")

except Exception as e:
    logger.error(f"Error al cargar el archivo de configuracion: {e}")
    raise