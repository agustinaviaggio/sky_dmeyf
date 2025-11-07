import logging
from datetime import datetime
import os

from src.features import create_sql_table, target_binario, target_ternario, generar_targets
from src.optimization_duck import optimizar
from src.config import *

### Configuración de logging ###
os.makedirs("logs", exist_ok=True)
fecha = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
nombre_log = f"log_{STUDY_NAME}_{fecha}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s",
    handlers=[
        logging.FileHandler("logs/" + nombre_log),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Iniciando programa de optimización con log fechado")

### Manejo de Configuración en YAML ###
logger.info("Configuración cargada desde YAML")
logger.info(f"STUDY_NAME: {STUDY_NAME}")
logger.info(f"DATA_PATH: {DATA_PATH}")
logger.info(f"SEMILLAS: {SEMILLAS}")
logger.info(f"MESES_TRAIN: {MESES_TRAIN}")
logger.info(f"MES_VALIDACION: {MES_VALIDACION}")
logger.info(f"GANANCIA_ACIERTO: {GANANCIA_ACIERTO}")
logger.info(f"COSTO_ESTIMULO: {COSTO_ESTIMULO}")

### Main ###
def main():
    """Pipeline principal con optimización usando configuración YAML."""
    logger.info("=== INICIANDO OPTIMIZACIÓN CON CONFIGURACIÓN YAML ===")

    conn = None # Inicializamos la conexión a None
    try:  
        # 1. Cargar datos y crear tabla sql
        conn = create_sql_table(DATA_PATH, SQL_TABLE_NAME)
        #conn = target_binario(conn, SQL_TABLE_NAME)
        #conn = target_ternario(conn, SQL_TABLE_NAME)
        conn = generar_targets(conn, SQL_TABLE_NAME)
  
        # 2. Ejecutar optimización
        study = optimizar(conn, SQL_TABLE_NAME, n_trials=5)
    
        # 5. Análisis adicional
        logger.info("=== ANÁLISIS DE RESULTADOS ===")
        trials_ordenados = sorted(study.trials, key=lambda t: t.value, reverse=True)[:5]
        logger.info("Top 5 mejores trials:")
        for trial in trials_ordenados:
            logger.info(f"  Trial {trial.number}: {trial.value:,.0f}")
    
        logger.info("=== OPTIMIZACIÓN COMPLETADA ===")
    
    except Exception as e:
        logger.error(f"Error durante la ejecución del pipeline: {e}")
        raise

    finally:
        if conn:
            conn.close()
            logger.info("Conexión a DuckDB cerrada.")

if __name__ == "__main__":
    main()