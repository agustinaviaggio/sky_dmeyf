import logging
from datetime import datetime
import os
from src.features import *
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
logger.info(f"SQL_TABLE_NAME: {SQL_TABLE_NAME}")

### Main ###
def main():
    """Pipeline principal con optimización usando configuración YAML."""
    logger.info("=== INICIANDO INGENIERIA DE ATRIBUTOS CON CONFIGURACIÓN YAML ===")
    conn = None
    try:  
        # 1. Cargar datos y crear tabla sql
        conn = create_sql_table(DATA_PATH, SQL_TABLE_NAME)
        
        # 2. Obtener columnas con baja cardinalidad (< 10 valores únicos)
        low_cardinality_cols = get_low_cardinality_columns(conn, SQL_TABLE_NAME, max_unique=10)
        logger.info(f"Columnas con baja cardinalidad: {low_cardinality_cols}")
        
    except Exception as e:
        logger.error(f"Error en el pipeline: {e}")
        raise
    finally:
        if conn:
            conn.close()
            logger.info("Conexión cerrada")

if __name__ == "__main__":
    main()