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
        
        # 2. Obtener esquema de la tabla
        logger.info("Obteniendo esquema de la tabla")
        schema_df = conn.execute(f"""
            SELECT 
                name AS variable,
                type AS tipo_dato
            FROM 
                pragma_table_info('{SQL_TABLE_NAME}')
            ORDER BY 
                name
        """).df()
        
        logger.info(f"Esquema de la tabla {SQL_TABLE_NAME}:")
        print(schema_df.to_string(index=False))
        
        # Opcional: guardar el esquema en CSV
        schema_path = f"logs/schema_{STUDY_NAME}_{fecha}.csv"
        schema_df.to_csv(schema_path, index=False)
        logger.info(f"Esquema guardado en {schema_path}")
        
    except Exception as e:
        logger.error(f"Error en el pipeline: {e}")
        raise
    finally:
        if conn:
            conn.close()
            logger.info("Conexión cerrada")

if __name__ == "__main__":
    main()