import logging
from datetime import datetime
import os
import duckdb
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
logger.info(f"DATA_PATH_FE: {DATA_PATH_FE}")

def setup_duckdb_connection():
    """Configura conexión DuckDB en memoria pura (sin archivo)"""
    
    conn = duckdb.connect(database=':memory:')
    
    # Configuración MUY conservadora
    conn.execute("SET memory_limit='14GB'")  # Reducido
    conn.execute("SET max_memory='14GB'")
    conn.execute("SET threads=3")  # Reducido
    conn.execute("SET preserve_insertion_order=false")
    
    logger.info(f"DuckDB configurado:")
    logger.info(f"  - database: :memory:")
    logger.info(f"  - memory_limit: 14GB")
    logger.info(f"  - threads: 3")
    
    return conn

def main():
    """Pipeline principal sin checkpoints (directo a GCS)"""
    logger.info("=== INICIANDO INGENIERIA DE ATRIBUTOS (Sin Checkpoints) ===")
    
    conn = None
    
    try:
        # Setup DuckDB
        conn = setup_duckdb_connection()
        
        # Configurar acceso a GCS
        from google.auth import default
        from google.auth.transport.requests import Request
        
        credentials, project = default()
        credentials.refresh(Request())
        token = credentials.token
        
        conn.execute("INSTALL httpfs;")
        conn.execute("LOAD httpfs;")
        conn.execute(f"""
            CREATE SECRET (
                TYPE GCS,
                PROVIDER config,
                BEARER_TOKEN '{token}'
            )
        """)
        logger.info("Secret de GCS configurado exitosamente")
        
        # 1. Cargar datos
        logger.info("PASO 1: Cargando datos...")
        conn = create_sql_table_from_parquet_csv(conn, DATA_PATH_FE, SQL_TABLE_NAME)
        
        # 2. Atributos binarios de status
        logger.info("PASO 2: Creando atributos binarios...")
        conn = create_status_binary_attributes(conn, SQL_TABLE_NAME)
        cols_to_drop = ["master_status", "visa_status"]
        conn = drop_columns(conn, SQL_TABLE_NAME, cols_to_drop)
        
        # 3. Columnas baja cardinalidad
        logger.info("PASO 3: Identificando columnas de baja cardinalidad...")
        low_cardinality_cols = get_low_cardinality_columns(conn, SQL_TABLE_NAME, max_unique=10)
        
        # 4. Atributos de fechas TC
        logger.info("PASO 4: Creando atributos de fechas de tarjetas...")
        column_pairs = [
            ("Master_Finiciomora", "Visa_Finiciomora", "tc_finiciomora"),
            ("Master_Fvencimiento", "Visa_Fvencimiento", "tc_fvencimiento"),
            ("Master_fultimo_cierre", "Visa_fultimo_cierre", "tc_fultimocierre"),
            ("Master_fechaalta", "Visa_fechaalta", "tc_fechaalta"),
        ]
        conn, cols_tc_fecha = create_latest_and_earliest_credit_card_attributes(conn, SQL_TABLE_NAME, column_pairs)
        
        cols_to_drop = [
            "Master_Finiciomora", "Visa_Finiciomora",
            "Master_Fvencimiento", "Visa_Fvencimiento",
            "Master_fultimo_cierre", "Visa_fultimo_cierre",
            "Master_fechaalta", "Visa_fechaalta"
        ]
        conn = drop_columns(conn, SQL_TABLE_NAME, cols_to_drop)
        
        # 5. Suma TC
        logger.info("PASO 5: Creando atributos suma de tarjetas...")
        cols_visa = [c[0] for c in conn.execute(f"SELECT name FROM pragma_table_info('{SQL_TABLE_NAME}') WHERE name ILIKE '%visa%'").fetchall()]
        cols_master = [c[0] for c in conn.execute(f"SELECT name FROM pragma_table_info('{SQL_TABLE_NAME}') WHERE name ILIKE '%master%'").fetchall()]
        
        conn = create_sum_credit_card_attributes(conn, SQL_TABLE_NAME, cols_visa, cols_master)
        conn = drop_columns(conn, SQL_TABLE_NAME, cols_visa + cols_master)
        
        # 6. Ratios
        logger.info("PASO 6: Creando atributos de ratios...")
        conn = create_ratio_m_c_attributes(conn, SQL_TABLE_NAME)
        
        # 7. Lags (SIN CHECKPOINT)
        logger.info("PASO 7: Creando atributos LAG...")
        excluir_columnas_lag = ['numero_de_cliente', 'foto_mes', 'cliente_edad', 'cliente_antiguedad'] + cols_tc_fecha + low_cardinality_cols
        conn = create_lag_attributes(conn, SQL_TABLE_NAME, excluir_columnas_lag, cant_lag=2)
        
        # 8. Deltas (SIN CHECKPOINT)
        logger.info("PASO 8: Creando atributos DELTA...")
        cols_lag_list = [c[0] for c in conn.execute(f"SELECT name FROM pragma_table_info('{SQL_TABLE_NAME}') WHERE name LIKE '%lag_1' OR name LIKE '%lag_2'").fetchall()]
        excluir_columnas_delta = ['numero_de_cliente', 'foto_mes', 'cliente_edad', 'cliente_antiguedad'] + cols_lag_list + cols_tc_fecha + low_cardinality_cols
        conn = create_delta_attributes(conn, SQL_TABLE_NAME, excluir_columnas_delta, cant_delta=2)
        
        # 9. MAX (OPCIONAL - solo si hay suficiente memoria)
        logger.info("PASO 9: Creando atributos MAX...")
        try:
            cols_lag_delta_list = [c[0] for c in conn.execute(f"""
                SELECT name FROM pragma_table_info('{SQL_TABLE_NAME}')
                WHERE name LIKE '%lag_1' OR name LIKE '%lag_2'
                   OR name LIKE '%delta_1' OR name LIKE '%delta_2'
            """).fetchall()]
            excluir_columnas_max = ['numero_de_cliente', 'foto_mes', 'cliente_edad', 'cliente_antiguedad'] + cols_lag_delta_list + cols_tc_fecha + low_cardinality_cols
            conn = create_max_attributes(conn, SQL_TABLE_NAME, excluir_columnas_max, month_window=3)
        except Exception as e:
            logger.warning(f"No se pudieron crear atributos MAX: {e}")
            logger.info("Continuando sin atributos MAX...")
        
        # 10. MIN (OPCIONAL - solo si hay suficiente memoria)
        logger.info("PASO 10: Creando atributos MIN...")
        try:
            cols_lag_delta_max_list = [c[0] for c in conn.execute(f"""
                SELECT name FROM pragma_table_info('{SQL_TABLE_NAME}')
                WHERE name LIKE '%lag_1' OR name LIKE '%lag_2'
                   OR name LIKE '%delta_1' OR name LIKE '%delta_2'
                   OR name LIKE '%max_3'
            """).fetchall()]
            excluir_columnas_min = ['numero_de_cliente', 'foto_mes', 'cliente_edad', 'cliente_antiguedad'] + cols_lag_delta_max_list + cols_tc_fecha + low_cardinality_cols
            conn = create_min_attributes(conn, SQL_TABLE_NAME, excluir_columnas_min, month_window=3)
        except Exception as e:
            logger.warning(f"No se pudieron crear atributos MIN: {e}")
            logger.info("Continuando sin atributos MIN...")
        
        # 11. AVG (OPCIONAL - solo si hay suficiente memoria)
        logger.info("PASO 11: Creando atributos AVG...")
        try:
            cols_lag_delta_max_min_list = [c[0] for c in conn.execute(f"""
                SELECT name FROM pragma_table_info('{SQL_TABLE_NAME}')
                WHERE name LIKE '%lag_1' OR name LIKE '%lag_2'
                   OR name LIKE '%delta_1' OR name LIKE '%delta_2'
                   OR name LIKE '%max_3' OR name LIKE '%min_3'
            """).fetchall()]
            excluir_columnas_avg = ['numero_de_cliente', 'foto_mes', 'cliente_edad', 'cliente_antiguedad'] + cols_lag_delta_max_min_list + cols_tc_fecha + low_cardinality_cols
            conn = create_avg_attributes(conn, SQL_TABLE_NAME, excluir_columnas_avg, month_window=3)
        except Exception as e:
            logger.warning(f"No se pudieron crear atributos AVG: {e}")
            logger.info("Continuando sin atributos AVG...")
        
        # 12. Targets
        logger.info("PASO 12: Generando targets...")
        conn = generar_targets(conn, SQL_TABLE_NAME)
        
        # 13. Guardar DIRECTO A GCS (sin checkpoint local)
        logger.info("PASO 13: Guardando resultado final directo a GCS...")
        save_sql_table_to_parquet(conn, SQL_TABLE_NAME, OUTPUT_PATH_FE)
        
        logger.info("=== PIPELINE COMPLETADO EXITOSAMENTE ===")

    except Exception as e:
        logger.error(f"Error durante la ejecución del pipeline: {e}")
        raise

    finally:
        if conn:
            conn.close()
            logger.info("Conexión a DuckDB cerrada.")

if __name__ == "__main__":
    main()