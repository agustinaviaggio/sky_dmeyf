import logging
from datetime import datetime
import os
import duckdb
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

def setup_duckdb_connection():
    """Configura conexión DuckDB con archivo en disco"""
    
    # Usar archivo en disco en lugar de :memory:
    db_file = f'/tmp/duckdb_{os.environ.get("USER", "default")}_{STUDY_NAME}.db'
    
    # Eliminar archivo anterior si existe
    if os.path.exists(db_file):
        os.remove(db_file)
        logger.info(f"Archivo de base de datos anterior eliminado: {db_file}")
    
    conn = duckdb.connect(database=db_file)
    
    # Configuración conservadora
    conn.execute("SET memory_limit='16GB'")
    conn.execute("SET max_memory='16GB'")
    conn.execute("SET threads=4")
    conn.execute("SET preserve_insertion_order=false")
    
    logger.info(f"DuckDB configurado:")
    logger.info(f"  - database: {db_file}")
    logger.info(f"  - memory_limit: 16GB")
    logger.info(f"  - threads: 4")
    
    return conn, db_file

def cleanup_database(db_file):
    """Limpia archivo de base de datos"""
    try:
        if os.path.exists(db_file):
            os.remove(db_file)
            logger.info(f"Archivo de base de datos eliminado: {db_file}")
    except Exception as e:
        logger.warning(f"Error eliminando base de datos: {e}")

def save_checkpoint(conn, table_name, step_name):
    """Guarda checkpoint intermedio en Parquet"""
    checkpoint_path = f'/tmp/checkpoint_{step_name}.parquet'
    logger.info(f"Guardando checkpoint: {checkpoint_path}")
    
    conn.execute(f"COPY {table_name} TO '{checkpoint_path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    
    # Verificar tamaño
    size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    logger.info(f"Checkpoint guardado: {size_mb:.2f} MB")
    
    return checkpoint_path

def main():
    """Pipeline principal con checkpoints intermedios"""
    logger.info("=== INICIANDO INGENIERIA DE ATRIBUTOS (Versión Optimizada) ===")
    
    db_file = None
    conn = None
    
    try:
        # Setup DuckDB
        conn, db_file = setup_duckdb_connection()
        
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
        
        # Importar funciones después de configurar logging
        from src.features import *
        
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
        
        # CHECKPOINT 1
        save_checkpoint(conn, SQL_TABLE_NAME, "after_basic_features")
        
        # 7. Lags
        logger.info("PASO 7: Creando atributos LAG...")
        excluir_columnas_lag = ['numero_de_cliente', 'foto_mes', 'cliente_edad', 'cliente_antiguedad'] + cols_tc_fecha + low_cardinality_cols
        conn = create_lag_attributes(conn, SQL_TABLE_NAME, excluir_columnas_lag, cant_lag=2)
        
        # CHECKPOINT 2
        save_checkpoint(conn, SQL_TABLE_NAME, "after_lags")
        
        # 8. Deltas
        logger.info("PASO 8: Creando atributos DELTA...")
        cols_lag_list = [c[0] for c in conn.execute(f"SELECT name FROM pragma_table_info('{SQL_TABLE_NAME}') WHERE name LIKE '%lag_1' OR name LIKE '%lag_2'").fetchall()]
        excluir_columnas_delta = ['numero_de_cliente', 'foto_mes', 'cliente_edad', 'cliente_antiguedad'] + cols_lag_list + cols_tc_fecha + low_cardinality_cols
        
        # PROCESAR DELTAS EN CHUNKS MÁS PEQUEÑOS
        conn = create_delta_attributes_chunked(conn, SQL_TABLE_NAME, excluir_columnas_delta, cant_delta=2)
        
        # CHECKPOINT 3
        save_checkpoint(conn, SQL_TABLE_NAME, "after_deltas")
        
        # 9-11. MAX, MIN, AVG - SALTAR POR AHORA si siguen fallando
        logger.info("PASO 9-11: Saltando MAX, MIN, AVG por limitaciones de recursos")
        logger.info("Considera procesarlos en una máquina con más espacio en disco")
        
        # 12. Targets
        logger.info("PASO 12: Generando targets...")
        conn = generar_targets(conn, SQL_TABLE_NAME)
        
        # 13. Guardar resultado final
        logger.info("PASO 13: Guardando resultado final...")
        save_sql_table_to_parquet(conn, SQL_TABLE_NAME, OUTPUT_PATH_FE)
        
        logger.info("=== PIPELINE COMPLETADO EXITOSAMENTE ===")

    except Exception as e:
        logger.error(f"Error durante la ejecución del pipeline: {e}")
        raise

    finally:
        if conn:
            conn.close()
            logger.info("Conexión a DuckDB cerrada.")
        
        if db_file:
            cleanup_database(db_file)
        
        # Limpiar checkpoints
        import glob
        for checkpoint in glob.glob('/tmp/checkpoint_*.parquet'):
            try:
                os.remove(checkpoint)
                logger.info(f"Checkpoint eliminado: {checkpoint}")
            except:
                pass

if __name__ == "__main__":
    main()