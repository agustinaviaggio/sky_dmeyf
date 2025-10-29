import duckdb
import logging

logger = logging.getLogger("__name__")


def create_sql_table(path: str) -> duckdb.DuckDBPyConnection | None:
    '''
    Carga un CSV desde 'path' en una tabla DuckDB en memoria y retorna 
    el objeto de conexión para interactuar con esa tabla.
    '''
    logger.info(f"Cargando dataset desde {path}")
    conn = duckdb.connect(database=':memory:')
    try:        
        conn.execute(f"""
            CREATE OR REPLACE TABLE tabla_sql AS
            SELECT *
            FROM read_csv_auto('{path}')
        """)
        return conn
    
    except Exception as e:
        logger.error(f"Error al cargar el dataset: {e}")
        conn.close()
        raise

def atributes_to_intergers(conn: duckdb.DuckDBPyConnection, cols_to_alter: list[str])-> duckdb.DuckDBPyConnection:
    for col in cols_to_alter:
        conn.execute(f"""
            ALTER TABLE tabla_sql
            ALTER COLUMN {col} SET DATA TYPE INTEGER
        """)
    return conn

def drop_columns(conn: duckdb.DuckDBPyConnection, cols_to_drop: list[str])-> duckdb.DuckDBPyConnection:
    for col in cols_to_drop:
        conn.execute(f"ALTER TABLE tabla_sql DROP COLUMN {col}")
    return conn

def create_latest_and_earliest_credit_card_atributes(conn: duckdb.DuckDBPyConnection, cols_pairs: list[str])-> duckdb.DuckDBPyConnection:
    sql = """CREATE OR REPLACE TABLE tabla_sql AS
    SELECT *"""
    
    for col1, col2, prefix in cols_pairs:
        sql += f"CAST(greatest({col1}, {col2}) AS INTEGER) AS {prefix}_latest"
        sql += f"CAST(least({col1}, {col2}) AS INTEGER) AS {prefix}_earliest"

    sql += " FROM tabla_sql"
    
    conn.execute(sql)
    return conn

def create_sum_credit_card_atributes(conn: duckdb.DuckDBPyConnection, cols_visa: list[str], cols_master: list[str])-> duckdb.DuckDBPyConnection:
    conn.execute("""
        CREATE OR REPLACE MACRO suma_sin_null(a, b) AS (
            ifnull(a, 0) + ifnull(b, 0)
        )
    """)

    sql = """CREATE OR REPLACE TABLE tabla_sql AS
    SELECT *"""

    for v_col, m_col in zip(cols_visa, cols_master):
         if '_status' not in v_col:
            sufijo = v_col.replace("Visa_", "").replace("visa_", "").replace("Visa", "").replace("visa", "")
            sql += f"suma_sin_null({v_col},{m_col}) AS {sufijo}_tc"
    
    sql += "FROM tabla_sql"

    conn.execute(sql)    

    return conn

def crear_atributos_lag(conn: duckdb.DuckDBPyConnection, excluir_columnas: list[str], cant_lag: int = 1) -> duckdb.DuckDBPyConnection:
    """
    Genera variables de lag para los atributos especificados y reemplaza la tabla.
  
    Parameters:
    -----------
    conn : duckdb.DuckDBPyConnection
        Conexión a la tabla DuckDB con los datos.
    excluir_columnas : list
        Lista de atributos a excluir para los cuales generar lags.
    cant_lag : int, default=1
        Cantidad de lags a generar para cada atributo
  
    Returns:
    --------
    duckdb.DuckDBPyConnection
       Conexión a la tabla DuckDB con los datos con las variables de lag agregadas.
    """

    logger.info(f"Realizando feature engineering con {cant_lag} lags para todos los atributos con excepción de las variables con tipo de de dato INTERGER o VARCHAR y los {len(excluir_columnas)} atributos excluídos explicitamente según la lista {excluir_columnas}")

    sql_get_cols = """
            SELECT 
                column_name 
            FROM 
                (DESCRIBE tabla_sql)
            WHERE 
                column_type NOT IN ('INTEGER', 'VARCHAR')
        """
    
    cols_numericas_list = conn.execute(sql_get_cols).fetchall()
    cols_numericas = [c[0] for c in cols_numericas_list if c[0] not in excluir_columnas]

    logger.info(f"Se generarán {cant_lag} lags para {len(cols_numericas)} columnas.")

    if not cols_numericas:
        logger.warning("No se encontraron columnas numéricas válidas para generar lags. Devolviendo la conexión sin cambios.")
        return conn
  
    # Construir la consulta SQL
    sql = """CREATE OR REPLACE TABLE tabla_sql AS
        SELECT *"""
  
    # Agregar los lags para los atributos especificados
    for attr in cols_numericas:
        for i in range(1, cant_lag + 1):
            sql += f", lag({attr}, {i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS {attr}_lag_{i}"
  
    # Completar la consulta
    sql += " FROM tabla_sql"

    logger.debug(f"Consulta SQL: {sql}")

    # Ejecutar la consulta SQL
    conn.execute(sql)

    new_schema_df = conn.execute("DESCRIBE tabla_sql").fetchall()
    new_num_cols = len(new_schema_df)

    logger.info(f"Feature engineering completado. La tabla resultante tiene {new_num_cols} columnas")

    return conn