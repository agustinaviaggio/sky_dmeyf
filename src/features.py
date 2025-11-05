import duckdb
import logging

logger = logging.getLogger(__name__)


def create_sql_table(path: str, table_name: str) -> duckdb.DuckDBPyConnection:
    '''
    Carga un CSV desde 'path' en una tabla DuckDB en memoria y retorna 
    el objeto de conexión para interactuar con esa tabla.
    '''
    logger.info(f"Cargando dataset desde {path}")
    conn = duckdb.connect(database=':memory:')
    try:        
        conn.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT *
            FROM read_csv_auto('{path}')
        """)
        return conn
    
    except Exception as e:
        logger.error(f"Error al cargar el dataset: {e}")
        conn.close()
        raise

def attributes_to_intergers(conn: duckdb.DuckDBPyConnection, table_name: str, cols_to_alter: list[str])-> duckdb.DuckDBPyConnection:
    for col in cols_to_alter:
        conn.execute(f"""
            ALTER TABLE {table_name}
            ALTER COLUMN {col} SET DATA TYPE INTEGER
        """)
    return conn

def drop_columns(conn: duckdb.DuckDBPyConnection, table_name: str, cols_to_drop: list[str])-> duckdb.DuckDBPyConnection:
    for col in cols_to_drop:
        conn.execute(f"ALTER TABLE {table_name} DROP COLUMN {col}")
    return conn

def create_latest_and_earliest_credit_card_attributes(conn: duckdb.DuckDBPyConnection, table_name: str, cols_pairs: list[str]) -> duckdb.DuckDBPyConnection:
    # 1. Crear la lista de expresiones SQL para las nuevas columnas
    new_cols_sql = []
    for col1, col2, prefix in cols_pairs:
        # Usamos format string para crear la expresión de cada columna
        latest_expr = f"CAST(greatest({col1}, {col2}) AS INTEGER) AS {prefix}_latest"
        earliest_expr = f"CAST(least({col1}, {col2}) AS INTEGER) AS {prefix}_earliest"
        new_cols_sql.append(latest_expr)
        new_cols_sql.append(earliest_expr)

    # 2. Unir las expresiones con comas
    new_cols_str = ", " + ", ".join(new_cols_sql)

    # 3. Construir la sentencia SQL completa de forma limpia
    sql = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT 
            * {new_cols_str}
        FROM 
            {table_name}
    """
    
    conn.execute(sql)
    return conn

def create_sum_credit_card_attributes(conn: duckdb.DuckDBPyConnection, table_name: str, cols_visa: list, cols_master: list)-> duckdb.DuckDBPyConnection:
    conn.execute("""
        CREATE OR REPLACE MACRO suma_sin_null(a, b) AS (
            ifnull(a, 0) + ifnull(b, 0)
        )
    """)

    new_cols_sql = []
    
    for v_col, m_col in zip(cols_visa, cols_master):
         if '_status' not in v_col:
            sufijo = v_col.replace("Visa_", "").replace("visa_", "").replace("Visa", "").replace("visa", "")
            new_cols_sql.append(f"suma_sin_null({v_col},{m_col}) AS {sufijo}_tc")
    
    new_cols_str = ", " + ", ".join(new_cols_sql) if new_cols_sql else ""

    sql = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT * {new_cols_str}
        FROM {table_name}
    """

    conn.execute(sql)    

    return conn

def create_ratio_m_c_attributes(conn: duckdb.DuckDBPyConnection, table_name: str)-> duckdb.DuckDBPyConnection:
    conn.execute("""
        CREATE OR REPLACE MACRO ratio(a, b) AS (a // (b + 0.01))
    """)

    sql_get_cols = f"""
            SELECT 
                name 
            FROM 
                 pragma_table_info('{table_name}')
        """
    
    columnas_tuples = conn.execute(sql_get_cols).fetchall()
    columnas = [c[0] for c in columnas_tuples]

    # Separar columnas que empiezan con 'm' y 'c'
    cols_m = [c for c in columnas if c.startswith('m')]
    cols_c = [c for c in columnas if c.startswith('c')]

    # Crear diccionario de sufijo a columna
    sufijo_a_m = {c[1:]: c for c in cols_m}  # clave = sufijo sin la primera letra
    sufijo_a_c = {c[1:]: c for c in cols_c}

    # Generar pares donde el sufijo coincide
    pares = [(sufijo_a_m[s], sufijo_a_c[s]) for s in sufijo_a_m if s in sufijo_a_c]

    new_cols_sql = []
    for a, b in pares:
        new_cols_sql.append(f"ratio({a},{b}) AS ratio_{a}_{b}")
    
    new_cols_str = ", " + ", ".join(new_cols_sql) if new_cols_sql else ""

    sql = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT * {new_cols_str}
        FROM {table_name}
    """

    conn.execute(sql)    

    return conn

def create_lag_attributes(conn: duckdb.DuckDBPyConnection, table_name: str, excluir_columnas: list[str], cant_lag: int = 1) -> duckdb.DuckDBPyConnection:
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

    sql_get_cols = f"""
            SELECT 
                name 
            FROM 
                pragma_table_info('{table_name}')
            WHERE 
                type NOT IN ('INTEGER', 'VARCHAR')
        """
    
    cols_numericas_list = conn.execute(sql_get_cols).fetchall()
    cols_numericas = [c[0] for c in cols_numericas_list if c[0] not in excluir_columnas]

    logger.info(f"Se generarán {cant_lag} lags para {len(cols_numericas)} columnas.")

    if not cols_numericas:
        logger.warning("No se encontraron columnas numéricas válidas para generar lags. Devolviendo la conexión sin cambios.")
        return conn
  
    new_cols_sql = []
    for attr in cols_numericas:
        for i in range(1, cant_lag + 1):
            # Agregamos la expresión a la lista
            new_cols_sql.append(f"lag({attr}, {i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS {attr}_lag_{i}")
  
    new_cols_str = ", " + ", ".join(new_cols_sql)

    # Construir la consulta SQL
    sql = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT * {new_cols_str}
        FROM {table_name}
    """

    logger.debug(f"Consulta SQL: {sql}")

    # Ejecutar la consulta SQL
    conn.execute(sql)
    return conn

def create_delta_attributes(conn: duckdb.DuckDBPyConnection, table_name: str, excluir_columnas: list[str], cant_delta: int = 1) -> duckdb.DuckDBPyConnection:
    """
    Genera variables de delta para los atributos especificados y reemplaza la tabla.
  
    Parameters:
    -----------
    conn : duckdb.DuckDBPyConnection
        Conexión a la tabla DuckDB con los datos.
    excluir_columnas : list
        Lista de atributos a excluir para los cuales generar deltas.
    cant_lag : int, default=1
        Cantidad de lags a generar para cada atributo
  
    Returns:
    --------
    duckdb.DuckDBPyConnection
       Conexión a la tabla DuckDB con los datos con las variables de delta agregadas.
    """

    logger.info(f"Realizando feature engineering con {cant_delta} deltas para todos los atributos con excepción de las variables con tipo de de dato INTERGER o VARCHAR y los {len(excluir_columnas)} atributos excluídos explicitamente según la lista {excluir_columnas}")

    sql_get_cols = f"""
            SELECT 
                name 
            FROM 
                pragma_table_info('{table_name}')
            WHERE 
                type NOT IN ('INTEGER', 'VARCHAR')
        """
    
    cols_numericas_list = conn.execute(sql_get_cols).fetchall()
    cols_numericas = [c[0] for c in cols_numericas_list if c[0] not in excluir_columnas]

    logger.info(f"Se generarán {cant_delta} deltas para {len(cols_numericas)} columnas.")

    if not cols_numericas:
        logger.warning("No se encontraron columnas numéricas válidas para generar deltas. Devolviendo la conexión sin cambios.")
        return conn
    
    new_cols_sql = []
    for attr in cols_numericas:
        for i in range(1, cant_delta + 1):
            new_cols_sql.append(f"{attr} - lag({attr}, {i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS {attr}_delta_{i}")
  
    new_cols_str = ", " + ", ".join(new_cols_sql)

    # Construir la consulta SQL
    sql = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT * {new_cols_str}
        FROM {table_name}
    """

    logger.debug(f"Consulta SQL: {sql}")

    # Ejecutar la consulta SQL
    conn.execute(sql)
    return conn

def create_max_attributes(conn: duckdb.DuckDBPyConnection, table_name: str, excluir_columnas: list[str], month_window: int = 1) -> duckdb.DuckDBPyConnection:
    """
    Genera variables de valores máximos por ventana temporal para los atributos especificados y reemplaza la tabla.
  
    Parameters:
    -----------
    conn : duckdb.DuckDBPyConnection
        Conexión a la tabla DuckDB con los datos.
    excluir_columnas : list
        Lista de atributos a excluir para los cuales generar máximos.
    month_window: int, default=1
        Cantidad de meses de la ventana temporal.
  
    Returns:
    --------
    duckdb.DuckDBPyConnection
       Conexión a la tabla DuckDB con los datos con las variables de máximos agregadas.
    """

    logger.info(f"Realizando feature engineering con valores máximos con {month_window} meses de ventana temporal para todos los atributos con excepción de las variables con tipo de de dato INTERGER o VARCHAR y los {len(excluir_columnas)} atributos excluídos explicitamente según la lista {excluir_columnas}")

    sql_get_cols = f"""
            SELECT 
                name 
            FROM 
                pragma_table_info('{table_name}')
            WHERE 
                type NOT IN ('INTEGER', 'VARCHAR')
        """
    
    cols_numericas_list = conn.execute(sql_get_cols).fetchall()
    cols_numericas = [c[0] for c in cols_numericas_list if c[0] not in excluir_columnas]

    logger.info(f"Se generarán máximos con ventana temporal para {len(cols_numericas)} columnas.")

    if not cols_numericas:
        logger.warning("No se encontraron columnas numéricas válidas para generar máximos. Devolviendo la conexión sin cambios.")
        return conn
    
    new_cols_sql = []
    for attr in cols_numericas:
        new_cols_sql.append(f"max({attr}) OVER w AS {attr}_max_{month_window}")
    new_cols_str = ", " + ", ".join(new_cols_sql) if new_cols_sql else ""
  
    # Construir la consulta SQL
    sql = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT * {new_cols_str}
        FROM {table_name} 
        WINDOW w AS (
            PARTITION BY numero_de_cliente
            ORDER BY foto_mes
            ROWS BETWEEN {month_window} PRECEDING AND CURRENT ROW
        )
    """

    logger.debug(f"Consulta SQL: {sql}")

    # Ejecutar la consulta SQL
    conn.execute(sql)

    return conn

'''def create_min_attributes(conn: duckdb.DuckDBPyConnection, table_name: str, excluir_columnas: list[str], month_window: int = 1) -> duckdb.DuckDBPyConnection:
    """
    Genera variables de valores mínimos por ventana temporal para los atributos especificados y reemplaza la tabla.
  
    Parameters:
    -----------
    conn : duckdb.DuckDBPyConnection
        Conexión a la tabla DuckDB con los datos.
    excluir_columnas : list
        Lista de atributos a excluir para los cuales generar mínimos.
    month_window: int, default=1
        Cantidad de meses de la ventana temporal.
  
    Returns:
    --------
    duckdb.DuckDBPyConnection
       Conexión a la tabla DuckDB con los datos con las variables de mínimos agregadas.
    """

    logger.info(f"Realizando feature engineering con valores mínimos con {month_window} meses de ventana temporal para todos los atributos con excepción de las variables con tipo de de dato INTERGER o VARCHAR y los {len(excluir_columnas)} atributos excluídos explicitamente según la lista {excluir_columnas}")

    sql_get_cols = f"""
            SELECT 
                name 
            FROM 
                pragma_table_info('{table_name}')
            WHERE 
                type NOT IN ('INTEGER', 'VARCHAR')
        """
    
    cols_numericas_list = conn.execute(sql_get_cols).fetchall()
    cols_numericas = [c[0] for c in cols_numericas_list if c[0] not in excluir_columnas]

    logger.info(f"Se generarán mínimos con ventana temporal para {len(cols_numericas)} columnas.")

    if not cols_numericas:
        logger.warning("No se encontraron columnas numéricas válidas para generar mínimos. Devolviendo la conexión sin cambios.")
        return conn
  
    new_cols_sql = []
    for attr in cols_numericas:
        new_cols_sql.append(f"min({attr}) OVER w AS {attr}_min_{month_window}")
    new_cols_str = ", " + ", ".join(new_cols_sql) if new_cols_sql else ""
  
    # Construir la consulta SQL
    sql = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT * {new_cols_str}
        FROM {table_name} 
        WINDOW w AS (
            PARTITION BY numero_de_cliente
            ORDER BY foto_mes
            ROWS BETWEEN {month_window} PRECEDING AND CURRENT ROW
        )
    """

    logger.debug(f"Consulta SQL: {sql}")

    # Ejecutar la consulta SQL
    conn.execute(sql)

    return conn

def create_avg_attributes(conn: duckdb.DuckDBPyConnection, table_name: str, excluir_columnas: list[str], month_window: int = 1) -> duckdb.DuckDBPyConnection:
    """
    Genera variables de valores promedios por ventana temporal para los atributos especificados y reemplaza la tabla.
  
    Parameters:
    -----------
    conn : duckdb.DuckDBPyConnection
        Conexión a la tabla DuckDB con los datos.
    excluir_columnas : list
        Lista de atributos a excluir para los cuales generar promedios.
    month_window: int, default=1
        Cantidad de meses de la ventana temporal.
  
    Returns:
    --------
    duckdb.DuckDBPyConnection
       Conexión a la tabla DuckDB con los datos con las variables de promedios agregadas.
    """

    logger.info(f"Realizando feature engineering con valores promedios con {month_window} meses de ventana temporal para todos los atributos con excepción de las variables con tipo de de dato INTERGER o VARCHAR y los {len(excluir_columnas)} atributos excluídos explicitamente según la lista {excluir_columnas}")

    sql_get_cols = f"""
            SELECT 
                name 
            FROM 
                pragma_table_info('{table_name}')
            WHERE 
                type NOT IN ('INTEGER', 'VARCHAR')
        """
    
    cols_numericas_list = conn.execute(sql_get_cols).fetchall()
    cols_numericas = [c[0] for c in cols_numericas_list if c[0] not in excluir_columnas]

    logger.info(f"Se generarán promedios con ventana temporal para {len(cols_numericas)} columnas.")

    if not cols_numericas:
        logger.warning("No se encontraron columnas numéricas válidas para generar promedios. Devolviendo la conexión sin cambios.")
        return conn
  
    new_cols_sql = []
    for attr in cols_numericas:
        new_cols_sql.append(f"avg({attr}) OVER w AS {attr}_avg_{month_window}")
    
    new_cols_str = ", " + ", ".join(new_cols_sql) if new_cols_sql else ""

    sql = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT * {new_cols_str}
        FROM {table_name} 
        WINDOW w AS (
            PARTITION BY numero_de_cliente
            ORDER BY foto_mes
            ROWS BETWEEN {month_window} PRECEDING AND CURRENT ROW
        )
    """

    logger.debug(f"Consulta SQL: {sql}")

    # Ejecutar la consulta SQL
    conn.execute(sql)
    return conn'''

def save_sql_table_to_csv(conn: duckdb.DuckDBPyConnection, table_name: str, path: str) -> None:
    '''
    Guarda la tabla sql en CSV a 'path'.
    '''
    logger.info(f"Guardando tabla en {path}")
    conn.execute(f"""
    COPY {table_name} TO '{path}' (FORMAT CSV, HEADER TRUE)
    """)
    
def create_max_attributes(conn: duckdb.DuckDBPyConnection, table_name: str, excluir_columnas: list[str], month_window: int = 1) -> duckdb.DuckDBPyConnection:
    """
    Genera variables de valores máximos por ventana temporal para los atributos especificados y reemplaza la tabla.
    Optimizado usando tabla temporal para mejor performance.
  
    Parameters:
    -----------
    conn : duckdb.DuckDBPyConnection
        Conexión a la tabla DuckDB con los datos.
    excluir_columnas : list
        Lista de atributos a excluir para los cuales generar máximos.
    month_window: int, default=1
        Cantidad de meses de la ventana temporal.
  
    Returns:
    --------
    duckdb.DuckDBPyConnection
       Conexión a la tabla DuckDB con los datos con las variables de máximos agregadas.
    """

    logger.info(f"Realizando feature engineering con valores máximos con {month_window} meses de ventana temporal para todos los atributos con excepción de las variables con tipo de dato INTEGER o VARCHAR y los {len(excluir_columnas)} atributos excluídos explícitamente")

    sql_get_cols = f"""
        SELECT 
            name 
        FROM 
            pragma_table_info('{table_name}')
        WHERE 
            type NOT IN ('INTEGER', 'VARCHAR')
    """
    
    cols_numericas_list = conn.execute(sql_get_cols).fetchall()
    cols_numericas = [c[0] for c in cols_numericas_list if c[0] not in excluir_columnas]

    logger.info(f"Se generarán máximos con ventana temporal para {len(cols_numericas)} columnas.")

    if not cols_numericas:
        logger.warning("No se encontraron columnas numéricas válidas para generar máximos. Devolviendo la conexión sin cambios.")
        return conn
    
    # Generar expresiones SQL para las nuevas columnas
    new_cols_sql = []
    for attr in cols_numericas:
        new_cols_sql.append(f"max({attr}) OVER w AS {attr}_max_{month_window}")
    
    new_cols_str = ", ".join(new_cols_sql)
    
    # Paso 1: Crear tabla temporal solo con keys y las nuevas features
    logger.info("Creando tabla temporal con features de máximos...")
    sql_temp = f"""
        CREATE TEMP TABLE IF NOT EXISTS temp_max AS
        SELECT 
            numero_de_cliente,
            foto_mes,
            {new_cols_str}
        FROM {table_name} 
        WINDOW w AS (
            PARTITION BY numero_de_cliente
            ORDER BY foto_mes
            ROWS BETWEEN {month_window} PRECEDING AND CURRENT ROW
        )
    """
    
    conn.execute(sql_temp)
    
    # Paso 2: Join con la tabla original
    logger.info("Realizando join con tabla original...")
    sql_join = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT t.*, temp_max.* EXCLUDE (numero_de_cliente, foto_mes)
        FROM {table_name} t
        JOIN temp_max 
        ON t.numero_de_cliente = temp_max.numero_de_cliente 
        AND t.foto_mes = temp_max.foto_mes
    """
    
    conn.execute(sql_join)
    
    # Limpiar tabla temporal
    conn.execute("DROP TABLE IF EXISTS temp_max")
    logger.info("Features de máximos agregadas exitosamente.")

    return conn


def create_min_attributes(conn: duckdb.DuckDBPyConnection, table_name: str, excluir_columnas: list[str], month_window: int = 1) -> duckdb.DuckDBPyConnection:
    """
    Genera variables de valores mínimos por ventana temporal para los atributos especificados y reemplaza la tabla.
    Optimizado usando tabla temporal para mejor performance.
  
    Parameters:
    -----------
    conn : duckdb.DuckDBPyConnection
        Conexión a la tabla DuckDB con los datos.
    excluir_columnas : list
        Lista de atributos a excluir para los cuales generar mínimos.
    month_window: int, default=1
        Cantidad de meses de la ventana temporal.
  
    Returns:
    --------
    duckdb.DuckDBPyConnection
       Conexión a la tabla DuckDB con los datos con las variables de mínimos agregadas.
    """

    logger.info(f"Realizando feature engineering con valores mínimos con {month_window} meses de ventana temporal para todos los atributos con excepción de las variables con tipo de dato INTEGER o VARCHAR y los {len(excluir_columnas)} atributos excluídos explícitamente")

    sql_get_cols = f"""
        SELECT 
            name 
        FROM 
            pragma_table_info('{table_name}')
        WHERE 
            type NOT IN ('INTEGER', 'VARCHAR')
    """
    
    cols_numericas_list = conn.execute(sql_get_cols).fetchall()
    cols_numericas = [c[0] for c in cols_numericas_list if c[0] not in excluir_columnas]

    logger.info(f"Se generarán mínimos con ventana temporal para {len(cols_numericas)} columnas.")

    if not cols_numericas:
        logger.warning("No se encontraron columnas numéricas válidas para generar mínimos. Devolviendo la conexión sin cambios.")
        return conn
    
    # Generar expresiones SQL para las nuevas columnas
    new_cols_sql = []
    for attr in cols_numericas:
        new_cols_sql.append(f"min({attr}) OVER w AS {attr}_min_{month_window}")
    
    new_cols_str = ", ".join(new_cols_sql)
    
    # Paso 1: Crear tabla temporal solo con keys y las nuevas features
    logger.info("Creando tabla temporal con features de mínimos...")
    sql_temp = f"""
        CREATE TEMP TABLE IF NOT EXISTS temp_min AS
        SELECT 
            numero_de_cliente,
            foto_mes,
            {new_cols_str}
        FROM {table_name} 
        WINDOW w AS (
            PARTITION BY numero_de_cliente
            ORDER BY foto_mes
            ROWS BETWEEN {month_window} PRECEDING AND CURRENT ROW
        )
    """
    
    conn.execute(sql_temp)
    
    # Paso 2: Join con la tabla original
    logger.info("Realizando join con tabla original...")
    sql_join = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT t.*, temp_min.* EXCLUDE (numero_de_cliente, foto_mes)
        FROM {table_name} t
        JOIN temp_min 
        ON t.numero_de_cliente = temp_min.numero_de_cliente 
        AND t.foto_mes = temp_min.foto_mes
    """
    
    conn.execute(sql_join)
    
    # Limpiar tabla temporal
    conn.execute("DROP TABLE IF EXISTS temp_min")
    logger.info("Features de mínimos agregadas exitosamente.")

    return conn


def create_avg_attributes(conn: duckdb.DuckDBPyConnection, table_name: str, excluir_columnas: list[str], month_window: int = 1) -> duckdb.DuckDBPyConnection:
    """
    Genera variables de valores promedios por ventana temporal para los atributos especificados y reemplaza la tabla.
    Optimizado usando tabla temporal para mejor performance.
  
    Parameters:
    -----------
    conn : duckdb.DuckDBPyConnection
        Conexión a la tabla DuckDB con los datos.
    excluir_columnas : list
        Lista de atributos a excluir para los cuales generar promedios.
    month_window: int, default=1
        Cantidad de meses de la ventana temporal.
  
    Returns:
    --------
    duckdb.DuckDBPyConnection
       Conexión a la tabla DuckDB con los datos con las variables de promedios agregadas.
    """

    logger.info(f"Realizando feature engineering con valores promedios con {month_window} meses de ventana temporal para todos los atributos con excepción de las variables con tipo de dato INTEGER o VARCHAR y los {len(excluir_columnas)} atributos excluídos explícitamente")

    sql_get_cols = f"""
        SELECT 
            name 
        FROM 
            pragma_table_info('{table_name}')
        WHERE 
            type NOT IN ('INTEGER', 'VARCHAR')
    """
    
    cols_numericas_list = conn.execute(sql_get_cols).fetchall()
    cols_numericas = [c[0] for c in cols_numericas_list if c[0] not in excluir_columnas]

    logger.info(f"Se generarán promedios con ventana temporal para {len(cols_numericas)} columnas.")

    if not cols_numericas:
        logger.warning("No se encontraron columnas numéricas válidas para generar promedios. Devolviendo la conexión sin cambios.")
        return conn
    
    # Generar expresiones SQL para las nuevas columnas
    new_cols_sql = []
    for attr in cols_numericas:
        new_cols_sql.append(f"avg({attr}) OVER w AS {attr}_avg_{month_window}")
    
    new_cols_str = ", ".join(new_cols_sql)
    
    # Paso 1: Crear tabla temporal solo con keys y las nuevas features
    logger.info("Creando tabla temporal con features de promedios...")
    sql_temp = f"""
        CREATE TEMP TABLE IF NOT EXISTS temp_avg AS
        SELECT 
            numero_de_cliente,
            foto_mes,
            {new_cols_str}
        FROM {table_name} 
        WINDOW w AS (
            PARTITION BY numero_de_cliente
            ORDER BY foto_mes
            ROWS BETWEEN {month_window} PRECEDING AND CURRENT ROW
        )
    """
    
    conn.execute(sql_temp)
    
    # Paso 2: Join con la tabla original
    logger.info("Realizando join con tabla original...")
    sql_join = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT t.*, temp_avg.* EXCLUDE (numero_de_cliente, foto_mes)
        FROM {table_name} t
        JOIN temp_avg 
        ON t.numero_de_cliente = temp_avg.numero_de_cliente 
        AND t.foto_mes = temp_avg.foto_mes
    """
    
    conn.execute(sql_join)
    
    # Limpiar tabla temporal
    conn.execute("DROP TABLE IF EXISTS temp_avg")
    logger.info("Features de promedios agregadas exitosamente.")

    return conn

def save_sql_table_to_parquet(conn: duckdb.DuckDBPyConnection, table_name: str, path: str) -> None:
    '''
    Guarda la tabla sql en formato Parquet (mucho más eficiente que CSV).
    '''
    logger.info(f"Guardando tabla en formato Parquet a {path}")
    
    # Asegurar que la extensión sea .parquet
    if not path.endswith('.parquet'):
        if path.endswith('.csv'):
            path = path.replace('.csv', '.parquet')
        else:
            path = path + '.parquet'
    
    conn.execute(f"""
        COPY {table_name} TO '{path}' 
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
    """)
    
    logger.info(f"Tabla guardada exitosamente en {path}")

