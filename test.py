import duckdb

# --- CONFIGURACIÓN ---
CSV_PATH = 'data/datasets_competencia_02_crudo.csv.gz'

# --- CONEXIÓN Y CARGA DE DATOS ---
con = duckdb.connect()
con.execute(f"""
    CREATE OR REPLACE TABLE datos AS
    SELECT * FROM read_csv_auto('{CSV_PATH}', HEADER=TRUE);
""")

# Análisis de los meses problemáticos
print("=== ANÁLISIS DETALLADO DE mrentabilidad_annual ===\n")

meses_problema = [201905, 201910, 202006]
meses_normales = [201904, 201909, 202005]

for i, mes in enumerate(meses_problema):
    mes_normal = meses_normales[i]
    
    print(f"\n{'='*60}")
    print(f"COMPARACIÓN: {mes_normal} (normal) vs {mes} (problema)")
    print('='*60)
    
    df = con.execute(f"""
        SELECT 
            foto_mes,
            COUNT(*) as total_registros,
            COUNT(mrentabilidad_annual) as valores_no_nulos,
            SUM(CASE WHEN mrentabilidad_annual = 0 THEN 1 ELSE 0 END) as cantidad_ceros,
            SUM(CASE WHEN mrentabilidad_annual IS NULL THEN 1 ELSE 0 END) as cantidad_nulos,
            MIN(mrentabilidad_annual) as minimo,
            MAX(mrentabilidad_annual) as maximo,
            AVG(mrentabilidad_annual) as media,
            AVG(CASE WHEN mrentabilidad_annual != 0 THEN mrentabilidad_annual END) as media_sin_ceros
        FROM datos
        WHERE foto_mes IN ({mes_normal}, {mes})
        GROUP BY foto_mes
        ORDER BY foto_mes
    """).fetchdf()
    
    print(df.to_string(index=False))
    
    # Distribución de valores en el mes problema
    print(f"\nDistribución de valores en {mes}:")
    df_dist = con.execute(f"""
        SELECT 
            CASE 
                WHEN mrentabilidad_annual IS NULL THEN 'NULL'
                WHEN mrentabilidad_annual = 0 THEN '0'
                WHEN mrentabilidad_annual > 0 AND mrentabilidad_annual <= 10000 THEN '0-10k'
                WHEN mrentabilidad_annual > 10000 AND mrentabilidad_annual <= 20000 THEN '10k-20k'
                WHEN mrentabilidad_annual > 20000 AND mrentabilidad_annual <= 30000 THEN '20k-30k'
                ELSE '>30k'
            END as rango,
            COUNT(*) as cantidad
        FROM datos
        WHERE foto_mes = {mes}
        GROUP BY rango
        ORDER BY rango
    """).fetchdf()
    print(df_dist.to_string(index=False))