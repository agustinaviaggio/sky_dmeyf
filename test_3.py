import duckdb

CSV_PATH = 'data/datasets_competencia_02_crudo.csv.gz'

con = duckdb.connect()
con.execute(f"""
    CREATE OR REPLACE TABLE datos AS
    SELECT * FROM read_csv_auto('{CSV_PATH}', HEADER=TRUE);
""")

# Contar registros por mes
df_registros = con.execute("""
    SELECT 
        foto_mes,
        COUNT(*) as total_clientes,
        COUNT(mrentabilidad) as con_mrentabilidad,
        AVG(mrentabilidad) as media_mrent,
        STDDEV(mrentabilidad) as std_mrent
    FROM datos
    GROUP BY foto_mes
    ORDER BY foto_mes
""").fetchdf()

print(df_registros)

# Ver específicamente alrededor de 202009
print("\n=== Zoom en 202008-202010 ===")
print(df_registros[df_registros['foto_mes'].isin([202008, 202009, 202010])])