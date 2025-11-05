import duckdb
import pandas as pd

CSV_PATH = 'data/datasets_competencia_02_crudo.csv.gz'

con = duckdb.connect()
con.execute(f"""
    CREATE OR REPLACE TABLE datos AS
    SELECT * FROM read_csv_auto('{CSV_PATH}', HEADER=TRUE);
""")

# Analizar percentiles para entender la distribución
df_percentiles = con.execute("""
    SELECT 
        foto_mes,
        COUNT(*) as n,
        APPROX_QUANTILE(mrentabilidad, 0.10) as p10,
        APPROX_QUANTILE(mrentabilidad, 0.25) as p25,
        APPROX_QUANTILE(mrentabilidad, 0.50) as p50,
        APPROX_QUANTILE(mrentabilidad, 0.75) as p75,
        APPROX_QUANTILE(mrentabilidad, 0.90) as p90,
        AVG(mrentabilidad) as media,
        STDDEV(mrentabilidad) as std
    FROM datos
    WHERE foto_mes BETWEEN 202008 AND 202011
    GROUP BY foto_mes
    ORDER BY foto_mes
""").fetchdf()

print("=== Percentiles de mrentabilidad 202008-202011 ===")
print(df_percentiles.to_string(index=False))

# Calcular el rango intercuartílico (IQR)
df_percentiles['IQR'] = df_percentiles['p75'] - df_percentiles['p25']
df_percentiles['rango'] = df_percentiles['p90'] - df_percentiles['p10']

print("\n=== Métricas de dispersión ===")
print(df_percentiles[['foto_mes', 'IQR', 'rango', 'std']].to_string(index=False))