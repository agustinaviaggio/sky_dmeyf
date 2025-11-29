import duckdb

parquet_path = r"C:\Users\Home\Desktop\repos\sky_dmeyf\2711_2\ensemble_final_2_final_submission_probabilidades_202108.parquet"

# Obtener la probabilidad en el puesto 11000 para cada columna de probabilidades
query = f"""
WITH ranked AS (
    SELECT 
        proba_1311_7,
        proba_1411_2,
        proba_1411_3,
        proba_1511_1,
        proba_1511_2,
        proba_1511_3,
        probabilidad_ensemble,
        ROW_NUMBER() OVER (ORDER BY proba_1311_7 DESC) as rank_1311_7,
        ROW_NUMBER() OVER (ORDER BY proba_1411_2 DESC) as rank_1411_2,
        ROW_NUMBER() OVER (ORDER BY proba_1411_3 DESC) as rank_1411_3,
        ROW_NUMBER() OVER (ORDER BY proba_1511_1 DESC) as rank_1511_1,
        ROW_NUMBER() OVER (ORDER BY proba_1511_2 DESC) as rank_1511_2,
        ROW_NUMBER() OVER (ORDER BY proba_1511_3 DESC) as rank_1511_3,
        ROW_NUMBER() OVER (ORDER BY probabilidad_ensemble DESC) as rank_ensemble
    FROM '{parquet_path}'
)
SELECT 
    MAX(CASE WHEN rank_1311_7 = 11000 THEN proba_1311_7 END) as umbral_1311_7,
    MAX(CASE WHEN rank_1411_2 = 11000 THEN proba_1411_2 END) as umbral_1411_2,
    MAX(CASE WHEN rank_1411_3 = 11000 THEN proba_1411_3 END) as umbral_1411_3,
    MAX(CASE WHEN rank_1511_1 = 11000 THEN proba_1511_1 END) as umbral_1511_1,
    MAX(CASE WHEN rank_1511_2 = 11000 THEN proba_1511_2 END) as umbral_1511_2,
    MAX(CASE WHEN rank_1511_3 = 11000 THEN proba_1511_3 END) as umbral_1511_3,
    MAX(CASE WHEN rank_ensemble = 11000 THEN probabilidad_ensemble END) as umbral_ensemble
FROM ranked
"""

result = duckdb.query(query).df()

print("=== UMBRAL EN PUESTO 11000 PARA CADA MODELO ===")
for col in result.columns:
    model_name = col.replace('umbral_', '')
    print(f"{model_name:25s}: {result[col].values[0]:.6f}")