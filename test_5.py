import duckdb
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = 'data/datasets_competencia_02_crudo.csv.gz'

con = duckdb.connect()
con.execute(f"""
    CREATE OR REPLACE TABLE datos AS
    SELECT * FROM read_csv_auto('{CSV_PATH}', HEADER=TRUE);
""")

# Analizar TODO el período COVID: marzo 2020 - marzo 2021
df_percentiles = con.execute("""
    SELECT 
        foto_mes,
        COUNT(*) as n_clientes,
        APPROX_QUANTILE(mrentabilidad, 0.10) as p10,
        APPROX_QUANTILE(mrentabilidad, 0.25) as p25,
        APPROX_QUANTILE(mrentabilidad, 0.50) as p50,
        APPROX_QUANTILE(mrentabilidad, 0.75) as p75,
        APPROX_QUANTILE(mrentabilidad, 0.90) as p90,
        AVG(mrentabilidad) as media,
        STDDEV(mrentabilidad) as std
    FROM datos
    WHERE foto_mes BETWEEN 202003 AND 202103
      AND foto_mes NOT IN (202006)  -- Excluir mes con datos erróneos
    GROUP BY foto_mes
    ORDER BY foto_mes
""").fetchdf()

print("=== ANÁLISIS COMPLETO: Marzo 2020 - Marzo 2021 ===\n")
print(df_percentiles.to_string(index=False))

# Calcular cambios relativos desde marzo 2020 (baseline)
baseline = df_percentiles[df_percentiles['foto_mes'] == 202003].iloc[0]

print(f"\n=== CAMBIOS RELATIVOS vs MARZO 2020 (baseline) ===")
print(f"Baseline - Marzo 2020:")
print(f"  P25: {baseline['p25']:.2f}")
print(f"  P50: {baseline['p50']:.2f}")
print(f"  P75: {baseline['p75']:.2f}")
print(f"  Media: {baseline['media']:.2f}\n")

for idx, row in df_percentiles.iterrows():
    if row['foto_mes'] == 202003:
        continue
    print(f"{row['foto_mes']}:")
    print(f"  P25: {row['p25']:7.2f} ({(row['p25']/baseline['p25']-1)*100:+6.1f}%)")
    print(f"  P50: {row['p50']:7.2f} ({(row['p50']/baseline['p50']-1)*100:+6.1f}%)")
    print(f"  P75: {row['p75']:7.2f} ({(row['p75']/baseline['p75']-1)*100:+6.1f}%)")
    print(f"  Media: {row['media']:7.2f} ({(row['media']/baseline['media']-1)*100:+6.1f}%)")
    print()

# Graficar la evolución
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Panel 1: Percentiles absolutos
ax1 = axes[0]
ax1.plot(df_percentiles['foto_mes'].astype(str), df_percentiles['p10'], marker='o', label='P10')
ax1.plot(df_percentiles['foto_mes'].astype(str), df_percentiles['p25'], marker='s', label='P25')
ax1.plot(df_percentiles['foto_mes'].astype(str), df_percentiles['p50'], marker='^', label='P50 (Mediana)')
ax1.plot(df_percentiles['foto_mes'].astype(str), df_percentiles['p75'], marker='D', label='P75')
ax1.plot(df_percentiles['foto_mes'].astype(str), df_percentiles['p90'], marker='v', label='P90')
ax1.plot(df_percentiles['foto_mes'].astype(str), df_percentiles['media'], marker='*', linewidth=2, markersize=10, label='Media')
ax1.axvline(x='202003', color='red', linestyle='--', alpha=0.3, label='Inicio Cuarentena')
ax1.set_xlabel('Mes')
ax1.set_ylabel('Mrentabilidad')
ax1.set_title('Evolución de Percentiles - mrentabilidad (Mar 2020 - Mar 2021)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# Panel 2: Cambio relativo vs baseline
ax2 = axes[1]
cambio_p25 = ((df_percentiles['p25'] / baseline['p25']) - 1) * 100
cambio_p50 = ((df_percentiles['p50'] / baseline['p50']) - 1) * 100
cambio_p75 = ((df_percentiles['p75'] / baseline['p75']) - 1) * 100
cambio_media = ((df_percentiles['media'] / baseline['media']) - 1) * 100

ax2.plot(df_percentiles['foto_mes'].astype(str), cambio_p25, marker='s', label='P25', linewidth=2)
ax2.plot(df_percentiles['foto_mes'].astype(str), cambio_p50, marker='^', label='P50 (Mediana)', linewidth=2)
ax2.plot(df_percentiles['foto_mes'].astype(str), cambio_p75, marker='D', label='P75', linewidth=2)
ax2.plot(df_percentiles['foto_mes'].astype(str), cambio_media, marker='*', linewidth=2, markersize=10, label='Media')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.axvline(x='202003', color='red', linestyle='--', alpha=0.3, label='Inicio Cuarentena')
ax2.set_xlabel('Mes')
ax2.set_ylabel('Cambio % vs Marzo 2020')
ax2.set_title('Cambio Relativo en Distribución vs Marzo 2020 (baseline)')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('analisis_covid_completo.png', dpi=150, bbox_inches='tight')
print("\n✅ Gráfico guardado como 'analisis_covid_completo.png'")

# Análisis del IQR (rango intercuartílico)
df_percentiles['IQR'] = df_percentiles['p75'] - df_percentiles['p25']
df_percentiles['cambio_IQR_pct'] = ((df_percentiles['IQR'] / baseline['p75'] - baseline['p25']) - 1) * 100

print("\n=== ANÁLISIS DE DISPERSIÓN (IQR = P75 - P25) ===")
print(df_percentiles[['foto_mes', 'IQR']].to_string(index=False))