import duckdb
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# --- CONFIGURACIÓN ---
CSV_PATH = 'data/datasets_competencia_02_crudo.csv.gz'
OUTPUT_PDF = 'analisis_drifting_vs_estructural.pdf'

# --- CONEXIÓN Y CARGA DE DATOS ---
con = duckdb.connect()
con.execute(f"""
    CREATE OR REPLACE TABLE datos AS
    SELECT * FROM read_csv_auto('{CSV_PATH}', HEADER=TRUE);
""")

# --- COLUMNAS NUMÉRICAS ---
cols_numericas = con.execute("""
    SELECT column_name
    FROM information_schema.columns
    WHERE table_name = 'datos'
      AND data_type IN ('INTEGER', 'BIGINT', 'DOUBLE', 'FLOAT', 'REAL', 'DECIMAL')
      AND column_name != 'foto_mes';
""").fetchdf()['column_name'].tolist()

# --- CREAR TABLA CON DECILES Y ESTADÍSTICAS ---
print("Calculando deciles y estadísticas...")

deciles_columns = []
for col in cols_numericas:
    deciles_columns.append(f"""
        NTILE(10) OVER (PARTITION BY foto_mes ORDER BY {col}) AS {col}_decil
    """)

query_deciles = f"""
    CREATE OR REPLACE TABLE datos_deciles AS
    SELECT 
        foto_mes,
        {','.join(deciles_columns)}
    FROM datos
    WHERE {' AND '.join([f'{col} IS NOT NULL' for col in cols_numericas])}
"""

con.execute(query_deciles)

# --- GENERAR GRÁFICOS COMPARATIVOS ---
with PdfPages(OUTPUT_PDF) as pdf:
    for col in cols_numericas:
        # Obtener estadísticas de la columna original
        df_original = con.execute(f"""
            SELECT 
                foto_mes,
                AVG({col}) AS media,
                MEDIAN({col}) AS mediana,
                STDDEV({col}) AS desv_std,
                COUNT(*) as n_registros,
                SUM(CASE WHEN {col} = 0 THEN 1 ELSE 0 END) as n_ceros
            FROM datos
            WHERE {col} IS NOT NULL
            GROUP BY foto_mes
            ORDER BY foto_mes
        """).fetchdf()
        
        # Obtener estadísticas de deciles
        df_deciles = con.execute(f"""
            SELECT 
                foto_mes,
                AVG({col}_decil) AS media_decil,
                STDDEV({col}_decil) AS desv_std_decil
            FROM datos_deciles
            GROUP BY foto_mes
            ORDER BY foto_mes
        """).fetchdf()
        
        # Convertir foto_mes a string
        df_original['foto_mes'] = df_original['foto_mes'].astype(str)
        df_deciles['foto_mes'] = df_deciles['foto_mes'].astype(str)
        
        # Calcular coeficiente de variación (CV) para detectar drifting
        df_original['cv'] = df_original['desv_std'] / (df_original['media'].abs() + 1)
        
        # Identificar meses problemáticos (todos ceros)
        meses_problema = df_original[df_original['n_ceros'] == df_original['n_registros']]['foto_mes'].tolist()
        
        # --- CREAR FIGURA CON 3 SUBPLOTS ---
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle(f'Análisis de {col}: Drifting vs Cambios Estructurales', fontsize=14, fontweight='bold')
        
        # SUBPLOT 1: Valores Originales (Media y Mediana)
        ax1 = axes[0]
        ax1.plot(df_original['foto_mes'], df_original['media'], 
                marker='o', linewidth=1.5, label='Media', color='blue')
        ax1.plot(df_original['foto_mes'], df_original['mediana'], 
                marker='s', linewidth=1.5, label='Mediana', color='green', alpha=0.7)
        
        # Marcar meses problemáticos
        for mes in meses_problema:
            idx = df_original[df_original['foto_mes'] == mes].index
            if len(idx) > 0:
                ax1.axvline(x=idx[0], color='red', linestyle='--', alpha=0.3)
        
        ax1.set_ylabel('Valor Original')
        ax1.set_title('Valores Originales (sensible a inflación y outliers)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # SUBPLOT 2: Deciles (mitiga drifting, preserva cambios estructurales)
        ax2 = axes[1]
        ax2.plot(df_deciles['foto_mes'], df_deciles['media_decil'], 
                marker='o', linewidth=1.5, color='purple')
        ax2.axhline(y=5.5, color='gray', linestyle='--', alpha=0.5, label='Esperado (5.5)')
        ax2.set_ylabel('Media de Deciles')
        ax2.set_ylim(4, 7)
        ax2.set_title('Deciles por foto_mes (mitiga drifting, detecta cambios estructurales)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # SUBPLOT 3: Coeficiente de Variación (detecta drifting)
        ax3 = axes[2]
        ax3.plot(df_original['foto_mes'], df_original['cv'], 
                marker='o', linewidth=1.5, color='orange')
        ax3.set_ylabel('Coeficiente de Variación')
        ax3.set_xlabel('foto_mes')
        ax3.set_title('Coeficiente de Variación (indica magnitud del drifting)')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # --- ANÁLISIS CUANTITATIVO ---
        # Detectar cambios estructurales en deciles
        df_deciles['cambio_decil'] = df_deciles['media_decil'].diff().abs()
        cambios_significativos = df_deciles[df_deciles['cambio_decil'] > 0.3]
        
        if len(cambios_significativos) > 0 and len(meses_problema) == 0:
            print(f"\n⚠️  {col}: Cambios estructurales detectados en deciles:")
            print(cambios_significativos[['foto_mes', 'media_decil', 'cambio_decil']])

print(f"\n✅ PDF generado correctamente: {OUTPUT_PDF}")
print("\n📊 INTERPRETACIÓN:")
print("- Subplot 1 (Original): Muestra drift por inflación + cambios estructurales")
print("- Subplot 2 (Deciles): Mitiga inflación pero PRESERVA cambios estructurales")
print("- Subplot 3 (CV): Alta variabilidad indica fuerte drifting")
print("\n💡 Si los deciles se mantienen ~5.5, el drifting es solo por escala (inflación)")
print("💡 Si los deciles cambian abruptamente, hay cambios estructurales reales")q 