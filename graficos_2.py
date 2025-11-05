import duckdb
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# --- CONFIGURACIÓN ---
CSV_PATH = 'data/datasets_competencia_02_crudo.csv.gz'
OUTPUT_PDF = 'medias_por_columna.pdf'

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

# --- GENERAR Y GUARDAR GRÁFICOS ---
with PdfPages(OUTPUT_PDF) as pdf:
    for col in cols_numericas:
        df = con.execute(f"""
            SELECT foto_mes, AVG({col}) AS media
            FROM datos
            WHERE {col} IS NOT NULL
            GROUP BY foto_mes
            ORDER BY foto_mes
        """).fetchdf()

        # Convertir foto_mes a string para mejor visualización
        df['foto_mes'] = df['foto_mes'].astype(str)

        # Crear gráfico
        plt.figure(figsize=(10, 4))
        plt.plot(df['foto_mes'], df['media'], marker='o', linewidth=1.5)
        plt.title(f'Media de {col} por foto_mes')
        plt.xlabel('foto_mes')
        plt.ylabel(f'Media de {col}')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True)
        plt.tight_layout()

        # Guardar en PDF
        pdf.savefig()
        plt.close()

print(f"✅ PDF generado correctamente: {OUTPUT_PDF}")
print(f"📊 Se graficaron {len(cols_numericas)} columnas")