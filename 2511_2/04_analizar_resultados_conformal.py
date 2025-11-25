import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
import os

# Configuración
STUDY_NAME = "2511_1"
BUCKET_NAME = "gs://sra_electron_bukito3/"

def descargar_desde_gcs():
    """
    Descarga resultados desde GCS.
    """
    print("Descargando resultados desde GCS...")
    
    local_path = os.path.expanduser("~/resultados_conformal")
    os.makedirs(local_path, exist_ok=True)
    
    gcs_path = f"{BUCKET_NAME}conformal_output/"
    
    try:
        subprocess.run(
            ['gsutil', '-m', 'rsync', '-r', gcs_path, local_path],
            check=True
        )
        print(f"✓ Resultados descargados a {local_path}")
        return local_path
    except subprocess.CalledProcessError as e:
        print(f"Error descargando: {e}")
        return None


def cargar_resultados(local_path):
    """
    Carga el archivo JSON con los resultados.
    """
    archivo_json = Path(local_path) / "resultados" / f"{STUDY_NAME}_conformal_results.json"
    
    if not archivo_json.exists():
        print(f"No se encontró {archivo_json}")
        return None
    
    with open(archivo_json, 'r') as f:
        resultados = json.load(f)
    
    print("✓ Resultados cargados")
    return resultados


def crear_visualizaciones(resultados):
    """
    Crea visualizaciones comparativas de las estrategias.
    """
    print("\nGenerando visualizaciones...")
    
    estrategias = resultados['resultados_por_estrategia']
    
    # Extraer datos
    nombres = [e['strategy'] for e in estrategias]
    ganancias = [e['ganancia'] for e in estrategias]
    envios = [e['envios'] for e in estrategias]
    porcentajes = [e['porcentaje_envios'] for e in estrategias]
    
    # Crear figura con subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Comparación de Estrategias de Ensemble con Conformal Prediction', 
                 fontsize=16, fontweight='bold')
    
    # 1. Ganancia por estrategia
    ax1 = axes[0, 0]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    bars1 = ax1.bar(nombres, ganancias, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Ganancia ($)', fontsize=12)
    ax1.set_title('Ganancia por Estrategia', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Agregar valores en las barras
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:,.0f}',
                ha='center', va='bottom', fontsize=10)
    
    # 2. Envíos por estrategia
    ax2 = axes[0, 1]
    bars2 = ax2.bar(nombres, envios, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Número de Envíos', fontsize=12)
    ax2.set_title('Cantidad de Envíos por Estrategia', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=10)
    
    # 3. Porcentaje de envíos
    ax3 = axes[1, 0]
    bars3 = ax3.bar(nombres, porcentajes, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Porcentaje de Envíos (%)', fontsize=12)
    ax3.set_title('Porcentaje de la Base Contactada', fontsize=13, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=10)
    
    # 4. Tabla resumen
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    tabla_datos = []
    for e in estrategias:
        tabla_datos.append([
            e['strategy'].upper(),
            f"${e['ganancia']:,.0f}",
            f"{e['envios']:,}",
            f"{e['porcentaje_envios']:.2f}%",
            f"{e['threshold']:.6f}"
        ])
    
    tabla = ax4.table(
        cellText=tabla_datos,
        colLabels=['Estrategia', 'Ganancia', 'Envíos', '%', 'Threshold'],
        cellLoc='center',
        loc='center',
        colWidths=[0.25, 0.2, 0.15, 0.15, 0.25]
    )
    
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(9)
    tabla.scale(1, 2)
    
    # Estilo de la tabla
    for i in range(len(tabla_datos) + 1):
        for j in range(5):
            cell = tabla[(i, j)]
            if i == 0:
                cell.set_facecolor('#34495e')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')
    
    plt.tight_layout()
    
    # Guardar
    output_path = Path.home() / 'resultados_conformal' / 'comparacion_estrategias.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico guardado: {output_path}")
    
    plt.show()


def imprimir_resumen(resultados):
    """
    Imprime un resumen detallado de los resultados.
    """
    print("\n" + "="*80)
    print("RESUMEN DE RESULTADOS - CONFORMAL PREDICTION ENSEMBLE")
    print("="*80)
    
    config = resultados['configuracion']
    print(f"\nCONFIGURACIÓN:")
    print(f"  Study: {resultados['study_name']}")
    print(f"  Períodos train: {config['periodos_train'][0]} a {config['periodos_train'][-1]} ({len(config['periodos_train'])} meses)")
    print(f"  Calibración: {config['mes_calibracion']}")
    print(f"  Evaluación: {config['mes_evaluacion']}")
    print(f"  Undersampling: {config['undersampling_ratio']*100}%")
    print(f"  Modelos: {config['n_modelos']}")
    
    print(f"\nHIPERPARÁMETROS ÓPTIMOS:")
    for param, valor in config['best_params'].items():
        print(f"  {param}: {valor}")
    print(f"  best_iteration: {config['best_iteration']}")
    
    print("\n" + "-"*80)
    print("RESULTADOS POR ESTRATEGIA:")
    print("-"*80)
    
    estrategias = resultados['resultados_por_estrategia']
    
    for e in estrategias:
        print(f"\n{e['strategy'].upper()}:")
        print(f"  Ganancia:        ${e['ganancia']:>15,.0f}")
        print(f"  Envíos:          {e['envios']:>15,} ({e['porcentaje_envios']:.2f}%)")
        print(f"  Threshold:       {e['threshold']:>15.6f}")
        
        if 'weights' in e:
            weights = np.array(e['weights'])
            print(f"  Pesos - min/max: {weights.min():.4f} / {weights.max():.4f}")
            print(f"  Pesos - std:     {weights.std():.4f}")
    
    # Comparación
    print("\n" + "-"*80)
    print("COMPARACIÓN:")
    print("-"*80)
    
    mejor = max(estrategias, key=lambda x: x['ganancia'])
    simple = next(e for e in estrategias if e['strategy'] == 'simple')
    
    print(f"\n🏆 MEJOR ESTRATEGIA: {mejor['strategy'].upper()}")
    print(f"   Ganancia: ${mejor['ganancia']:,.0f}")
    
    if mejor['strategy'] != 'simple':
        mejora = mejor['ganancia'] - simple['ganancia']
        mejora_pct = (mejora / abs(simple['ganancia'])) * 100 if simple['ganancia'] != 0 else 0
        print(f"\n   Mejora vs Simple: ${mejora:+,.0f} ({mejora_pct:+.2f}%)")
    
    print("\n" + "="*80)


def analizar_pesos_fijos(resultados):
    """
    Analiza la distribución de pesos en la estrategia de peso fijo.
    """
    estrategia_fija = next((e for e in resultados['resultados_por_estrategia'] 
                           if e['strategy'] == 'peso_fijo'), None)
    
    if not estrategia_fija or 'weights' not in estrategia_fija:
        print("No hay información de pesos fijos")
        return
    
    weights = np.array(estrategia_fija['weights'])
    
    print("\n" + "="*80)
    print("ANÁLISIS DE PESOS FIJOS POR MODELO")
    print("="*80)
    
    print(f"\nEstadísticas:")
    print(f"  Media:    {weights.mean():.6f}")
    print(f"  Mediana:  {np.median(weights):.6f}")
    print(f"  Std:      {weights.std():.6f}")
    print(f"  Min:      {weights.min():.6f}")
    print(f"  Max:      {weights.max():.6f}")
    
    # Histograma
    plt.figure(figsize=(10, 6))
    plt.hist(weights, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
    plt.xlabel('Peso', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.title('Distribución de Pesos por Modelo (Estrategia Peso Fijo)', 
              fontsize=13, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    output_path = Path.home() / 'resultados_conformal' / 'distribucion_pesos.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Gráfico guardado: {output_path}")
    plt.show()


def main():
    """
    Script principal de análisis.
    """
    print("="*80)
    print("ANÁLISIS DE RESULTADOS - CONFORMAL PREDICTION ENSEMBLE")
    print("="*80)
    
    # 1. Descargar resultados
    local_path = descargar_desde_gcs()
    
    if not local_path:
        print("Error descargando resultados")
        return
    
    # 2. Cargar resultados
    resultados = cargar_resultados(local_path)
    
    if not resultados:
        print("Error cargando resultados")
        return
    
    # 3. Imprimir resumen
    imprimir_resumen(resultados)
    
    # 4. Crear visualizaciones
    crear_visualizaciones(resultados)
    
    # 5. Análisis de pesos
    analizar_pesos_fijos(resultados)
    
    # 6. Subir TODO (incluyendo gráficos) al bucket
    print("\n" + "="*80)
    print("SUBIENDO RESULTADOS Y GRÁFICOS A GCS")
    print("="*80)
    
    gcs_path = f"{BUCKET_NAME}conformal_output/"
    
    try:
        subprocess.run(
            ['gsutil', '-m', 'rsync', '-r', local_path, gcs_path],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✓ TODO sincronizado con GCS: {gcs_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error subiendo a GCS: {e}")
    
    print("\n✓ Análisis completado")


if __name__ == "__main__":
    main()