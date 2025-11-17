# analyze_probability_distributions.py
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class ProbabilityDistributionAnalyzer:
    def __init__(self, ensemble_dirs: dict):
        """
        Analiza distribuciones de probabilidades entre meses.
        
        Args:
            ensemble_dirs: Dict con {period: ensemble_dir}
                          Ej: {'202105': '~/buckets/b1/ensemble_final_1', 
                               '202106': '~/buckets/b1/ensemble_final_1',
                               '202108': '~/buckets/b1/ensemble_final_2'}
        """
        self.ensemble_dirs = {k: Path(v).expanduser() for k, v in ensemble_dirs.items()}
        
        # Directorio de salida (usar el del primer ensemble)
        first_dir = list(self.ensemble_dirs.values())[0]
        self.analysis_dir = first_dir / "probability_analysis"
        self.analysis_dir.mkdir(exist_ok=True)
    
    def load_probabilities(self, period: str) -> pl.DataFrame:
        """Carga probabilidades de un período"""
        ensemble_dir = self.ensemble_dirs[period]
        
        if period == '202108':
            file_path = ensemble_dir / "final_submission" / f"probabilidades_{period}.parquet"
        else:
            file_path = ensemble_dir / "historical_evaluation" / f"probabilidades_{period}.parquet"
        
        if not file_path.exists():
            raise FileNotFoundError(f"No se encontró {file_path}")
        
        return pl.read_parquet(file_path)
    
    def calculate_statistics(self, df: pl.DataFrame, period: str) -> dict:
        """Calcula estadísticas descriptivas"""
        proba = df['probabilidad_ensemble'].to_numpy()
        
        stats = {
            'period': period,
            'n_clientes': len(proba),
            'mean': float(np.mean(proba)),
            'std': float(np.std(proba)),
            'min': float(np.min(proba)),
            'max': float(np.max(proba)),
            'median': float(np.median(proba)),
            'q25': float(np.percentile(proba, 25)),
            'q75': float(np.percentile(proba, 75)),
            'q90': float(np.percentile(proba, 90)),
            'q95': float(np.percentile(proba, 95)),
            'q99': float(np.percentile(proba, 99)),
        }
        
        # Si tiene clase_real, agregar stats por clase
        if 'clase_real' in df.columns:
            proba_churners = df.filter(pl.col('clase_real') == 1)['probabilidad_ensemble'].to_numpy()
            proba_no_churners = df.filter(pl.col('clase_real') == 0)['probabilidad_ensemble'].to_numpy()
            
            stats['n_churners'] = len(proba_churners)
            stats['churn_rate'] = len(proba_churners) / len(proba)
            stats['mean_churners'] = float(np.mean(proba_churners)) if len(proba_churners) > 0 else None
            stats['mean_no_churners'] = float(np.mean(proba_no_churners))
            stats['median_churners'] = float(np.median(proba_churners)) if len(proba_churners) > 0 else None
            stats['median_no_churners'] = float(np.median(proba_no_churners))
            stats['std_churners'] = float(np.std(proba_churners)) if len(proba_churners) > 0 else None
            stats['std_no_churners'] = float(np.std(proba_no_churners))
        
        return stats
    
    def plot_distributions(self, dfs: dict, output_file: str = "probability_distributions.png"):
        """
        Plotea distribuciones de probabilidades para múltiples períodos.
        
        Args:
            dfs: Dict con {period: dataframe}
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Distribuciones de Probabilidades de Churn por Mes', fontsize=16, y=0.995)
        
        colors = {'202105': '#1f77b4', '202106': '#ff7f0e', '202108': '#2ca02c'}
        
        # 1. Histogramas superpuestos
        ax = axes[0, 0]
        for period, df in dfs.items():
            proba = df['probabilidad_ensemble'].to_numpy()
            ax.hist(proba, bins=100, alpha=0.5, label=period, color=colors.get(period, 'gray'), density=True)
        ax.set_xlabel('Probabilidad')
        ax.set_ylabel('Densidad')
        ax.set_title('Distribución General de Probabilidades')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Box plots comparativos
        ax = axes[0, 1]
        box_data = []
        box_labels = []
        for period in sorted(dfs.keys()):
            box_data.append(dfs[period]['probabilidad_ensemble'].to_numpy())
            box_labels.append(period)
        
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
        for patch, period in zip(bp['boxes'], box_labels):
            patch.set_facecolor(colors.get(period, 'gray'))
        ax.set_ylabel('Probabilidad')
        ax.set_title('Distribución por Percentiles')
        ax.grid(True, alpha=0.3)
        
        # 3. CDFs (Cumulative Distribution Functions)
        ax = axes[1, 0]
        for period in sorted(dfs.keys()):
            proba = np.sort(dfs[period]['probabilidad_ensemble'].to_numpy())
            cdf = np.arange(1, len(proba) + 1) / len(proba)
            ax.plot(proba, cdf, label=period, color=colors.get(period, 'gray'), linewidth=2)
        ax.set_xlabel('Probabilidad')
        ax.set_ylabel('Proporción acumulada')
        ax.set_title('Función de Distribución Acumulada (CDF)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Top percentiles (zoom en colas superiores)
        ax = axes[1, 1]
        for period in sorted(dfs.keys()):
            proba = dfs[period]['probabilidad_ensemble'].to_numpy()
            percentiles = np.arange(90, 100.1, 0.1)
            values = [np.percentile(proba, p) for p in percentiles]
            ax.plot(percentiles, values, label=period, color=colors.get(period, 'gray'), linewidth=2)
        ax.set_xlabel('Percentil')
        ax.set_ylabel('Probabilidad')
        ax.set_title('Distribución en Top 10% (percentiles 90-100)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.analysis_dir / output_file
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico guardado: {output_path}")
        plt.close()
    
    def plot_churners_vs_no_churners(self, dfs: dict, output_file: str = "churners_comparison.png"):
        """
        Compara distribuciones entre churners y no-churners para mayo y junio.
        """
        # Filtrar solo meses con clase_real
        dfs_with_labels = {p: df for p, df in dfs.items() if 'clase_real' in df.columns}
        
        if not dfs_with_labels:
            print("⚠️  No hay datos con clase_real para comparar")
            return
        
        n_periods = len(dfs_with_labels)
        fig, axes = plt.subplots(n_periods, 2, figsize=(14, 5*n_periods))
        
        if n_periods == 1:
            axes = axes.reshape(1, -1)
        
        for idx, period in enumerate(sorted(dfs_with_labels.keys())):
            df = dfs_with_labels[period]
            churners = df.filter(pl.col('clase_real') == 1)['probabilidad_ensemble'].to_numpy()
            no_churners = df.filter(pl.col('clase_real') == 0)['probabilidad_ensemble'].to_numpy()
            
            # Histogramas
            ax = axes[idx, 0]
            ax.hist(churners, bins=50, alpha=0.6, label=f'Churners (n={len(churners)})', 
                   color='red', density=True)
            ax.hist(no_churners, bins=50, alpha=0.6, label=f'No Churners (n={len(no_churners)})', 
                   color='blue', density=True)
            ax.set_xlabel('Probabilidad')
            ax.set_ylabel('Densidad')
            ax.set_title(f'{period}: Distribución por Clase Real')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Box plots
            ax = axes[idx, 1]
            bp = ax.boxplot([churners, no_churners], 
                           labels=['Churners', 'No Churners'],
                           patch_artist=True)
            bp['boxes'][0].set_facecolor('red')
            bp['boxes'][1].set_facecolor('blue')
            ax.set_ylabel('Probabilidad')
            ax.set_title(f'{period}: Comparación por Percentiles')
            ax.grid(True, alpha=0.3)
            
            # Agregar estadísticas
            stats_text = f"Media Churners: {np.mean(churners):.4f}\n"
            stats_text += f"Media No-Churners: {np.mean(no_churners):.4f}\n"
            stats_text += f"Ratio: {np.mean(churners) / np.mean(no_churners):.2f}x"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        output_path = self.analysis_dir / output_file
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico guardado: {output_path}")
        plt.close()
    
    def plot_top_n_analysis(self, dfs: dict, n_values: list = None, 
                           output_file: str = "top_n_analysis.png"):
        """
        Analiza características de los top-N clientes por probabilidad.
        """
        if n_values is None:
            n_values = [1000, 5000, 11000, 15000, 20000]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        colors = {'202105': '#1f77b4', '202106': '#ff7f0e', '202108': '#2ca02c'}
        
        # 1. Probabilidad mínima en top-N
        ax = axes[0]
        for period in sorted(dfs.keys()):
            proba_sorted = np.sort(dfs[period]['probabilidad_ensemble'].to_numpy())[::-1]
            min_probas = [proba_sorted[n-1] if n <= len(proba_sorted) else np.nan for n in n_values]
            ax.plot(n_values, min_probas, marker='o', label=period, 
                   color=colors.get(period, 'gray'), linewidth=2)
        
        ax.axvline(x=11000, color='red', linestyle='--', alpha=0.5, label='Límite competencia (11k)')
        ax.set_xlabel('Número de envíos (Top-N)')
        ax.set_ylabel('Probabilidad mínima en Top-N')
        ax.set_title('Umbral de Probabilidad vs Cantidad de Envíos')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Recall esperado en top-N (solo para mayo y junio)
        ax = axes[1]
        for period in sorted(dfs.keys()):
            if 'clase_real' not in dfs[period].columns:
                continue
            
            # Ordenar por probabilidad
            df_sorted = dfs[period].sort('probabilidad_ensemble', descending=True)
            clase_real = df_sorted['clase_real'].to_numpy()
            total_churners = clase_real.sum()
            
            recalls = []
            for n in n_values:
                if n <= len(clase_real):
                    churners_in_top_n = clase_real[:n].sum()
                    recall = churners_in_top_n / total_churners if total_churners > 0 else 0
                    recalls.append(recall)
                else:
                    recalls.append(np.nan)
            
            ax.plot(n_values, recalls, marker='o', label=period, 
                   color=colors.get(period, 'gray'), linewidth=2)
        
        ax.axvline(x=11000, color='red', linestyle='--', alpha=0.5, label='Límite competencia (11k)')
        ax.set_xlabel('Número de envíos (Top-N)')
        ax.set_ylabel('Recall (% de churners detectados)')
        ax.set_title('Cobertura de Churners vs Cantidad de Envíos')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        output_path = self.analysis_dir / output_file
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico guardado: {output_path}")
        plt.close()
    
    def generate_summary_table(self, stats_list: list) -> pl.DataFrame:
        """Genera tabla resumen de estadísticas"""
        df_stats = pl.DataFrame(stats_list)
        return df_stats
    
    def run_analysis(self):
        """Ejecuta análisis completo"""
        print(f"\n{'='*70}")
        print(f"ANÁLISIS DE DISTRIBUCIONES DE PROBABILIDADES")
        print(f"{'='*70}\n")
        
        # Cargar datos
        dfs = {}
        stats_list = []
        
        for period in sorted(self.ensemble_dirs.keys()):
            try:
                print(f"Cargando {period}...")
                df = self.load_probabilities(period)
                dfs[period] = df
                
                # Calcular estadísticas
                stats = self.calculate_statistics(df, period)
                stats_list.append(stats)
                
                print(f"  ✓ {period}: {stats['n_clientes']:,} clientes")
                if 'n_churners' in stats:
                    print(f"    Churners: {stats['n_churners']:,} ({stats['churn_rate']*100:.2f}%)")
                print(f"    Probabilidad media: {stats['mean']:.4f}")
                print(f"    Percentil 99: {stats['q99']:.4f}\n")
                
            except FileNotFoundError as e:
                print(f"  ⚠️  {e}")
                continue
        
        if not dfs:
            print("❌ No se encontraron datos para analizar")
            return
        
        # Generar visualizaciones
        print("\nGenerando visualizaciones...")
        
        self.plot_distributions(dfs)
        self.plot_churners_vs_no_churners(dfs)
        self.plot_top_n_analysis(dfs)
        
        # Guardar tabla resumen
        df_summary = self.generate_summary_table(stats_list)
        summary_file = self.analysis_dir / "statistics_summary.parquet"
        df_summary.write_parquet(summary_file)
        
        print(f"\n{'='*70}")
        print(f"RESUMEN DE ESTADÍSTICAS")
        print(f"{'='*70}\n")
        print(df_summary)
        
        # Guardar también en CSV para facilitar lectura
        csv_file = self.analysis_dir / "statistics_summary.csv"
        df_summary.write_csv(csv_file)
        print(f"\n✓ Resumen guardado en: {summary_file}")
        print(f"✓ CSV guardado en: {csv_file}")
        
        print(f"\n✓ Análisis completado. Archivos en: {self.analysis_dir}")


if __name__ == "__main__":
    # Configurar directorios por período
    ensemble_dirs = {
        '202105': '~/buckets/b1/ensemble_final_1',
        '202106': '~/buckets/b1/ensemble_final_1',
        '202108': '~/buckets/b1/ensemble_final_2'
    }
    
    analyzer = ProbabilityDistributionAnalyzer(ensemble_dirs)
    analyzer.run_analysis()