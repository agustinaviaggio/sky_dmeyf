# analyze_probability_distributions.py
import polars as pl
import numpy as np
from pathlib import Path
import json

class ProbabilityDistributionAnalyzer:
    def __init__(self, ensemble_dirs: dict):
        """
        Analiza distribuciones de probabilidades entre meses.
        
        Args:
            ensemble_dirs: Dict con {period: ensemble_dir}
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
        """Calcula estadísticas descriptivas completas"""
        proba = df['probabilidad_ensemble'].to_numpy()
        
        stats = {
            'period': period,
            'n_clientes': len(proba),
            'mean': float(np.mean(proba)),
            'std': float(np.std(proba)),
            'min': float(np.min(proba)),
            'max': float(np.max(proba)),
            'median': float(np.median(proba)),
            'q01': float(np.percentile(proba, 1)),
            'q05': float(np.percentile(proba, 5)),
            'q10': float(np.percentile(proba, 10)),
            'q25': float(np.percentile(proba, 25)),
            'q75': float(np.percentile(proba, 75)),
            'q90': float(np.percentile(proba, 90)),
            'q95': float(np.percentile(proba, 95)),
            'q99': float(np.percentile(proba, 99)),
        }
        
        # Top-N thresholds
        for n in [1000, 5000, 11000, 15000, 20000]:
            if n <= len(proba):
                proba_sorted = np.sort(proba)[::-1]
                stats[f'threshold_top{n}'] = float(proba_sorted[n-1])
        
        # Si tiene clase_real, agregar stats por clase
        if 'clase_real' in df.columns:
            churners = df.filter(pl.col('clase_real') == 1)
            no_churners = df.filter(pl.col('clase_real') == 0)
            
            proba_churners = churners['probabilidad_ensemble'].to_numpy()
            proba_no_churners = no_churners['probabilidad_ensemble'].to_numpy()
            
            stats['n_churners'] = len(proba_churners)
            stats['n_no_churners'] = len(proba_no_churners)
            stats['churn_rate'] = len(proba_churners) / len(proba)
            
            # Stats de churners
            stats['mean_churners'] = float(np.mean(proba_churners)) if len(proba_churners) > 0 else None
            stats['std_churners'] = float(np.std(proba_churners)) if len(proba_churners) > 0 else None
            stats['median_churners'] = float(np.median(proba_churners)) if len(proba_churners) > 0 else None
            stats['q25_churners'] = float(np.percentile(proba_churners, 25)) if len(proba_churners) > 0 else None
            stats['q75_churners'] = float(np.percentile(proba_churners, 75)) if len(proba_churners) > 0 else None
            
            # Stats de no-churners
            stats['mean_no_churners'] = float(np.mean(proba_no_churners))
            stats['std_no_churners'] = float(np.std(proba_no_churners))
            stats['median_no_churners'] = float(np.median(proba_no_churners))
            stats['q25_no_churners'] = float(np.percentile(proba_no_churners, 25))
            stats['q75_no_churners'] = float(np.percentile(proba_no_churners, 75))
            
            # Ratios
            if stats['mean_no_churners'] > 0:
                stats['mean_ratio'] = stats['mean_churners'] / stats['mean_no_churners']
                stats['median_ratio'] = stats['median_churners'] / stats['median_no_churners']
            
            # Recall en diferentes top-N
            df_sorted = df.sort('probabilidad_ensemble', descending=True)
            clase_real = df_sorted['clase_real'].to_numpy()
            total_churners = clase_real.sum()
            
            for n in [1000, 5000, 11000, 15000, 20000]:
                if n <= len(clase_real):
                    churners_in_top_n = clase_real[:n].sum()
                    stats[f'recall_top{n}'] = float(churners_in_top_n / total_churners) if total_churners > 0 else 0
                    stats[f'precision_top{n}'] = float(churners_in_top_n / n)
        
        return stats
    
    def analyze_distribution_shifts(self, stats_list: list) -> dict:
        """Analiza cambios en distribuciones entre períodos"""
        if len(stats_list) < 2:
            return {}
        
        shifts = {}
        
        # Comparar cada par consecutivo
        for i in range(len(stats_list) - 1):
            current = stats_list[i]
            next_period = stats_list[i + 1]
            
            comparison_key = f"{current['period']}_vs_{next_period['period']}"
            
            shifts[comparison_key] = {
                'delta_mean': next_period['mean'] - current['mean'],
                'delta_median': next_period['median'] - current['median'],
                'delta_std': next_period['std'] - current['std'],
                'pct_change_mean': ((next_period['mean'] - current['mean']) / current['mean'] * 100) if current['mean'] > 0 else 0,
            }
            
            # Si ambos tienen churners
            if 'mean_churners' in current and 'mean_churners' in next_period:
                if current['mean_churners'] and next_period['mean_churners']:
                    shifts[comparison_key]['delta_mean_churners'] = next_period['mean_churners'] - current['mean_churners']
                    shifts[comparison_key]['delta_churn_rate'] = next_period['churn_rate'] - current['churn_rate']
        
        return shifts
    
    def print_summary_table(self, stats_list: list):
        """Imprime tabla resumen en consola"""
        print(f"\n{'='*100}")
        print(f"TABLA COMPARATIVA DE PROBABILIDADES")
        print(f"{'='*100}\n")
        
        # Métricas básicas
        print(f"{'Métrica':<25} {'202105':>15} {'202106':>15} {'202108':>15}")
        print(f"{'-'*25} {'-'*15} {'-'*15} {'-'*15}")
        
        metrics = [
            ('n_clientes', 'Total Clientes', ','),
            ('mean', 'Media', '.4f'),
            ('std', 'Desv. Estándar', '.4f'),
            ('median', 'Mediana', '.4f'),
            ('q25', 'Q25', '.4f'),
            ('q75', 'Q75', '.4f'),
            ('q90', 'Q90', '.4f'),
            ('q95', 'Q95', '.4f'),
            ('q99', 'Q99', '.4f'),
        ]
        
        for key, label, fmt in metrics:
            values = []
            for stats in stats_list:
                val = stats.get(key)
                if val is not None:
                    if fmt == ',':
                        values.append(f"{int(val):,}")
                    else:
                        values.append(f"{val:{fmt}}")
                else:
                    values.append('-')
            
            print(f"{label:<25} {values[0]:>15} {values[1]:>15} {values[2]:>15}")
        
        # Si hay churners (mayo y junio)
        if 'n_churners' in stats_list[0]:
            print(f"\n{'='*100}")
            print(f"ESTADÍSTICAS POR CLASE (Mayo y Junio)")
            print(f"{'='*100}\n")
            
            print(f"{'Métrica':<25} {'202105':>15} {'202106':>15}")
            print(f"{'-'*25} {'-'*15} {'-'*15}")
            
            churn_metrics = [
                ('n_churners', 'Churners', ','),
                ('churn_rate', 'Tasa de Churn (%)', '.2%'),
                ('mean_churners', 'Media Churners', '.4f'),
                ('mean_no_churners', 'Media No-Churners', '.4f'),
                ('mean_ratio', 'Ratio Media', '.2f'),
                ('median_churners', 'Mediana Churners', '.4f'),
                ('median_no_churners', 'Mediana No-Churners', '.4f'),
            ]
            
            for key, label, fmt in churn_metrics:
                values = []
                for stats in stats_list[:2]:  # Solo mayo y junio
                    val = stats.get(key)
                    if val is not None:
                        if fmt == ',':
                            values.append(f"{int(val):,}")
                        elif fmt == '.2%':
                            values.append(f"{val*100:.2f}%")
                        else:
                            values.append(f"{val:{fmt}}")
                    else:
                        values.append('-')
                
                print(f"{label:<25} {values[0]:>15} {values[1]:>15}")
        
        # Top-N thresholds
        print(f"\n{'='*100}")
        print(f"UMBRALES DE PROBABILIDAD POR TOP-N")
        print(f"{'='*100}\n")
        
        print(f"{'Top-N':<15} {'202105':>15} {'202106':>15} {'202108':>15}")
        print(f"{'-'*15} {'-'*15} {'-'*15} {'-'*15}")
        
        for n in [1000, 5000, 11000, 15000, 20000]:
            key = f'threshold_top{n}'
            values = []
            for stats in stats_list:
                val = stats.get(key)
                if val is not None:
                    values.append(f"{val:.6f}")
                else:
                    values.append('-')
            
            print(f"Top {n:,}".ljust(15) + f"{values[0]:>15} {values[1]:>15} {values[2]:>15}")
        
        # Recall y precision (solo mayo y junio)
        if 'recall_top11000' in stats_list[0]:
            print(f"\n{'='*100}")
            print(f"RECALL Y PRECISION POR TOP-N (Mayo y Junio)")
            print(f"{'='*100}\n")
            
            print(f"{'Top-N':<15} {'Mayo Recall':>15} {'Junio Recall':>15} {'Mayo Precision':>15} {'Junio Precision':>15}")
            print(f"{'-'*15} {'-'*15} {'-'*15} {'-'*15} {'-'*15}")
            
            for n in [1000, 5000, 11000, 15000, 20000]:
                recall_key = f'recall_top{n}'
                precision_key = f'precision_top{n}'
                
                mayo_recall = stats_list[0].get(recall_key)
                junio_recall = stats_list[1].get(recall_key)
                mayo_prec = stats_list[0].get(precision_key)
                junio_prec = stats_list[1].get(precision_key)
                
                print(f"Top {n:,}".ljust(15) + 
                      f"{mayo_recall*100:>14.2f}%" + 
                      f"{junio_recall*100:>14.2f}%" +
                      f"{mayo_prec*100:>14.2f}%" +
                      f"{junio_prec*100:>14.2f}%")
    
    def run_analysis(self):
        """Ejecuta análisis completo"""
        print(f"\n{'='*100}")
        print(f"ANÁLISIS DE DISTRIBUCIONES DE PROBABILIDADES")
        print(f"{'='*100}\n")
        
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
                
            except FileNotFoundError as e:
                print(f"  ⚠️  {e}")
                continue
        
        if not dfs:
            print("❌ No se encontraron datos para analizar")
            return
        
        # Imprimir tabla resumen
        self.print_summary_table(stats_list)
        
        # Analizar shifts
        shifts = self.analyze_distribution_shifts(stats_list)
        
        if shifts:
            print(f"\n{'='*100}")
            print(f"CAMBIOS ENTRE PERÍODOS")
            print(f"{'='*100}\n")
            
            for comparison, values in shifts.items():
                print(f"\n{comparison}:")
                for key, val in values.items():
                    print(f"  {key}: {val:.4f}")
        
        # Guardar todo en archivos
        summary_file = self.analysis_dir / "statistics_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'statistics': stats_list,
                'shifts': shifts
            }, f, indent=2)
        
        print(f"\n{'='*100}")
        print(f"✓ Análisis completado")
        print(f"✓ Resultados guardados en: {summary_file}")
        print(f"{'='*100}\n")


if __name__ == "__main__":
    # Configurar directorios por período
    ensemble_dirs = {
        '202105': '~/buckets/b1/ensemble_final_1',
        '202106': '~/buckets/b1/ensemble_final_1',
        '202108': '~/buckets/b1/ensemble_final_2'
    }
    
    analyzer = ProbabilityDistributionAnalyzer(ensemble_dirs)
    analyzer.run_analysis()