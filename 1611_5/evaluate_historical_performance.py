# evaluate_historical_performance.py
import yaml
import duckdb
import numpy as np
import polars as pl
from pathlib import Path
from typing import List, Tuple, Dict
import joblib
import logging
import os
import gc
from tqdm import tqdm
from collections import defaultdict
from src.features import create_sql_table_from_parquet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class HistoricalPerformanceEvaluator:
    def __init__(self, ensemble_config_path: str):
        """
        Evaluador de performance histórica del ensemble.
        
        Args:
            ensemble_config_path: Ruta al YAML de configuración del ensemble
        """
        with open(ensemble_config_path) as f:
            self.ensemble_config = yaml.safe_load(f)
        
        self.output_dir = Path(self.ensemble_config['ensemble']['output_dir']).expanduser()
        self.models_dir = self.output_dir / "trained_models"
        self.eval_output_dir = self.output_dir / "historical_evaluation"
        self.eval_output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Models directory: {self.models_dir}")
        logger.info(f"Evaluation output: {self.eval_output_dir}")
    
    def load_model_config(self, config_dir: str) -> dict:
        """Carga el conf.yaml del modelo"""
        config_path = Path(config_dir).expanduser() / "conf.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"No se encontró {config_path}")
        
        with open(config_path) as f:
            full_config = yaml.safe_load(f)
        
        model_config = {
            'STUDY_NAME': full_config.get('STUDY_NAME'),
            **full_config.get('configuracion', {})
        }
        
        return model_config
    
    def predict_by_study(
        self,
        study_name: str,
        model_files: List[Path],
        conn: duckdb.DuckDBPyConnection,
        table_name: str,
        prediction_period: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predice con todos los modelos de un study específico.
        
        Returns:
            Tuple[probabilidades_promedio, numero_de_cliente, clase_real]
        """
        query = f"SELECT * FROM {table_name} WHERE foto_mes = '{prediction_period}'"
        
        logger.info(f"  Cargando datos de {prediction_period}...")
        data = conn.execute(query).fetchnumpy()
        
        # Extraer info necesaria
        numero_de_cliente = data['numero_de_cliente']
        clase_real = data.get('clase_ternaria', np.zeros(len(numero_de_cliente)))
        
        # Convertir clase_real a binario (1 si es BAJA+2, 0 si no)
        if isinstance(clase_real[0], (bytes, str)):
            clase_real = np.array([1 if c in [b'BAJA+2', 'BAJA+2'] else 0 for c in clase_real])
        
        logger.info(f"  Total clientes: {len(numero_de_cliente):,}")
        logger.info(f"  Churns reales: {clase_real.sum():,} ({clase_real.mean()*100:.2f}%)")
        
        # Predecir con cada modelo
        study_predictions = []
        
        for model_file in tqdm(model_files, desc=f"  Prediciendo {study_name}"):
            model_package = joblib.load(model_file)
            model = model_package['model']
            feature_cols = model_package['feature_cols']
            
            # Preparar features
            X = np.column_stack([data[col] for col in feature_cols])
            
            # Predecir
            pred = model.predict(X)
            study_predictions.append(pred)
            
            # Limpiar
            del model, model_package, X
            gc.collect()
        
        # Promedio de este study
        study_proba_avg = np.mean(study_predictions, axis=0)
        
        logger.info(f"  ✓ Predicciones completadas para {study_name}")
        logger.info(f"    Modelos usados: {len(study_predictions)}")
        logger.info(f"    Probabilidad media: {study_proba_avg.mean():.4f}")
        
        # Limpiar
        del data, study_predictions
        gc.collect()
        
        return study_proba_avg, numero_de_cliente, clase_real
    
    def calculate_gains_by_threshold(
        self,
        probas: np.ndarray,
        clase_real: np.ndarray,
        ganancia_acierto: int = 780000,
        costo_estimulo: int = 20000,
        thresholds: np.ndarray = None
    ) -> Dict:
        """
        Calcula ganancias para diferentes umbrales de probabilidad.
        
        Returns:
            Dict con métricas por umbral
        """
        if thresholds is None:
            # Probar varios umbrales
            thresholds = np.linspace(0.001, 0.1, 100)
        
        results = []
        
        for threshold in thresholds:
            # Predicción: contactar si proba >= threshold
            pred = (probas >= threshold).astype(int)
            
            # Confusión
            TP = ((pred == 1) & (clase_real == 1)).sum()  # Contactamos y era churn
            FP = ((pred == 1) & (clase_real == 0)).sum()  # Contactamos y NO era churn
            TN = ((pred == 0) & (clase_real == 0)).sum()  # No contactamos y no era churn
            FN = ((pred == 0) & (clase_real == 1)).sum()  # No contactamos y era churn
            
            # Ganancia
            ganancia = TP * ganancia_acierto - (TP + FP) * costo_estimulo
            
            # Métricas adicionales
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results.append({
                'threshold': float(threshold),
                'n_envios': int(TP + FP),
                'TP': int(TP),
                'FP': int(FP),
                'TN': int(TN),
                'FN': int(FN),
                'ganancia': float(ganancia),
                'ganancia_por_envio': float(ganancia / (TP + FP)) if (TP + FP) > 0 else 0,
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1)
            })
        
        return results
    
    def calculate_gains_top_n(
        self,
        probas: np.ndarray,
        clase_real: np.ndarray,
        ganancia_acierto: int = 780000,
        costo_estimulo: int = 20000,
        n_values: List[int] = None
    ) -> Dict:
        """
        Calcula ganancias para diferentes valores de top-N clientes.
        """
        if n_values is None:
            n_values = [5000, 7000, 9000, 11000, 13000, 15000, 17000, 19000, 21000]
        
        # Ordenar por probabilidad
        idx_sorted = np.argsort(probas)[::-1]
        
        results = []
        
        for n in n_values:
            if n > len(probas):
                continue
            
            # Top N clientes
            top_n_idx = idx_sorted[:n]
            
            # ¿Cuántos de esos top N son realmente churns?
            TP = clase_real[top_n_idx].sum()
            FP = n - TP
            
            # Ganancia
            ganancia = TP * ganancia_acierto - n * costo_estimulo
            
            # Umbral usado
            threshold = probas[idx_sorted[n-1]]
            
            results.append({
                'n_envios': int(n),
                'threshold': float(threshold),
                'TP': int(TP),
                'FP': int(FP),
                'ganancia': float(ganancia),
                'ganancia_por_envio': float(ganancia / n),
                'precision': float(TP / n)
            })
        
        return results
    
    def evaluate_month(self, prediction_period: str) -> Dict:
        """
        Evalúa performance del ensemble en un mes específico.
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"EVALUANDO: {prediction_period}")
        logger.info(f"{'='*70}")
        
        ganancia_acierto = self.ensemble_config['test']['GANANCIA_ACIERTO']
        costo_estimulo = self.ensemble_config['test']['COSTO_ESTIMULO']
        models_config = self.ensemble_config['models']
        
        # Agrupar modelos por study
        models_by_study = defaultdict(list)
        for model_file in self.models_dir.glob("*.pkl"):
            study_name = model_file.stem.rsplit('_seed', 1)[0]
            models_by_study[study_name].append(model_file)
        
        # Predecir con cada study
        study_dataframes = []
        
        for model_info in models_config:
            study_name = model_info['study_name']
            
            if study_name not in models_by_study:
                logger.warning(f"⚠️  No se encontraron modelos para {study_name}, saltando...")
                continue
            
            logger.info(f"\nProcesando {study_name}...")
            
            # Cargar config del modelo
            model_config = self.load_model_config(model_info['config_dir'])
            
            # Data path
            if 'DATA_PATH_OPT' in model_config:
                data_path = model_config['DATA_PATH_OPT']
            else:
                raise ValueError(f"No se encontró DATA_PATH en config de {study_name}")
            
            # Crear tabla en DuckDB
            table_name = f"eval_{study_name}_{prediction_period}"
            data_path_expanded = os.path.expanduser(data_path)
            
            conn = create_sql_table_from_parquet(data_path_expanded, table_name)
            
            try:
                # Predecir
                study_proba, clientes, clase_real = self.predict_by_study(
                    study_name,
                    models_by_study[study_name],
                    conn,
                    table_name,
                    prediction_period
                )
                
                # Guardar en DataFrame
                df_study = pl.DataFrame({
                    'numero_de_cliente': clientes,
                    f'proba_{study_name}': study_proba,
                    'clase_real': clase_real
                })
                
                study_dataframes.append(df_study)
                
            finally:
                conn.close()
                gc.collect()
        
        # Alinear todos los DataFrames
        logger.info(f"\nAlineando {len(study_dataframes)} modelos por cliente...")
        
        df_ensemble = study_dataframes[0]
        for df_study in study_dataframes[1:]:
            df_ensemble = df_ensemble.join(
                df_study.drop('clase_real'),  # clase_real ya está en el primer DF
                on='numero_de_cliente',
                how='inner'
            )
        
        logger.info(f"Clientes comunes: {len(df_ensemble):,}")
        
        # Calcular promedio de probabilidades
        proba_cols = [col for col in df_ensemble.columns if col.startswith('proba_')]
        
        df_ensemble = df_ensemble.with_columns([
            pl.mean_horizontal(proba_cols).alias('probabilidad_ensemble')
        ])
        
        ensemble_proba = df_ensemble['probabilidad_ensemble'].to_numpy()
        clase_real = df_ensemble['clase_real'].to_numpy()
        
        logger.info(f"\nEstadísticas:")
        logger.info(f"  Total clientes: {len(clase_real):,}")
        logger.info(f"  Churns reales: {clase_real.sum():,} ({clase_real.mean()*100:.2f}%)")
        logger.info(f"  Probabilidad media: {ensemble_proba.mean():.4f}")
        
        # Guardar probabilidades y resultados reales
        probas_file = self.eval_output_dir / f"probabilidades_{prediction_period}.parquet"
        df_ensemble.write_parquet(probas_file)
        logger.info(f"✓ Probabilidades guardadas: {probas_file}")
        
        # Evaluar por threshold
        logger.info(f"\nCalculando ganancias por threshold...")
        gains_by_threshold = self.calculate_gains_by_threshold(
            ensemble_proba, clase_real, ganancia_acierto, costo_estimulo
        )
        
        df_threshold = pl.DataFrame(gains_by_threshold)
        threshold_file = self.eval_output_dir / f"gains_by_threshold_{prediction_period}.parquet"
        df_threshold.write_parquet(threshold_file)
        
        # Mejor threshold
        best_threshold_idx = df_threshold['ganancia'].arg_max()
        best_threshold_result = gains_by_threshold[best_threshold_idx]
        
        logger.info(f"\nMejor threshold:")
        logger.info(f"  Threshold: {best_threshold_result['threshold']:.6f}")
        logger.info(f"  Ganancia: ${best_threshold_result['ganancia']:,.0f}")
        logger.info(f"  Envíos: {best_threshold_result['n_envios']:,}")
        logger.info(f"  TP: {best_threshold_result['TP']:,}")
        logger.info(f"  FP: {best_threshold_result['FP']:,}")
        
        # Evaluar por top-N
        logger.info(f"\nCalculando ganancias por top-N...")
        gains_by_topn = self.calculate_gains_top_n(
            ensemble_proba, clase_real, ganancia_acierto, costo_estimulo
        )
        
        df_topn = pl.DataFrame(gains_by_topn)
        topn_file = self.eval_output_dir / f"gains_by_topn_{prediction_period}.parquet"
        df_topn.write_parquet(topn_file)
        
        # Mejor top-N
        best_topn_idx = df_topn['ganancia'].arg_max()
        best_topn_result = gains_by_topn[best_topn_idx]
        
        logger.info(f"\nMejor top-N:")
        logger.info(f"  N envíos: {best_topn_result['n_envios']:,}")
        logger.info(f"  Ganancia: ${best_topn_result['ganancia']:,.0f}")
        logger.info(f"  TP: {best_topn_result['TP']:,}")
        logger.info(f"  Threshold implícito: {best_topn_result['threshold']:.6f}")
        
        # Ganancia específica para top 11,000
        top11k_result = next((r for r in gains_by_topn if r['n_envios'] == 11000), None)
        
        if top11k_result:
            logger.info(f"\nTop 11,000 (competencia):")
            logger.info(f"  Ganancia: ${top11k_result['ganancia']:,.0f}")
            logger.info(f"  TP: {top11k_result['TP']:,}")
            logger.info(f"  Precision: {top11k_result['precision']:.4f}")
        
        return {
            'period': prediction_period,
            'total_clientes': int(len(clase_real)),
            'total_churns': int(clase_real.sum()),
            'churn_rate': float(clase_real.mean()),
            'best_threshold': best_threshold_result,
            'best_topn': best_topn_result,
            'top11k': top11k_result if top11k_result else None,
            'files': {
                'probabilidades': str(probas_file),
                'gains_by_threshold': str(threshold_file),
                'gains_by_topn': str(topn_file)
            }
        }
    
    def run_evaluation(self):
        """Evalúa performance en mayo y junio"""
        logger.info(f"\n{'='*70}")
        logger.info(f"EVALUACIÓN HISTÓRICA DE PERFORMANCE")
        logger.info(f"{'='*70}")
        
        # Evaluar cada mes
        results = {}
        
        for period in ['202105', '202106']:
            try:
                results[period] = self.evaluate_month(period)
            except Exception as e:
                logger.error(f"Error evaluando {period}: {e}")
                continue
        
        # Resumen comparativo
        logger.info(f"\n{'='*70}")
        logger.info(f"RESUMEN COMPARATIVO")
        logger.info(f"{'='*70}")
        
        for period, result in results.items():
            logger.info(f"\n{period}:")
            logger.info(f"  Total clientes: {result['total_clientes']:,}")
            logger.info(f"  Churns reales: {result['total_churns']:,} ({result['churn_rate']*100:.2f}%)")
            logger.info(f"  Mejor ganancia (threshold óptimo): ${result['best_threshold']['ganancia']:,.0f}")
            logger.info(f"  Mejor ganancia (top-N óptimo): ${result['best_topn']['ganancia']:,.0f}")
            if result['top11k']:
                logger.info(f"  Ganancia con top 11,000: ${result['top11k']['ganancia']:,.0f}")
        
        # Guardar resumen
        import json
        summary_file = self.eval_output_dir / "evaluation_summary.json"
        
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n✓ Resumen guardado: {summary_file}")
        
        return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python evaluate_historical_performance.py <ensemble_config.yaml>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    evaluator = HistoricalPerformanceEvaluator(config_path)
    evaluator.run_evaluation()