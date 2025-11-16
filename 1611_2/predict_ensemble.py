# predict_ensemble.py
import yaml
import duckdb
import numpy as np
import polars as pl
from pathlib import Path
from typing import List, Dict
import joblib
import logging
import os
import gc
from tqdm import tqdm
from src.features import create_sql_table_from_parquet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class EnsemblePredictor:
    def __init__(self, ensemble_config_path: str):
        """
        Inicializa el predictor de ensemble.
        
        Args:
            ensemble_config_path: Ruta al YAML de configuración del ensemble
        """
        with open(ensemble_config_path) as f:
            self.ensemble_config = yaml.safe_load(f)
        
        self.output_dir = Path(self.ensemble_config['ensemble']['output_dir']).expanduser()
        self.models_dir = self.output_dir / "trained_models"
        self.predictions_dir = self.output_dir / "predictions"
        self.predictions_dir.mkdir(exist_ok=True)
        
        logger.info(f"Ensemble directory: {self.output_dir}")
        logger.info(f"Models directory: {self.models_dir}")
    
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
    
    def predict_on_test(
        self,
        model,
        feature_cols: List[str],
        conn: duckdb.DuckDBPyConnection,
        table_name: str,
        test_periods: List[str],
        ganancia_acierto: int,
        costo_estimulo: int
    ) -> dict:
        """
        Hace predicción en conjunto de test y calcula ganancia.
        """
        periodos_test_str = ','.join(map(str, test_periods))
        query_test = f"SELECT * FROM {table_name} WHERE foto_mes IN ({periodos_test_str})"
        
        test_data = conn.execute(query_test).fetchnumpy()
        
        # Preparar features
        X_test = np.column_stack([test_data[col] for col in feature_cols])
        y_test = test_data['target_ternario']
        
        # Predecir
        y_pred_proba = model.predict(X_test)
        
        # Calcular ganancia óptima
        df_eval = pl.DataFrame({
            'y_true': y_test,
            'y_pred_proba': y_pred_proba
        })
        
        df_ordenado = df_eval.sort('y_pred_proba', descending=True)
        df_ordenado = df_ordenado.with_columns([
            pl.when(pl.col('y_true') == 1)
              .then(ganancia_acierto)
              .otherwise(-costo_estimulo)
              .cast(pl.Int64)
              .alias('ganancia_individual')
        ])
        df_ordenado = df_ordenado.with_columns([
            pl.col('ganancia_individual').cum_sum().alias('ganancia_acumulada')
        ])
        
        idx_max = df_ordenado.select(pl.col('ganancia_acumulada').arg_max()).item()
        ganancia_maxima = df_ordenado.select(pl.col('ganancia_acumulada').max()).item()
        
        n_envios = idx_max + 1
        
        result = {
            'y_pred_proba': y_pred_proba,
            'y_true': y_test,
            'ganancia': float(ganancia_maxima),
            'n_envios': int(n_envios),
            'n_test': len(y_test)
        }
        
        del test_data, X_test
        gc.collect()
        
        return result
    
    def predict_all(self):
        """Genera predicciones para todos los modelos en test1 y test2"""
        
        models_config = self.ensemble_config['models']
        test_periods_1 = self.ensemble_config['test']['MES_TEST_1']
        test_periods_2 = self.ensemble_config['test']['MES_TEST_2']
        ganancia_acierto = self.ensemble_config['test']['GANANCIA_ACIERTO']
        costo_estimulo = self.ensemble_config['test']['COSTO_ESTIMULO']
        
        all_predictions_test1 = []
        all_predictions_test2 = []
        all_results = []
        
        for model_info in models_config:
            study_name = model_info['study_name']
            
            logger.info(f"\n{'='*70}")
            logger.info(f"PREDICCIONES: {study_name}")
            logger.info(f"{'='*70}")
            
            conn = None
            
            try:
                # 1. Cargar config del modelo
                model_config = self.load_model_config(model_info['config_dir'])
                
                # 2. Determinar data_path
                if 'DATA_PATH_OPT' in model_config:
                    data_path = model_config['DATA_PATH_OPT']
                else:
                    raise ValueError(f"No se encontró DATA_PATH en config de {study_name}")
                
                # 3. Crear tabla en DuckDB
                table_name = f"tabla_{study_name}"
                data_path_expanded = os.path.expanduser(data_path)
                
                conn = create_sql_table_from_parquet(data_path_expanded, table_name)
                
                # 4. Cargar todos los modelos de este study
                model_files = list(self.models_dir.glob(f"{study_name}_seed*.pkl"))
                logger.info(f"Encontrados {len(model_files)} modelos")
                
                # 5. Predecir con cada modelo
                for model_file in tqdm(model_files, desc=f"Prediciendo {study_name}"):
                    try:
                        # Cargar modelo
                        model_package = joblib.load(model_file)
                        model = model_package['model']
                        feature_cols = model_package['feature_cols']
                        semilla = model_package['semilla']
                        
                        # Predecir en TEST 1
                        pred_test1 = self.predict_on_test(
                            model, feature_cols, conn, table_name,
                            test_periods_1, ganancia_acierto, costo_estimulo
                        )
                        
                        # Predecir en TEST 2
                        pred_test2 = self.predict_on_test(
                            model, feature_cols, conn, table_name,
                            test_periods_2, ganancia_acierto, costo_estimulo
                        )
                        
                        # Guardar predicciones
                        all_predictions_test1.append({
                            'study_name': study_name,
                            'semilla': semilla,
                            'y_pred_proba': pred_test1['y_pred_proba'],
                            'y_true': pred_test1['y_true']
                        })
                        
                        all_predictions_test2.append({
                            'study_name': study_name,
                            'semilla': semilla,
                            'y_pred_proba': pred_test2['y_pred_proba'],
                            'y_true': pred_test2['y_true']
                        })
                        
                        # Registrar resultado
                        all_results.append({
                            'study_name': study_name,
                            'semilla': semilla,
                            'ganancia_test1': pred_test1['ganancia'],
                            'n_envios_test1': pred_test1['n_envios'],
                            'ganancia_test2': pred_test2['ganancia'],
                            'n_envios_test2': pred_test2['n_envios']
                        })
                        
                        # Limpiar modelo de memoria
                        del model, model_package
                        gc.collect()
                        
                    except Exception as e:
                        logger.error(f"Error prediciendo con {model_file.name}: {str(e)}")
                        continue
                
            except Exception as e:
                logger.error(f"Error en modelo {study_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                
            finally:
                if conn is not None:
                    try:
                        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                        conn.close()
                        logger.info(f"Conexión cerrada para {study_name}")
                    except Exception as e:
                        logger.warning(f"Error al cerrar conexión de {study_name}: {e}")
                gc.collect()
        
        # Guardar predicciones
        logger.info("\nGuardando predicciones...")
        self._save_predictions(all_predictions_test1, 'test1')
        self._save_predictions(all_predictions_test2, 'test2')
        
        # Guardar resumen de resultados
        if len(all_results) > 0:
            results_df = pl.DataFrame(all_results)
            results_csv = self.output_dir / "predictions_summary.csv"
            results_df.write_csv(results_csv)
            logger.info(f"Resumen guardado en: {results_csv}")
        
        # Analizar ensembles
        logger.info("\nAnalizando estrategias de ensemble...")
        self._analyze_ensembles(ganancia_acierto, costo_estimulo)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"PREDICCIONES COMPLETADAS")
        logger.info(f"Total predicciones: {len(all_results)}")
        logger.info(f"Predicciones guardadas en: {self.predictions_dir}")
        logger.info(f"{'='*70}")
        
        return results_df if len(all_results) > 0 else pl.DataFrame()
    
    def _save_predictions(self, predictions: List[dict], test_name: str):
        """Guarda predicciones en formato parquet"""
        for pred in predictions:
            filename = f"{pred['study_name']}_seed{pred['semilla']}_{test_name}.parquet"
            filepath = self.predictions_dir / filename
            
            df = pl.DataFrame({
                'y_pred_proba': pred['y_pred_proba'],
                'y_true': pred['y_true']
            })
            
            df.write_parquet(filepath)
        
        logger.info(f"Predicciones de {test_name} guardadas: {len(predictions)} archivos")
    
    def _analyze_ensembles(self, ganancia_acierto: int, costo_estimulo: int):
        """
        Analiza dos estrategias de ensemble:
        1. Promedio simple de todas las predicciones
        2. Promedio por modelo y luego entre modelos
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"ANÁLISIS DE ENSEMBLES")
        logger.info(f"{'='*70}")
        
        for test_name in ['test1', 'test2']:
            logger.info(f"\n--- {test_name.upper()} ---")
            
            # Cargar todas las predicciones
            pred_files = list(self.predictions_dir.glob(f"*_{test_name}.parquet"))
            
            if len(pred_files) == 0:
                logger.warning(f"No se encontraron predicciones para {test_name}")
                continue
            
            all_preds = []
            study_preds = {}
            
            for pred_file in pred_files:
                df_pred = pl.read_parquet(pred_file)
                y_true = df_pred['y_true'].to_numpy()
                y_pred = df_pred['y_pred_proba'].to_numpy()
                
                all_preds.append(y_pred)
                
                # Extraer study_name
                study_name = pred_file.stem.split('_seed')[0]
                if study_name not in study_preds:
                    study_preds[study_name] = []
                study_preds[study_name].append(y_pred)
            
            # Estrategia 1: Promedio simple
            y_pred_ensemble_simple = np.mean(all_preds, axis=0)
            ganancia_simple = self._calculate_ganancia(
                y_true, y_pred_ensemble_simple, ganancia_acierto, costo_estimulo
            )
            
            logger.info(f"\nEstrategia 1 - Promedio simple ({len(all_preds)} predicciones):")
            logger.info(f"  Ganancia: {ganancia_simple['ganancia']:,.0f}")
            logger.info(f"  Envíos: {ganancia_simple['n_envios']:,}")
            
            # Estrategia 2: Promedio por modelo, luego entre modelos
            study_averages = []
            for study_name, preds in study_preds.items():
                study_avg = np.mean(preds, axis=0)
                study_averages.append(study_avg)
                
                ganancia_study = self._calculate_ganancia(
                    y_true, study_avg, ganancia_acierto, costo_estimulo
                )
                logger.info(f"\n  {study_name} (promedio {len(preds)} semillas):")
                logger.info(f"    Ganancia: {ganancia_study['ganancia']:,.0f}")
                logger.info(f"    Envíos: {ganancia_study['n_envios']:,}")
            
            # Promedio final
            y_pred_ensemble_2stage = np.mean(study_averages, axis=0)
            ganancia_2stage = self._calculate_ganancia(
                y_true, y_pred_ensemble_2stage, ganancia_acierto, costo_estimulo
            )
            
            logger.info(f"\nEstrategia 2 - Promedio 2 etapas ({len(study_averages)} modelos):")
            logger.info(f"  Ganancia: {ganancia_2stage['ganancia']:,.0f}")
            logger.info(f"  Envíos: {ganancia_2stage['n_envios']:,}")
            
            logger.info(f"\nComparación:")
            logger.info(f"  Diferencia: {ganancia_2stage['ganancia'] - ganancia_simple['ganancia']:,.0f}")
            
            # Guardar análisis
            import json
            analysis_results = {
                'test_name': test_name,
                'estrategia_1_ganancia': float(ganancia_simple['ganancia']),
                'estrategia_1_envios': int(ganancia_simple['n_envios']),
                'estrategia_2_ganancia': float(ganancia_2stage['ganancia']),
                'estrategia_2_envios': int(ganancia_2stage['n_envios']),
                'diferencia': float(ganancia_2stage['ganancia'] - ganancia_simple['ganancia'])
            }
            
            analysis_file = self.output_dir / f"ensemble_analysis_{test_name}.json"
            with open(analysis_file, 'w') as f:
                json.dump(analysis_results, f, indent=2)
            
            logger.info(f"\nAnálisis guardado en: {analysis_file}")
    
    def _calculate_ganancia(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                           ganancia_acierto: int, costo_estimulo: int) -> dict:
        """Calcula ganancia óptima"""
        df_eval = pl.DataFrame({
            'y_true': y_true,
            'y_pred_proba': y_pred_proba
        })
        
        df_ordenado = df_eval.sort('y_pred_proba', descending=True)
        df_ordenado = df_ordenado.with_columns([
            pl.when(pl.col('y_true') == 1)
              .then(ganancia_acierto)
              .otherwise(-costo_estimulo)
              .cast(pl.Int64)
              .alias('ganancia_individual')
        ])
        df_ordenado = df_ordenado.with_columns([
            pl.col('ganancia_individual').cum_sum().alias('ganancia_acumulada')
        ])
        
        idx_max = df_ordenado.select(pl.col('ganancia_acumulada').arg_max()).item()
        ganancia_maxima = df_ordenado.select(pl.col('ganancia_acumulada').max()).item()
        
        return {
            'ganancia': float(ganancia_maxima),
            'n_envios': int(idx_max + 1)
        }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "ensemble_config.yaml"
    
    predictor = EnsemblePredictor(config_path)
    results = predictor.predict_all()
    
    print("\n=== RESUMEN FINAL ===")
    if len(results) > 0:
        print(results.group_by('study_name').agg([
            pl.count('semilla').alias('n_modelos'),
            pl.mean('ganancia_test1').alias('ganancia_test1_mean'),
            pl.mean('ganancia_test2').alias('ganancia_test2_mean')
        ]))