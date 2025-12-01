import duckdb
import logging
import numpy as np
import json
import os
import gc
import sys
import yaml
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
import lightgbm as lgb
import optuna

# Lista de estudios a evaluar
ESTUDIOS = ['2511_2', '2611_2', '2711_1', '2711_2', '2811_1', '2911_1', '3011_1']

# Primeras 25 semillas
SEMILLAS = [600011, 600043, 600053, 600071, 600073,
            600091, 600107, 600109, 600113, 600137,
            600169, 600179, 600191, 600193, 600197,
            600209, 600211, 600221, 600227, 600253,
            600257, 600259, 600263, 600281, 600293]

# Configuración de logging
os.makedirs("logs", exist_ok=True)
fecha = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
nombre_log = f"log_maestro_conformal_{fecha}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/" + nombre_log),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class ConfiguracionEstudio:
    """Carga configuración de un estudio específico."""
    
    def __init__(self, study_name, base_path="~/sky_dmeyf"):
        self.study_name = study_name
        self.base_path = Path(base_path).expanduser()
        self.study_path = self.base_path / study_name
        
        # Cargar conf.yaml
        conf_file = self.study_path / "conf.yaml"
        with open(conf_file, 'r') as f:
            config = yaml.safe_load(f)
            self.config = config['configuracion']
        
        self.bucket_name = self.config['BUCKET_NAME']
        self.data_path = self.config['DATA_PATH_OPT']
        self.ganancia_acierto = self.config['GANANCIA_ACIERTO']
        self.costo_estimulo = self.config['COSTO_ESTIMULO']
        self.undersampling_ratio = self.config.get('UNDERSAMPLING_RATIO', 0.075)
        
        # Intentar detectar si usa features seleccionadas
        self.features_seleccionadas = self._cargar_features_seleccionadas()
    
    def _cargar_features_seleccionadas(self):
        """Intenta cargar features seleccionadas desde GCS."""
        try:
            # Buscar archivo de features en GCS
            gcs_pattern = f"{self.bucket_name}resultados/union_features_{self.study_name}_*.json"
            
            result = subprocess.run(
                ['gsutil', 'ls', gcs_pattern],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.info(f"{self.study_name}: No hay features seleccionadas, se usarán todas")
                return None
            
            # Tomar el más reciente
            archivos = result.stdout.strip().split('\n')
            if not archivos or archivos[0] == '':
                return None
            
            archivo_mas_reciente = sorted(archivos)[-1]
            
            # Descargar y leer
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                tmp_path = tmp.name
            
            try:
                subprocess.run(
                    ['gsutil', 'cp', archivo_mas_reciente, tmp_path],
                    capture_output=True,
                    check=True
                )
                
                with open(tmp_path, 'r') as f:
                    data = json.load(f)
                
                features = data['union_features']['lista_completa']
                logger.info(f"{self.study_name}: {len(features)} features seleccionadas cargadas")
                return features
                
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    
        except Exception as e:
            logger.warning(f"{self.study_name}: Error cargando features: {e}")
            return None


class ConformalPredictor:
    """Implementa conformal prediction para clasificación binaria."""
    
    def __init__(self, models, feature_cols):
        self.models = models
        self.feature_cols = feature_cols
        self.calibration_scores = None
    
    def fit_calibration(self, X_cal, y_cal):
        """Calcula nonconformity scores en calibración."""
        # Promedio de probabilidades de todos los modelos
        all_probs = [model.predict(X_cal) for model in self.models]
        probs_ensemble = np.mean(all_probs, axis=0)
        
        # Nonconformity score: 1 - prob(clase_verdadera)
        self.calibration_scores = np.array([
            1 - probs_ensemble[i] if y_cal[i] == 1 else probs_ensemble[i]
            for i in range(len(y_cal))
        ])
        
        return self
    
    def calcular_confianza_individual(self, prob):
        """Calcula confianza (1-alpha) para una predicción individual."""
        score_0 = prob
        score_1 = 1 - prob
        
        quantile_0 = np.mean(self.calibration_scores >= score_0)
        quantile_1 = np.mean(self.calibration_scores >= score_1)
        
        alpha = max(quantile_0, quantile_1)
        confidence = 1 - alpha
        
        return confidence, alpha
    
    def evaluar_modelo_individual(self, model_idx, X_test):
        """Evalúa un modelo individual con métricas de conformal prediction."""
        probs = self.models[model_idx].predict(X_test)
        
        confidences = []
        alphas = []
        
        for prob in probs:
            conf, alpha = self.calcular_confianza_individual(prob)
            confidences.append(conf)
            alphas.append(alpha)
        
        return {
            'probabilities': probs,
            'confidences': np.array(confidences),
            'alphas': np.array(alphas),
            'confidence_mean': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'confidence_min': np.min(confidences),
            'confidence_max': np.max(confidences)
        }
    
    def evaluar_ensemble(self, X_test):
        """Evalúa el ensemble con métricas de conformal prediction."""
        all_probs = [model.predict(X_test) for model in self.models]
        probs_ensemble = np.mean(all_probs, axis=0)
        
        confidences = []
        alphas = []
        
        for prob in probs_ensemble:
            conf, alpha = self.calcular_confianza_individual(prob)
            confidences.append(conf)
            alphas.append(alpha)
        
        return {
            'probabilities': probs_ensemble,
            'confidences': np.array(confidences),
            'alphas': np.array(alphas),
            'confidence_mean': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'confidence_min': np.min(confidences),
            'confidence_max': np.max(confidences)
        }


def cargar_hiperparametros_optuna(study_name, bucket_name):
    """Carga hiperparámetros desde Optuna en GCS."""
    logger.info(f"{study_name}: Cargando hiperparámetros de Optuna")
    
    local_db_dir = Path.home() / "optuna_db"
    local_db_dir.mkdir(exist_ok=True)
    
    db_file = local_db_dir / f"{study_name}.db"
    gcs_path = f"{bucket_name}optuna_db/{study_name}.db"
    
    subprocess.run(
        ['gsutil', 'cp', gcs_path, str(db_file)],
        check=True,
        capture_output=True
    )
    
    storage = f"sqlite:///{db_file}"
    study = optuna.load_study(study_name=study_name, storage=storage)
    
    best_params = study.best_params
    best_iteration = study.best_trial.user_attrs.get('best_iteration', 1000)
    
    logger.info(f"{study_name}: Mejor ganancia={study.best_value:,.0f}, iteration={best_iteration}")
    
    return best_params, best_iteration


def entrenar_modelos(conn, tabla, config, best_params, best_iteration):
    """Entrena 25 modelos hasta 202105."""
    logger.info(f"{config.study_name}: Entrenando 25 modelos hasta 202105")
    
    # Períodos hasta 202105
    query_train = f"""
        WITH clase_0_sample AS (
            SELECT * FROM {tabla}
            WHERE foto_mes < 202106
              AND target_binario = 0
            USING SAMPLE {config.undersampling_ratio * 100} PERCENT (bernoulli, {SEMILLAS[0]})
        ),
        clase_1_completa AS (
            SELECT * FROM {tabla}
            WHERE foto_mes < 202106
              AND target_binario = 1
        )
        SELECT * FROM clase_0_sample
        UNION ALL
        SELECT * FROM clase_1_completa
    """
    
    train_data = conn.execute(query_train).fetchnumpy()
    
    feature_cols = [col for col in train_data.keys() 
                    if col not in ['target_binario', 'target_ternario', 'foto_mes']]
    
    X_train = np.column_stack([train_data[col] for col in feature_cols])
    y_train = train_data['target_binario']
    
    logger.info(f"{config.study_name}: {len(train_data['target_binario']):,} registros, {len(feature_cols)} features")
    
    # Entrenar 25 modelos
    models = []
    for i, semilla in enumerate(SEMILLAS):
        params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'max_bin': 31,
            'verbose': -1,
            'is_unbalance': True,
            'bagging_freq': 1,
            'n_jobs': -1,
            'seed': semilla,
            **best_params
        }
        
        train_set = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
        model = lgb.train(params, train_set, num_boost_round=best_iteration, 
                         callbacks=[lgb.log_evaluation(period=0)])
        
        models.append(model)
        del train_set
        gc.collect()
    
    logger.info(f"{config.study_name}: ✓ 25 modelos entrenados")
    
    del X_train, y_train, train_data
    gc.collect()
    
    return models, feature_cols


def cargar_datos_cal_test(conn, tabla, feature_cols):
    """Carga 202106 (calibración) y 202107 (test)."""
    # Calibración
    query_cal = f"SELECT * FROM {tabla} WHERE foto_mes = 202106"
    data_cal = conn.execute(query_cal).fetchnumpy()
    
    X_cal = np.column_stack([data_cal[col] for col in feature_cols])
    y_cal = data_cal['target_binario']
    
    # Test
    query_test = f"SELECT * FROM {tabla} WHERE foto_mes = 202107"
    data_test = conn.execute(query_test).fetchnumpy()
    
    X_test = np.column_stack([data_test[col] for col in feature_cols])
    y_test_ternario = data_test['target_ternario']
    
    del data_cal, data_test
    gc.collect()
    
    return X_cal, y_cal, X_test, y_test_ternario


def encontrar_threshold_optimo(y_true, probs, ganancia_acierto, costo_estimulo):
    """Encuentra threshold que maximiza ganancia."""
    sorted_idx = np.argsort(probs)[::-1]
    y_sorted = y_true[sorted_idx]
    probs_sorted = probs[sorted_idx]
    
    ganancias = np.where(y_sorted == 1, ganancia_acierto, -costo_estimulo)
    ganancia_acum = np.cumsum(ganancias)
    
    idx_max = np.argmax(ganancia_acum)
    
    return {
        'threshold': float(probs_sorted[idx_max]),
        'ganancia': float(ganancia_acum[idx_max]),
        'envios': int(idx_max + 1)
    }


def evaluar_estudio(study_name):
    """Evalúa un estudio completo con conformal prediction."""
    logger.info(f"\n{'='*80}")
    logger.info(f"EVALUANDO ESTUDIO: {study_name}")
    logger.info(f"{'='*80}")
    
    conn = None
    try:
        # 1. Cargar configuración
        config = ConfiguracionEstudio(study_name)
        
        # 2. Conectar DuckDB y GCS
        conn = duckdb.connect(database=':memory:')
        
        from google.auth import default
        from google.auth.transport.requests import Request
        
        credentials, project = default()
        credentials.refresh(Request())
        
        conn.execute("INSTALL httpfs;")
        conn.execute("LOAD httpfs;")
        conn.execute(f"""
            CREATE SECRET (
                TYPE GCS,
                PROVIDER config,
                BEARER_TOKEN '{credentials.token}'
            )
        """)
        
        # 3. Cargar datos
        tabla = "datos"
        
        if config.features_seleccionadas:
            cols_necesarias = ['target_binario', 'target_ternario', 'foto_mes']
            cols_str = ', '.join(config.features_seleccionadas + cols_necesarias)
            conn.execute(f"""
                CREATE TABLE {tabla} AS 
                SELECT {cols_str}
                FROM read_parquet('{config.data_path}')
            """)
        else:
            conn.execute(f"""
                CREATE TABLE {tabla} AS 
                SELECT * FROM read_parquet('{config.data_path}')
            """)
        
        # 4. Cargar hiperparámetros
        best_params, best_iteration = cargar_hiperparametros_optuna(study_name, config.bucket_name)
        
        # 5. Entrenar modelos
        models, feature_cols = entrenar_modelos(conn, tabla, config, best_params, best_iteration)
        
        # 6. Cargar datos cal/test
        X_cal, y_cal, X_test, y_test = cargar_datos_cal_test(conn, tabla, feature_cols)
        
        # 7. Conformal prediction
        cp = ConformalPredictor(models, feature_cols)
        cp.fit_calibration(X_cal, y_cal)
        
        logger.info(f"{study_name}: Evaluando modelos individuales...")
        
        # 8. Evaluar cada modelo individual
        resultados_individuales = []
        for i in range(len(SEMILLAS)):
            resultado = cp.evaluar_modelo_individual(i, X_test)
            
            # Calcular ganancia
            ganancia_info = encontrar_threshold_optimo(
                y_test, 
                resultado['probabilities'],
                config.ganancia_acierto,
                config.costo_estimulo
            )
            
            resultados_individuales.append({
                'semilla': int(SEMILLAS[i]),
                'modelo_idx': i,
                'confidence_mean': float(resultado['confidence_mean']),
                'confidence_std': float(resultado['confidence_std']),
                'confidence_min': float(resultado['confidence_min']),
                'confidence_max': float(resultado['confidence_max']),
                **ganancia_info
            })
        
        # 9. Evaluar ensemble
        logger.info(f"{study_name}: Evaluando ensemble...")
        resultado_ensemble = cp.evaluar_ensemble(X_test)
        
        ganancia_ensemble = encontrar_threshold_optimo(
            y_test,
            resultado_ensemble['probabilities'],
            config.ganancia_acierto,
            config.costo_estimulo
        )
        
        resultado_ensemble_final = {
            'confidence_mean': float(resultado_ensemble['confidence_mean']),
            'confidence_std': float(resultado_ensemble['confidence_std']),
            'confidence_min': float(resultado_ensemble['confidence_min']),
            'confidence_max': float(resultado_ensemble['confidence_max']),
            **ganancia_ensemble
        }
        
        logger.info(f"{study_name}: Ganancia ensemble = {ganancia_ensemble['ganancia']:,.0f}")
        logger.info(f"{study_name}: Confianza media = {resultado_ensemble['confidence_mean']:.4f}")
        
        return {
            'study_name': study_name,
            'resultados_individuales': resultados_individuales,
            'resultado_ensemble': resultado_ensemble_final,
            'config': {
                'n_features': len(feature_cols),
                'usa_features_seleccionadas': config.features_seleccionadas is not None,
                'undersampling_ratio': config.undersampling_ratio
            }
        }
        
    finally:
        if conn:
            conn.close()


def guardar_resultados(resultados, output_path="gs://sra_electron_bukito3/conformal_comparacion/"):
    """Guarda resultados en GCS."""
    logger.info("Guardando resultados en GCS...")
    
    local_temp = Path.home() / "temp_conformal_comparacion"
    local_temp.mkdir(exist_ok=True)
    
    for estudio in resultados:
        study_name = estudio['study_name']
        estudio_path = local_temp / study_name
        estudio_path.mkdir(exist_ok=True)
        
        # Guardar individuales
        with open(estudio_path / "resultados_individuales.json", 'w') as f:
            json.dump(estudio['resultados_individuales'], f, indent=2)
        
        # Guardar ensemble
        with open(estudio_path / "resultados_ensemble.json", 'w') as f:
            json.dump(estudio['resultado_ensemble'], f, indent=2)
        
        # Guardar config
        with open(estudio_path / "config.json", 'w') as f:
            json.dump(estudio['config'], f, indent=2)
    
    # Reporte comparativo
    comparacion = {
        'fecha': datetime.now().isoformat(),
        'estudios_evaluados': [r['study_name'] for r in resultados],
        'comparacion': [
            {
                'study_name': r['study_name'],
                'ganancia_ensemble': r['resultado_ensemble']['ganancia'],
                'confidence_mean': r['resultado_ensemble']['confidence_mean'],
                'threshold': r['resultado_ensemble']['threshold'],
                'envios': r['resultado_ensemble']['envios'],
                'n_features': r['config']['n_features']
            }
            for r in resultados
        ]
    }
    
    # Ordenar por ganancia
    comparacion['comparacion'] = sorted(
        comparacion['comparacion'], 
        key=lambda x: x['ganancia_ensemble'], 
        reverse=True
    )
    
    with open(local_temp / "reporte_comparativo.json", 'w') as f:
        json.dump(comparacion, f, indent=2)
    
    # Subir a GCS
    subprocess.run(
        ['gsutil', '-m', 'rsync', '-r', str(local_temp), output_path],
        check=True
    )
    
    logger.info(f"✓ Resultados guardados en {output_path}")
    
    # Limpiar
    import shutil
    shutil.rmtree(local_temp)


def main():
    """Pipeline maestro."""
    logger.info("="*80)
    logger.info("PIPELINE MAESTRO - COMPARACIÓN DE MODELOS CON CONFORMAL PREDICTION")
    logger.info("="*80)
    logger.info(f"Estudios a evaluar: {ESTUDIOS}")
    logger.info(f"Semillas: {len(SEMILLAS)} primeras")
    
    resultados = []
    
    for study_name in ESTUDIOS:
        try:
            resultado = evaluar_estudio(study_name)
            resultados.append(resultado)
        except Exception as e:
            logger.error(f"Error evaluando {study_name}: {e}", exc_info=True)
            continue
    
    # Guardar resultados
    guardar_resultados(resultados)
    
    # Imprimir resumen
    logger.info("\n" + "="*80)
    logger.info("RESUMEN COMPARATIVO")
    logger.info("="*80)
    
    for r in sorted(resultados, key=lambda x: x['resultado_ensemble']['ganancia'], reverse=True):
        logger.info(f"\n{r['study_name']}:")
        logger.info(f"  Ganancia:   ${r['resultado_ensemble']['ganancia']:>15,.0f}")
        logger.info(f"  Confianza:  {r['resultado_ensemble']['confidence_mean']:>15.4f}")
        logger.info(f"  Threshold:  {r['resultado_ensemble']['threshold']:>15.6f}")
        logger.info(f"  Features:   {r['config']['n_features']:>15,}")
    
    logger.info("\n✓ Pipeline maestro completado")


if __name__ == "__main__":
    main()