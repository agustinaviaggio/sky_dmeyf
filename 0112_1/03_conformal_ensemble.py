import duckdb
import logging
import numpy as np
import pickle
import json
import os
import gc
from datetime import datetime
import lightgbm as lgb
import optuna
import subprocess

from src.features import create_sql_table_from_parquet_csv
from src.config import *

### Configuración de logging ###
os.makedirs("logs", exist_ok=True)
fecha = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
nombre_log = f"log_conformal_{STUDY_NAME}_{fecha}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s",
    handlers=[
        logging.FileHandler("logs/" + nombre_log),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Iniciando pipeline de Conformal Prediction con Ensemble")

### Clase para Conformal Prediction ###
class ConformalPredictor:
    """
    Implementa conformal prediction para clasificación binaria.
    """
    def __init__(self, models, feature_cols):
        self.models = models
        self.feature_cols = feature_cols
        self.calibration_scores = None
        self.n_models = len(models)
        
    def fit_calibration(self, X_cal, y_cal):
        """
        Calcula nonconformity scores en conjunto de calibración.
        """
        logger.info("Calculando nonconformity scores en calibración...")
        
        # Obtener predicciones de todos los modelos
        all_probs = []
        for model in self.models:
            probs = model.predict(X_cal)
            all_probs.append(probs)
        
        # Promedio de probabilidades (ensemble)
        probs_ensemble = np.mean(all_probs, axis=0)
        
        # Nonconformity score: 1 - probabilidad de la clase verdadera
        self.calibration_scores = np.array([
            1 - probs_ensemble[i] if y_cal[i] == 1 else probs_ensemble[i]
            for i in range(len(y_cal))
        ])
        
        logger.info(f"Calibración completada. Scores: min={self.calibration_scores.min():.4f}, "
                   f"max={self.calibration_scores.max():.4f}, "
                   f"mean={self.calibration_scores.mean():.4f}")
        
        return self
    
    def predict_with_confidence(self, X_test):
        """
        Predice con métricas de confianza de conformal prediction.
        """
        logger.info("Generando predicciones con conformal prediction...")
        
        # Obtener predicciones de todos los modelos
        all_probs = []
        for model in self.models:
            probs = model.predict(X_test)
            all_probs.append(probs)
        
        probs_matrix = np.array(all_probs)
        probs_ensemble = probs_matrix.mean(axis=0)
        
        return {
            'probabilities': probs_ensemble,
            'probs_per_model': probs_matrix
        }


### Funciones principales ###

def cargar_hiperparametros_optuna():
    """
    Carga los mejores hiperparámetros desde la DB de Optuna en GCS.
    """
    logger.info("=== CARGANDO HIPERPARÁMETROS DE OPTUNA ===")
    
    local_db_dir = os.path.expanduser("~/optuna_db")
    os.makedirs(local_db_dir, exist_ok=True)
    
    db_file = os.path.join(local_db_dir, f"{STUDY_NAME}.db")
    gcs_path = f"{BUCKET_NAME}optuna_db/{STUDY_NAME}.db"
    
    try:
        logger.info(f"Descargando DB desde {gcs_path}...")
        subprocess.run(
            ['gsutil', 'cp', gcs_path, db_file],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info("✓ DB descargada exitosamente")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error descargando DB: {e.stderr}")
        raise
    
    storage = f"sqlite:///{db_file}"
    study = optuna.load_study(study_name=STUDY_NAME, storage=storage)
    
    best_params = study.best_params
    best_iteration = study.best_trial.user_attrs.get('best_iteration', 1000)
    
    logger.info(f"✓ Estudio cargado: {len(study.trials)} trials")
    logger.info(f"✓ Mejor ganancia: {study.best_value:,.0f}")
    logger.info(f"✓ Mejores parámetros: {best_params}")
    logger.info(f"✓ Best iteration: {best_iteration}")
    
    return best_params, best_iteration


def re_entrenar_modelos(conn, tabla, best_params, best_iteration):
    """
    Re-entrena 25 modelos hasta 202105 con undersampling.
    """
    logger.info("=== RE-ENTRENAMIENTO DE MODELOS HASTA 202105 ===")
    
    # Períodos de entrenamiento (hasta 202105)
    periodos_train = [p for p in PERIODOS_TRAIN if p not in ['202106', '202107']]
    logger.info(f"Períodos de entrenamiento: {periodos_train[0]} a {periodos_train[-1]}")
    logger.info(f"Total: {len(periodos_train)} meses")
    
    periodos_str = ','.join(map(str, periodos_train))
    
    # Query con undersampling
    query_train = f"""
        WITH clase_0_sample AS (
            SELECT * FROM {tabla}
            WHERE foto_mes IN ({periodos_str}) 
              AND target_binario = 0
            USING SAMPLE {UNDERSAMPLING_RATIO * 100} PERCENT (bernoulli, {SEMILLAS[0]})
        ),
        clase_1_completa AS (
            SELECT * FROM {tabla}
            WHERE foto_mes IN ({periodos_str}) 
              AND target_binario = 1
        )
        SELECT * FROM clase_0_sample
        UNION ALL
        SELECT * FROM clase_1_completa
    """
    
    logger.info("Cargando datos de entrenamiento...")
    train_data = conn.execute(query_train).fetchnumpy()
    
    n_clase_0 = (train_data['target_binario'] == 0).sum()
    n_clase_1 = (train_data['target_binario'] == 1).sum()
    
    logger.info(f"Datos (post-undersampling): {len(train_data['target_binario']):,} registros")
    logger.info(f"  Clase 0: {n_clase_0:,} | Clase 1: {n_clase_1:,} | Ratio: {n_clase_1/n_clase_0:.2f}:1")
    
    # Preparar features
    feature_cols = [col for col in train_data.keys() 
                    if col not in ['target_binario', 'target_ternario','foto_mes']]
    
    X_train = np.column_stack([train_data[col] for col in feature_cols])
    y_train = train_data['target_binario']
    
    logger.info(f"Features: {len(feature_cols)} columnas")
    
    # Entrenar modelos
    models = []
    
    for i, semilla in enumerate(SEMILLAS):
        logger.info(f"Entrenando modelo {i+1}/{len(SEMILLAS)} con semilla {semilla}")
        
        params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'first_metric_only': True,
            'boost_from_average': True,
            'feature_pre_filter': False,
            'max_bin': 31,
            'verbose': -1,
            'is_unbalance': True,
            'bagging_freq': 1,
            'n_jobs': -1,
            'seed': semilla,
            **best_params
        }
        
        train_set = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
        
        model = lgb.train(
            params,
            train_set,
            num_boost_round=best_iteration,
            callbacks=[lgb.log_evaluation(period=0)]
        )
        
        models.append(model)
        del train_set
        gc.collect()
    
    logger.info(f"✓ {len(models)} modelos entrenados exitosamente")
    
    del X_train, y_train, train_data
    gc.collect()
    
    return models, feature_cols


def cargar_datos_calibracion_test(conn, tabla):
    """
    Carga 202106 para calibración y 202107 para test.
    """
    logger.info("=== CARGANDO DATOS: 202106 (CALIBRACIÓN) Y 202107 (TEST) ===")
    
    # Cargar 202106 para calibración
    query_cal = f"SELECT * FROM {tabla} WHERE foto_mes = 202106"
    data_cal = conn.execute(query_cal).fetchnumpy()
    
    logger.info(f"Calibración (202106): {len(data_cal['target_binario']):,} registros")
    
    # Cargar 202107 para test
    query_test = f"SELECT * FROM {tabla} WHERE foto_mes = 202107"
    data_test = conn.execute(query_test).fetchnumpy()
    
    logger.info(f"Test (202107): {len(data_test['target_binario']):,} registros")
    
    # Preparar features
    feature_cols = [col for col in data_cal.keys() 
                    if col not in ['target_binario', 'target_ternario','foto_mes']]
    
    # Calibración
    X_cal = np.column_stack([data_cal[col] for col in feature_cols])
    y_cal = data_cal['target_binario']
    
    # Test
    X_test = np.column_stack([data_test[col] for col in feature_cols])
    y_test_binario = data_test['target_binario']
    y_test_ternario = data_test['target_ternario']
    
    logger.info(f"Calibración: Clase 0={( y_cal==0).sum():,} | Clase 1={(y_cal==1).sum():,}")
    logger.info(f"Test: Clase 0={(y_test_binario==0).sum():,} | Clase 1={(y_test_binario==1).sum():,}")
    logger.info(f"Test BAJA+2: {(y_test_ternario==1).sum():,} ({(y_test_ternario==1).sum()/len(y_test_ternario)*100:.2f}%)")
    
    del data_cal, data_test
    gc.collect()
    
    return {
        'X_cal': X_cal,
        'y_cal': y_cal,
        'X_test': X_test,
        'y_test_binario': y_test_binario,
        'y_test_ternario': y_test_ternario,
        'feature_cols': feature_cols
    }


def encontrar_threshold_optimo(y_true, y_pred_proba):
    """
    Encuentra el threshold que maximiza la ganancia.
    """
    # Ordenar por probabilidad descendente
    sorted_indices = np.argsort(y_pred_proba)[::-1]
    y_true_sorted = y_true[sorted_indices]
    proba_sorted = y_pred_proba[sorted_indices]
    
    # Calcular ganancia acumulada
    ganancias_individuales = np.where(
        y_true_sorted == 1,
        GANANCIA_ACIERTO,
        -COSTO_ESTIMULO
    )
    
    ganancia_acumulada = np.cumsum(ganancias_individuales)
    
    # Encontrar máximo
    idx_max = np.argmax(ganancia_acumulada)
    ganancia_max = ganancia_acumulada[idx_max]
    threshold_optimo = proba_sorted[idx_max]
    envios_optimos = idx_max + 1
    
    return threshold_optimo, ganancia_max, envios_optimos


def evaluar_ensemble_simple(cp, datos):
    """
    Evalúa ensemble simple (promedio sin pesos).
    """
    logger.info("=== EVALUANDO ENSEMBLE SIMPLE ===")
    
    results = cp.predict_with_confidence(datos['X_test'])
    probs = results['probabilities']
    
    threshold_opt, ganancia_max, envios_opt = encontrar_threshold_optimo(
        datos['y_test_ternario'],
        probs
    )
    
    logger.info(f"Threshold óptimo: {threshold_opt:.6f}")
    logger.info(f"Ganancia máxima: {ganancia_max:,.0f}")
    logger.info(f"Envíos óptimos: {envios_opt:,} ({envios_opt/len(probs)*100:.2f}%)")
    
    return {
        'strategy': 'simple',
        'threshold': float(threshold_opt),
        'ganancia': float(ganancia_max),
        'envios': int(envios_opt),
        'porcentaje_envios': float(envios_opt / len(probs) * 100)
    }


def evaluar_ensemble_peso_fijo(cp, datos):
    """
    Evalúa ensemble con peso fijo por modelo basado en confianza promedio.
    """
    logger.info("=== EVALUANDO ENSEMBLE CON PESO FIJO POR MODELO ===")
    
    # Calcular confianza de cada modelo en calibración
    confidences_per_model = []
    
    for i, model in enumerate(cp.models):
        probs_cal = model.predict(datos['X_cal'])
        
        scores = np.array([
            1 - probs_cal[j] if datos['y_cal'][j] == 1 else probs_cal[j]
            for j in range(len(datos['y_cal']))
        ])
        
        alpha_promedio = np.mean([np.mean(cp.calibration_scores >= s) for s in scores[:100]])
        confidence = 1 - alpha_promedio
        
        confidences_per_model.append(confidence)
        logger.info(f"Modelo {i+1}: confianza={confidence:.4f}")
    
    # Normalizar pesos
    weights = np.array(confidences_per_model)
    weights = weights / weights.sum()
    
    logger.info(f"Pesos - min: {weights.min():.4f}, max: {weights.max():.4f}, std: {weights.std():.4f}")
    
    # Predecir en test con pesos
    all_probs = []
    for model in cp.models:
        probs = model.predict(datos['X_test'])
        all_probs.append(probs)
    
    probs_matrix = np.array(all_probs)
    probs_weighted = np.average(probs_matrix, axis=0, weights=weights)
    
    threshold_opt, ganancia_max, envios_opt = encontrar_threshold_optimo(
        datos['y_test_ternario'],
        probs_weighted
    )
    
    logger.info(f"Threshold óptimo: {threshold_opt:.6f}")
    logger.info(f"Ganancia máxima: {ganancia_max:,.0f}")
    logger.info(f"Envíos óptimos: {envios_opt:,} ({envios_opt/len(probs_weighted)*100:.2f}%)")
    
    return {
        'strategy': 'peso_fijo',
        'threshold': float(threshold_opt),
        'ganancia': float(ganancia_max),
        'envios': int(envios_opt),
        'porcentaje_envios': float(envios_opt / len(probs_weighted) * 100),
        'weights': weights.tolist()
    }


def evaluar_ensemble_peso_dinamico(cp, datos):
    """
    Evalúa ensemble con peso dinámico por predicción.
    """
    logger.info("=== EVALUANDO ENSEMBLE CON PESO DINÁMICO ===")
    
    all_probs = []
    all_confidences = []
    
    for i, model in enumerate(cp.models):
        logger.info(f"Procesando modelo {i+1}/{len(cp.models)}...")
        
        probs = model.predict(datos['X_test'])
        all_probs.append(probs)
        
        confidences = []
        for prob in probs:
            score_0 = prob
            score_1 = 1 - prob
            
            quantile_0 = np.mean(cp.calibration_scores >= score_0)
            quantile_1 = np.mean(cp.calibration_scores >= score_1)
            
            alpha = max(quantile_0, quantile_1)
            confidence = 1 - alpha
            confidences.append(confidence)
        
        all_confidences.append(np.array(confidences))
    
    probs_matrix = np.array(all_probs)
    confidences_matrix = np.array(all_confidences)
    
    # Ponderar por confianza individual
    probs_dynamic = np.zeros(probs_matrix.shape[1])
    
    for i in range(probs_matrix.shape[1]):
        weights_i = confidences_matrix[:, i]
        weights_i = weights_i / (weights_i.sum() + 1e-10)
        probs_dynamic[i] = np.average(probs_matrix[:, i], weights=weights_i)
    
    logger.info(f"Confianza promedio: {confidences_matrix.mean():.4f}")
    
    threshold_opt, ganancia_max, envios_opt = encontrar_threshold_optimo(
        datos['y_test_ternario'],
        probs_dynamic
    )
    
    logger.info(f"Threshold óptimo: {threshold_opt:.6f}")
    logger.info(f"Ganancia máxima: {ganancia_max:,.0f}")
    logger.info(f"Envíos óptimos: {envios_opt:,} ({envios_opt/len(probs_dynamic)*100:.2f}%)")
    
    return {
        'strategy': 'peso_dinamico',
        'threshold': float(threshold_opt),
        'ganancia': float(ganancia_max),
        'envios': int(envios_opt),
        'porcentaje_envios': float(envios_opt / len(probs_dynamic) * 100)
    }


def guardar_resultados(models, feature_cols, resultados, best_params, best_iteration):
    """
    Guarda modelos y resultados, luego sincroniza con GCS.
    """
    logger.info("=== GUARDANDO RESULTADOS ===")
    
    local_path = os.path.expanduser("~/temp_conformal_output")
    os.makedirs(local_path, exist_ok=True)
    
    modelos_path = os.path.join(local_path, "modelos")
    resultados_path = os.path.join(local_path, "resultados")
    
    os.makedirs(modelos_path, exist_ok=True)
    os.makedirs(resultados_path, exist_ok=True)
    
    # Guardar modelos
    for i, (model, semilla) in enumerate(zip(models, SEMILLAS)):
        archivo = os.path.join(modelos_path, f"{STUDY_NAME}_conformal_seed_{semilla}.txt")
        model.save_model(archivo)
        logger.info(f"Modelo {i+1}/{len(models)} guardado")
    
    # Guardar ensemble
    ensemble_data = {
        'models': models,
        'feature_cols': feature_cols,
        'best_params': best_params,
        'best_iteration': best_iteration,
        'semillas': SEMILLAS,
        'datetime': datetime.now().isoformat()
    }
    
    archivo_ensemble = os.path.join(modelos_path, f"{STUDY_NAME}_conformal_ensemble.pkl")
    with open(archivo_ensemble, 'wb') as f:
        pickle.dump(ensemble_data, f)
    
    # Guardar resultados
    resultados_completos = {
        'study_name': STUDY_NAME,
        'configuracion': {
            'periodos_train': [p for p in PERIODOS_TRAIN if p not in ['202106', '202107']],
            'mes_calibracion': '202106',
            'mes_test': '202107',
            'undersampling_ratio': UNDERSAMPLING_RATIO,
            'n_modelos': len(SEMILLAS),
            'semillas': SEMILLAS,
            'best_params': best_params,
            'best_iteration': best_iteration
        },
        'resultados_por_estrategia': resultados,
        'datetime': datetime.now().isoformat()
    }
    
    archivo_resultados = os.path.join(resultados_path, f"{STUDY_NAME}_conformal_results.json")
    with open(archivo_resultados, 'w') as f:
        json.dump(resultados_completos, f, indent=2)
    
    logger.info("Resultados guardados localmente")
    
    # Sincronizar con GCS
    logger.info("Sincronizando con GCS...")
    gcs_path = f'{BUCKET_NAME}conformal_output/'
    
    try:
        subprocess.run(
            ['gsutil', '-m', 'rsync', '-r', local_path, gcs_path],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"✓ Sincronizado con GCS: {gcs_path}")
        
        import shutil
        shutil.rmtree(local_path)
        logger.info("✓ Archivos temporales eliminados")
        
    except Exception as e:
        logger.warning(f"Error sincronizando: {e}")


def main():
    """
    Pipeline principal.
    """
    logger.info("=== INICIANDO PIPELINE DE CONFORMAL PREDICTION ===")
    
    conn = None
    try:
        # Conectar a DuckDB
        conn = duckdb.connect(database=':memory:')
        
        from google.auth import default
        from google.auth.transport.requests import Request
        
        credentials, project = default()
        credentials.refresh(Request())
        token = credentials.token
        
        conn.execute("INSTALL httpfs;")
        conn.execute("LOAD httpfs;")
        conn.execute(f"""
            CREATE SECRET (
                TYPE GCS,
                PROVIDER config,
                BEARER_TOKEN '{token}'
            )
        """)
        
        # Cargar datos
        conn = create_sql_table_from_parquet_csv(conn, DATA_PATH_OPT, SQL_TABLE_NAME)
        
        # Cargar hiperparámetros
        best_params, best_iteration = cargar_hiperparametros_optuna()
        
        # Re-entrenar hasta 202105
        models, feature_cols = re_entrenar_modelos(conn, SQL_TABLE_NAME, best_params, best_iteration)
        
        # Cargar calibración (202106) y test (202107)
        datos = cargar_datos_calibracion_test(conn, SQL_TABLE_NAME)
        
        # Crear predictor y calibrar
        cp = ConformalPredictor(models, feature_cols)
        cp.fit_calibration(datos['X_cal'], datos['y_cal'])
        
        # Evaluar estrategias
        resultados = []
        
        resultado_simple = evaluar_ensemble_simple(cp, datos)
        resultados.append(resultado_simple)
        
        resultado_fijo = evaluar_ensemble_peso_fijo(cp, datos)
        resultados.append(resultado_fijo)
        
        resultado_dinamico = evaluar_ensemble_peso_dinamico(cp, datos)
        resultados.append(resultado_dinamico)
        
        # Comparar
        logger.info("\n" + "="*80)
        logger.info("COMPARACIÓN DE ESTRATEGIAS")
        logger.info("="*80)
        
        for res in resultados:
            logger.info(f"\n{res['strategy'].upper()}:")
            logger.info(f"  Ganancia: {res['ganancia']:,.0f}")
            logger.info(f"  Threshold: {res['threshold']:.6f}")
            logger.info(f"  Envíos: {res['envios']:,} ({res['porcentaje_envios']:.2f}%)")
        
        mejor = max(resultados, key=lambda x: x['ganancia'])
        logger.info(f"\n🏆 MEJOR ESTRATEGIA: {mejor['strategy'].upper()}")
        logger.info(f"   Ganancia: {mejor['ganancia']:,.0f}")
        
        # Guardar todo
        guardar_resultados(models, feature_cols, resultados, best_params, best_iteration)
        
        logger.info("\n=== PIPELINE COMPLETADO ===")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise
    
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    main()