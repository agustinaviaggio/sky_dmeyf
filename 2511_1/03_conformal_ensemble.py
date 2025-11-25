import duckdb
import logging
import numpy as np
import pickle
import json
import os
import gc
from datetime import datetime
from sklearn.model_selection import train_test_split
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
        """
        Args:
            models: Lista de modelos LightGBM entrenados
            feature_cols: Lista de nombres de features
        """
        self.models = models
        self.feature_cols = feature_cols
        self.calibration_scores = None
        self.n_models = len(models)
        
    def fit_calibration(self, X_cal, y_cal):
        """
        Calcula nonconformity scores en conjunto de calibración.
        
        Args:
            X_cal: Features de calibración
            y_cal: Labels de calibración
        """
        logger.info("Calculando nonconformity scores en calibración...")
        
        # Obtener predicciones de todos los modelos
        all_probs = []
        for i, model in enumerate(self.models):
            probs = model.predict(X_cal)
            all_probs.append(probs)
        
        # Promedio de probabilidades (ensemble)
        probs_ensemble = np.mean(all_probs, axis=0)
        
        # Nonconformity score: 1 - probabilidad de la clase verdadera
        # Si y=1 y prob=0.9 -> score=0.1 (conformidad alta)
        # Si y=1 y prob=0.2 -> score=0.8 (conformidad baja)
        self.calibration_scores = np.array([
            1 - probs_ensemble[i] if y_cal[i] == 1 else probs_ensemble[i]
            for i in range(len(y_cal))
        ])
        
        logger.info(f"Calibración completada. Scores: min={self.calibration_scores.min():.4f}, "
                   f"max={self.calibration_scores.max():.4f}, "
                   f"mean={self.calibration_scores.mean():.4f}")
        
        return self
    
    def predict_with_confidence(self, X_test, use_individual_alphas=False):
        """
        Predice con métricas de confianza de conformal prediction.
        
        Args:
            X_test: Features de test
            use_individual_alphas: Si True, calcula alpha individual por predicción
            
        Returns:
            dict con predicciones, probabilidades y métricas de confianza
        """
        logger.info(f"Generando predicciones con conformal prediction (individual_alphas={use_individual_alphas})...")
        
        # Obtener predicciones de todos los modelos
        all_probs = []
        for model in self.models:
            probs = model.predict(X_test)
            all_probs.append(probs)
        
        probs_matrix = np.array(all_probs)  # shape: (n_models, n_samples)
        probs_ensemble = probs_matrix.mean(axis=0)
        
        results = {
            'probabilities': probs_ensemble,
            'predictions': (probs_ensemble > 0.025).astype(int),
            'probs_per_model': probs_matrix
        }
        
        if use_individual_alphas:
            # Calcular alpha individual para cada predicción
            individual_alphas = []
            individual_confidences = []
            
            for prob in probs_ensemble:
                # Nonconformity scores para cada clase
                score_class_0 = prob  # si predigo 0, cuán no-conforme es
                score_class_1 = 1 - prob  # si predigo 1, cuán no-conforme es
                
                # Percentil en calibración
                quantile_0 = np.mean(self.calibration_scores >= score_class_0)
                quantile_1 = np.mean(self.calibration_scores >= score_class_1)
                
                # Alpha individual: el máximo de los dos quantiles
                alpha = max(quantile_0, quantile_1)
                confidence = 1 - alpha
                
                individual_alphas.append(alpha)
                individual_confidences.append(confidence)
            
            results['individual_alphas'] = np.array(individual_alphas)
            results['individual_confidences'] = np.array(individual_confidences)
            
            logger.info(f"Confianza promedio: {np.mean(individual_confidences):.4f}")
            logger.info(f"Confianza min/max: {np.min(individual_confidences):.4f}/{np.max(individual_confidences):.4f}")
        
        return results


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
    
    # Descargar DB desde GCS
    try:
        logger.info(f"Descargando DB desde {gcs_path}...")
        result = subprocess.run(
            ['gsutil', 'cp', gcs_path, db_file],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info("✓ DB descargada exitosamente")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error descargando DB: {e.stderr}")
        raise
    
    # Cargar estudio
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
    Re-entrena 25 modelos hasta 202106 con undersampling.
    """
    logger.info("=== RE-ENTRENAMIENTO DE MODELOS HASTA 202106 ===")
    
    # Períodos de entrenamiento (hasta 202106, excluyendo 202107)
    periodos_train = [p for p in PERIODOS_TRAIN if p != '202107']
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
                    if col not in ['target_binario', 'target_ternario']]
    
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
    
    # Limpiar memoria
    del X_train, y_train, train_data
    gc.collect()
    
    return models, feature_cols


def dividir_202107(conn, tabla, test_size=0.5, random_state=42):
    """
    Divide 202107 en calibración y evaluación de forma aleatoria.
    """
    logger.info(f"=== DIVIDIENDO 202107 (calibración {(1-test_size)*100:.0f}% - evaluación {test_size*100:.0f}%) ===")
    
    # Cargar datos de 202107
    query_202107 = f"SELECT * FROM {tabla} WHERE foto_mes = 202107"
    data_202107 = conn.execute(query_202107).fetchnumpy()
    
    logger.info(f"Total registros 202107: {len(data_202107['target_binario']):,}")
    
    # Preparar features
    feature_cols = [col for col in data_202107.keys() 
                    if col not in ['target_binario', 'target_ternario']]
    
    X = np.column_stack([data_202107[col] for col in feature_cols])
    y_binario = data_202107['target_binario']
    y_ternario = data_202107['target_ternario']
    numero_cliente = data_202107['numero_de_cliente']
    
    # Split estratificado por target_binario
    indices = np.arange(len(y_binario))
    idx_cal, idx_eval = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y_binario
    )
    
    # Dividir datos
    X_cal = X[idx_cal]
    y_cal_binario = y_binario[idx_cal]
    
    X_eval = X[idx_eval]
    y_eval_binario = y_binario[idx_eval]
    y_eval_ternario = y_ternario[idx_eval]
    cliente_eval = numero_cliente[idx_eval]
    
    logger.info(f"Calibración: {len(X_cal):,} registros")
    logger.info(f"  Clase 0: {(y_cal_binario==0).sum():,} | Clase 1: {(y_cal_binario==1).sum():,}")
    logger.info(f"Evaluación: {len(X_eval):,} registros")
    logger.info(f"  Clase 0: {(y_eval_binario==0).sum():,} | Clase 1: {(y_eval_binario==1).sum():,}")
    logger.info(f"  BAJA+2 (objetivo): {(y_eval_ternario==1).sum():,} ({(y_eval_ternario==1).sum()/len(y_eval_ternario)*100:.2f}%)")
    
    # Limpiar
    del data_202107, X, y_binario, y_ternario, numero_cliente
    gc.collect()
    
    return {
        'X_cal': X_cal,
        'y_cal': y_cal_binario,
        'X_eval': X_eval,
        'y_eval_binario': y_eval_binario,
        'y_eval_ternario': y_eval_ternario,
        'cliente_eval': cliente_eval,
        'feature_cols': feature_cols
    }


def calcular_ganancia_con_threshold(y_true, y_pred_proba, threshold=0.025):
    """
    Calcula ganancia usando threshold óptimo.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Ganancia
    ganancia = np.sum(
        (y_true == 1) & (y_pred == 1) * GANANCIA_ACIERTO +
        (y_true == 0) & (y_pred == 1) * (-COSTO_ESTIMULO)
    )
    
    envios = y_pred.sum()
    
    return ganancia, envios


def encontrar_threshold_optimo(y_true, y_pred_proba):
    """
    Encuentra el threshold que maximiza la ganancia.
    """
    # Ordenar por probabilidad
    sorted_indices = np.argsort(y_pred_proba)[::-1]
    y_true_sorted = y_true[sorted_indices]
    
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
    threshold_optimo = y_pred_proba[sorted_indices[idx_max]]
    envios_optimos = idx_max + 1
    
    return threshold_optimo, ganancia_max, envios_optimos


def evaluar_ensemble_simple(cp, data_eval):
    """
    Evalúa ensemble simple (promedio sin pesos).
    """
    logger.info("=== EVALUANDO ENSEMBLE SIMPLE ===")
    
    results = cp.predict_with_confidence(data_eval['X_eval'], use_individual_alphas=False)
    probs = results['probabilities']
    
    # Encontrar threshold óptimo
    threshold_opt, ganancia_max, envios_opt = encontrar_threshold_optimo(
        data_eval['y_eval_ternario'],
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
        'porcentaje_envios': float(envios_opt / len(probs) * 100),
        'probs': probs
    }


def evaluar_ensemble_peso_fijo(cp, data_eval):
    """
    Evalúa ensemble con peso fijo por modelo basado en confianza promedio.
    """
    logger.info("=== EVALUANDO ENSEMBLE CON PESO FIJO POR MODELO ===")
    
    # Primero calcular confianza de cada modelo en calibración
    logger.info("Calculando confianza de cada modelo en calibración...")
    
    confidences_per_model = []
    
    for i, model in enumerate(cp.models):
        # Predecir en calibración
        probs_cal = model.predict(data_eval['X_cal'])
        
        # Calcular nonconformity scores
        scores = np.array([
            1 - probs_cal[j] if data_eval['y_cal'][j] == 1 else probs_cal[j]
            for j in range(len(data_eval['y_cal']))
        ])
        
        # Confianza promedio = 1 - alpha_promedio
        # Alpha es el percentil del score en calibración
        alpha_promedio = np.mean([np.mean(cp.calibration_scores >= s) for s in scores[:100]])  # sample para eficiencia
        confidence = 1 - alpha_promedio
        
        confidences_per_model.append(confidence)
        logger.info(f"Modelo {i+1}: confianza={confidence:.4f}")
    
    # Normalizar pesos
    weights = np.array(confidences_per_model)
    weights = weights / weights.sum()
    
    logger.info(f"Pesos normalizados - min: {weights.min():.4f}, max: {weights.max():.4f}, std: {weights.std():.4f}")
    
    # Predecir en evaluación con pesos
    all_probs = []
    for model in cp.models:
        probs = model.predict(data_eval['X_eval'])
        all_probs.append(probs)
    
    probs_matrix = np.array(all_probs)
    probs_weighted = np.average(probs_matrix, axis=0, weights=weights)
    
    # Encontrar threshold óptimo
    threshold_opt, ganancia_max, envios_opt = encontrar_threshold_optimo(
        data_eval['y_eval_ternario'],
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
        'weights': weights.tolist(),
        'probs': probs_weighted
    }


def evaluar_ensemble_peso_dinamico(cp, data_eval):
    """
    Evalúa ensemble con peso dinámico por predicción basado en confianza individual.
    """
    logger.info("=== EVALUANDO ENSEMBLE CON PESO DINÁMICO POR PREDICCIÓN ===")
    
    # Obtener confianzas individuales por predicción y por modelo
    all_probs = []
    all_confidences = []
    
    for i, model in enumerate(cp.models):
        logger.info(f"Procesando modelo {i+1}/{len(cp.models)}...")
        
        probs = model.predict(data_eval['X_eval'])
        all_probs.append(probs)
        
        # Calcular confianza individual para cada predicción de este modelo
        confidences = []
        for prob in probs:
            score_class_0 = prob
            score_class_1 = 1 - prob
            
            quantile_0 = np.mean(cp.calibration_scores >= score_class_0)
            quantile_1 = np.mean(cp.calibration_scores >= score_class_1)
            
            alpha = max(quantile_0, quantile_1)
            confidence = 1 - alpha
            
            confidences.append(confidence)
        
        all_confidences.append(np.array(confidences))
    
    probs_matrix = np.array(all_probs)  # shape: (n_models, n_samples)
    confidences_matrix = np.array(all_confidences)  # shape: (n_models, n_samples)
    
    # Para cada predicción, ponderar por confianza
    probs_dynamic = np.zeros(probs_matrix.shape[1])
    
    for i in range(probs_matrix.shape[1]):
        # Pesos para esta predicción
        weights_i = confidences_matrix[:, i]
        weights_i = weights_i / (weights_i.sum() + 1e-10)  # normalizar
        
        # Promedio ponderado
        probs_dynamic[i] = np.average(probs_matrix[:, i], weights=weights_i)
    
    logger.info(f"Confianza promedio global: {confidences_matrix.mean():.4f}")
    logger.info(f"Confianza std: {confidences_matrix.std():.4f}")
    
    # Encontrar threshold óptimo
    threshold_opt, ganancia_max, envios_opt = encontrar_threshold_optimo(
        data_eval['y_eval_ternario'],
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
        'porcentaje_envios': float(envios_opt / len(probs_dynamic) * 100),
        'probs': probs_dynamic,
        'confidences_matrix': confidences_matrix
    }


def guardar_resultados(models, feature_cols, resultados, best_params, best_iteration):
    """
    Guarda modelos y resultados primero localmente y luego sincroniza con GCS.
    """
    logger.info("=== GUARDANDO RESULTADOS ===")
    
    # Crear carpeta local temporal
    local_path = os.path.expanduser("~/modelos_conformal_temp")
    os.makedirs(local_path, exist_ok=True)
    
    # Guardar modelos individuales
    for i, (model, semilla) in enumerate(zip(models, SEMILLAS)):
        archivo_modelo = os.path.join(local_path, f"{STUDY_NAME}_conformal_seed_{semilla}.txt")
        model.save_model(archivo_modelo)
        logger.info(f"Modelo {i+1}/{len(models)} guardado localmente")
    
    # Guardar ensemble completo
    ensemble_data = {
        'models': models,
        'feature_cols': feature_cols,
        'best_params': best_params,
        'best_iteration': best_iteration,
        'semillas': SEMILLAS,
        'datetime': datetime.now().isoformat()
    }
    
    archivo_ensemble = os.path.join(local_path, f"{STUDY_NAME}_conformal_ensemble.pkl")
    with open(archivo_ensemble, 'wb') as f:
        pickle.dump(ensemble_data, f)
    logger.info(f"Ensemble guardado localmente: {archivo_ensemble}")
    
    # Guardar resultados de evaluación
    resultados_completos = {
        'study_name': STUDY_NAME,
        'configuracion': {
            'periodos_train': [p for p in PERIODOS_TRAIN if p != '202107'],
            'mes_calibracion': '202107 (50%)',
            'mes_evaluacion': '202107 (50%)',
            'undersampling_ratio': UNDERSAMPLING_RATIO,
            'n_modelos': len(SEMILLAS),
            'semillas': SEMILLAS,
            'best_params': best_params,
            'best_iteration': best_iteration
        },
        'resultados_por_estrategia': resultados,
        'datetime': datetime.now().isoformat()
    }
    
    archivo_resultados = os.path.join(local_path, f"{STUDY_NAME}_conformal_results.json")
    with open(archivo_resultados, 'w') as f:
        # Convertir arrays numpy a listas para JSON
        resultados_json = resultados_completos.copy()
        for estrategia in resultados_json['resultados_por_estrategia']:
            if 'probs' in estrategia:
                del estrategia['probs']  # No guardar todas las probabilidades
            if 'confidences_matrix' in estrategia:
                del estrategia['confidences_matrix']
        
        json.dump(resultados_json, f, indent=2)
    
    logger.info(f"Resultados guardados localmente: {archivo_resultados}")
    
    # Sincronizar con GCS
    logger.info("Sincronizando con GCS...")
    gcs_path = f'{BUCKET_NAME}modelos_conformal/'
    
    try:
        subprocess.run(
            ['gsutil', '-m', 'rsync', '-r', local_path, gcs_path],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"✓ Sincronizado con GCS: {gcs_path}")
        
        # Limpiar archivos locales después de sincronizar
        import shutil
        shutil.rmtree(local_path)
        logger.info("✓ Archivos temporales eliminados")
        
    except Exception as e:
        logger.warning(f"Error sincronizando: {e}")
        logger.info(f"Archivos guardados localmente en: {local_path}")


def main():
    """
    Pipeline principal de conformal prediction con ensemble.
    """
    logger.info("=== INICIANDO PIPELINE DE CONFORMAL PREDICTION ===")
    
    conn = None
    try:
        # 1. Conectar a DuckDB y configurar GCS
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
        
        # 2. Cargar datos
        conn = create_sql_table_from_parquet_csv(conn, DATA_PATH_OPT, SQL_TABLE_NAME)
        
        # 3. Cargar hiperparámetros de Optuna
        best_params, best_iteration = cargar_hiperparametros_optuna()
        
        # 4. Re-entrenar modelos hasta 202106
        models, feature_cols = re_entrenar_modelos(conn, SQL_TABLE_NAME, best_params, best_iteration)
        
        # 5. Dividir 202107
        data_splits = dividir_202107(conn, SQL_TABLE_NAME, test_size=0.5, random_state=42)
        
        # 6. Crear predictor conformal y calibrar
        cp = ConformalPredictor(models, feature_cols)
        cp.fit_calibration(data_splits['X_cal'], data_splits['y_cal'])
        
        # Agregar X_cal y y_cal a data_eval para uso posterior
        data_eval = {**data_splits, 'X_cal': data_splits['X_cal'], 'y_cal': data_splits['y_cal']}
        
        # 7. Evaluar diferentes estrategias de ensemble
        resultados = []
        
        # Estrategia 1: Simple
        resultado_simple = evaluar_ensemble_simple(cp, data_eval)
        resultados.append(resultado_simple)
        
        # Estrategia 2: Peso fijo
        resultado_fijo = evaluar_ensemble_peso_fijo(cp, data_eval)
        resultados.append(resultado_fijo)
        
        # Estrategia 3: Peso dinámico
        resultado_dinamico = evaluar_ensemble_peso_dinamico(cp, data_eval)
        resultados.append(resultado_dinamico)
        
        # 8. Comparar resultados
        logger.info("\n" + "="*80)
        logger.info("COMPARACIÓN DE ESTRATEGIAS")
        logger.info("="*80)
        
        for res in resultados:
            logger.info(f"\n{res['strategy'].upper()}:")
            logger.info(f"  Ganancia: {res['ganancia']:,.0f}")
            logger.info(f"  Threshold: {res['threshold']:.6f}")
            logger.info(f"  Envíos: {res['envios']:,} ({res['porcentaje_envios']:.2f}%)")
        
        # Identificar mejor estrategia
        mejor = max(resultados, key=lambda x: x['ganancia'])
        logger.info(f"\n🏆 MEJOR ESTRATEGIA: {mejor['strategy'].upper()}")
        logger.info(f"   Ganancia: {mejor['ganancia']:,.0f}")
        
        # 9. Guardar todo
        guardar_resultados(models, feature_cols, resultados, best_params, best_iteration)
        
        logger.info("\n=== PIPELINE COMPLETADO EXITOSAMENTE ===")
        
    except Exception as e:
        logger.error(f"Error durante la ejecución: {e}", exc_info=True)
        raise
    
    finally:
        if conn:
            conn.close()
            logger.info("Conexión a DuckDB cerrada")


if __name__ == "__main__":
    main()