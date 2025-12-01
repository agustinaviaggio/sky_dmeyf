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
    
    def evaluar_modelo_individual(self, model_idx, X_test, y_test=None):
        """Evalúa un modelo individual con métricas de conformal prediction."""
        probs = self.models[model_idx].predict(X_test)
        
        confidences = []
        alphas = []
        
        for prob in probs:
            conf, alpha = self.calcular_confianza_individual(prob)
            confidences.append(conf)
            alphas.append(alpha)
        
        result = {
            'probabilities': probs,
            'confidences': np.array(confidences),
            'alphas': np.array(alphas),
            'confidence_mean': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'confidence_min': np.min(confidences),
            'confidence_max': np.max(confidences)
        }
        
        # Calcular calibración si tenemos y_test
        if y_test is not None:
            result['calibration'] = self._calcular_calibracion(probs, confidences, y_test)
        
        return result
    
    def evaluar_ensemble(self, X_test, y_test=None):
        """Evalúa el ensemble con métricas de conformal prediction."""
        all_probs = [model.predict(X_test) for model in self.models]
        probs_ensemble = np.mean(all_probs, axis=0)
        
        confidences = []
        alphas = []
        
        for prob in probs_ensemble:
            conf, alpha = self.calcular_confianza_individual(prob)
            confidences.append(conf)
            alphas.append(alpha)
        
        result = {
            'probabilities': probs_ensemble,
            'confidences': np.array(confidences),
            'alphas': np.array(alphas),
            'confidence_mean': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'confidence_min': np.min(confidences),
            'confidence_max': np.max(confidences)
        }
        
        # Calcular calibración si tenemos y_test
        if y_test is not None:
            result['calibration'] = self._calcular_calibracion(probs_ensemble, confidences, y_test)
        
        # Calcular variabilidad interna entre modelos
        result['variabilidad_interna'] = self._calcular_variabilidad_interna(all_probs)
        
        return result
    
    def _calcular_calibracion(self, probs, confidences, y_true):
        """Calcula Expected Calibration Error (ECE)."""
        # Crear bins por nivel de confianza
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        ece = 0.0
        bin_stats = []
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # Casos en este bin
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            
            if i == n_bins - 1:  # Último bin incluye el límite superior
                in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
            
            n_in_bin = np.sum(in_bin)
            
            if n_in_bin > 0:
                # Confianza promedio en el bin
                conf_in_bin = np.mean(confidences[in_bin])
                
                # Accuracy en el bin (predicción correcta si prob>0.5 y y=1, o prob<0.5 y y=0)
                preds_in_bin = (probs[in_bin] > 0.5).astype(int)
                acc_in_bin = np.mean(preds_in_bin == y_true[in_bin])
                
                # Contribución al ECE
                ece += (n_in_bin / len(y_true)) * abs(acc_in_bin - conf_in_bin)
                
                bin_stats.append({
                    'bin_lower': float(bin_lower),
                    'bin_upper': float(bin_upper),
                    'n_samples': int(n_in_bin),
                    'confidence_mean': float(conf_in_bin),
                    'accuracy': float(acc_in_bin),
                    'gap': float(abs(acc_in_bin - conf_in_bin))
                })
        
        return {
            'ece': float(ece),
            'bin_stats': bin_stats
        }
    
    def _calcular_variabilidad_interna(self, all_probs):
        """Calcula variabilidad entre los modelos individuales."""
        # Convertir a array [n_models, n_samples]
        probs_array = np.array(all_probs)
        
        # Variabilidad por muestra (std entre modelos)
        std_por_muestra = np.std(probs_array, axis=0)
        
        return {
            'std_mean': float(np.mean(std_por_muestra)),
            'std_std': float(np.std(std_por_muestra)),
            'std_max': float(np.max(std_por_muestra)),
            'std_percentile_95': float(np.percentile(std_por_muestra, 95))
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
            resultado = cp.evaluar_modelo_individual(i, X_test, y_test)
            
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
                'calibration_ece': float(resultado['calibration']['ece']),
                **ganancia_info
            })
        
        # 9. Evaluar ensemble
        logger.info(f"{study_name}: Evaluando ensemble...")
        resultado_ensemble = cp.evaluar_ensemble(X_test, y_test)
        
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
            'calibration_ece': float(resultado_ensemble['calibration']['ece']),
            'calibration_bins': resultado_ensemble['calibration']['bin_stats'],
            'variabilidad_interna': resultado_ensemble['variabilidad_interna'],
            'probabilidades': resultado_ensemble['probabilities'].tolist(),  # Para correlaciones
            **ganancia_ensemble
        }
        
        logger.info(f"{study_name}: Ganancia ensemble = {ganancia_ensemble['ganancia']:,.0f}")
        logger.info(f"{study_name}: Confianza media = {resultado_ensemble['confidence_mean']:.4f}")
        logger.info(f"{study_name}: ECE = {resultado_ensemble['calibration']['ece']:.4f}")
        logger.info(f"{study_name}: Variabilidad interna (std mean) = {resultado_ensemble['variabilidad_interna']['std_mean']:.4f}")
        
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


def calcular_correlaciones_entre_ensembles(resultados):
    """Calcula matriz de correlación entre las predicciones de los ensembles."""
    logger.info("Calculando correlaciones entre ensembles...")
    
    # Extraer probabilidades de cada ensemble
    estudios = [r['study_name'] for r in resultados]
    probs_dict = {r['study_name']: np.array(r['resultado_ensemble']['probabilidades']) 
                  for r in resultados}
    
    n = len(estudios)
    correlation_matrix = np.zeros((n, n))
    
    for i, estudio_i in enumerate(estudios):
        for j, estudio_j in enumerate(estudios):
            if i == j:
                correlation_matrix[i, j] = 1.0
            else:
                corr = np.corrcoef(probs_dict[estudio_i], probs_dict[estudio_j])[0, 1]
                correlation_matrix[i, j] = corr
    
    # Convertir a formato serializable
    correlation_data = {
        'estudios': estudios,
        'matrix': correlation_matrix.tolist(),
        'correlaciones_pares': []
    }
    
    # Generar lista de pares con correlaciones
    for i in range(n):
        for j in range(i+1, n):
            correlation_data['correlaciones_pares'].append({
                'estudio_1': estudios[i],
                'estudio_2': estudios[j],
                'correlacion': float(correlation_matrix[i, j])
            })
    
    # Ordenar por correlación (menor = más complementarios)
    correlation_data['correlaciones_pares'] = sorted(
        correlation_data['correlaciones_pares'],
        key=lambda x: x['correlacion']
    )
    
    logger.info(f"  Correlación más baja: {correlation_data['correlaciones_pares'][0]['correlacion']:.4f}")
    logger.info(f"  Correlación más alta: {correlation_data['correlaciones_pares'][-1]['correlacion']:.4f}")
    
    return correlation_data


def sugerir_estrategia_ensemble(resultados, correlaciones):
    """Sugiere estrategia de ensamblado basada en métricas."""
    logger.info("\nAnalizando estrategia de ensemble...")
    
    # Ordenar por ECE (menor es mejor)
    estudios_por_calibracion = sorted(
        resultados,
        key=lambda x: x['resultado_ensemble']['calibration_ece']
    )
    
    # Ordenar por ganancia
    estudios_por_ganancia = sorted(
        resultados,
        key=lambda x: x['resultado_ensemble']['ganancia'],
        reverse=True
    )
    
    # Ordenar por variabilidad interna (menor es mejor = más estable)
    estudios_por_estabilidad = sorted(
        resultados,
        key=lambda x: x['resultado_ensemble']['variabilidad_interna']['std_mean']
    )
    
    # Calcular pesos basados en calibración inversa
    eces = [r['resultado_ensemble']['calibration_ece'] for r in resultados]
    ece_min = min(eces)
    ece_max = max(eces)
    
    # Normalizar ECE invertido (menor ECE = mayor peso)
    if ece_max > ece_min:
        pesos_calibracion = [(ece_max - ece) / (ece_max - ece_min) for ece in eces]
    else:
        pesos_calibracion = [1.0] * len(eces)
    
    # Normalizar para que sumen 1
    suma_pesos = sum(pesos_calibracion)
    pesos_calibracion = [p / suma_pesos for p in pesos_calibracion]
    
    # Identificar pares más complementarios (menor correlación)
    pares_complementarios = correlaciones['correlaciones_pares'][:3]  # Top 3
    
    sugerencias = {
        'ranking_por_calibracion': [
            {
                'study_name': r['study_name'],
                'ece': r['resultado_ensemble']['calibration_ece'],
                'ganancia': r['resultado_ensemble']['ganancia']
            }
            for r in estudios_por_calibracion
        ],
        'ranking_por_ganancia': [
            {
                'study_name': r['study_name'],
                'ganancia': r['resultado_ensemble']['ganancia'],
                'ece': r['resultado_ensemble']['calibration_ece']
            }
            for r in estudios_por_ganancia
        ],
        'ranking_por_estabilidad': [
            {
                'study_name': r['study_name'],
                'std_mean': r['resultado_ensemble']['variabilidad_interna']['std_mean'],
                'ganancia': r['resultado_ensemble']['ganancia']
            }
            for r in estudios_por_estabilidad
        ],
        'pares_mas_complementarios': pares_complementarios,
        'pesos_sugeridos_por_calibracion': [
            {
                'study_name': r['study_name'],
                'peso': float(pesos_calibracion[i]),
                'ece': r['resultado_ensemble']['calibration_ece']
            }
            for i, r in enumerate(resultados)
        ],
        'recomendacion': generar_recomendacion(
            estudios_por_calibracion,
            estudios_por_ganancia,
            pares_complementarios,
            correlaciones
        )
    }
    
    return sugerencias


def generar_recomendacion(por_calibracion, por_ganancia, pares_compl, correlaciones):
    """Genera recomendación textual de estrategia."""
    mejor_calibrado = por_calibracion[0]['study_name']
    mejor_ganancia = por_ganancia[0]['study_name']
    
    # Calcular correlación promedio
    corrs = [p['correlacion'] for p in correlaciones['correlaciones_pares']]
    corr_promedio = np.mean(corrs)
    
    recomendacion = {
        'estrategia_principal': '',
        'razonamiento': []
    }
    
    if corr_promedio > 0.95:
        recomendacion['estrategia_principal'] = 'MODELO_UNICO'
        recomendacion['razonamiento'].append(
            f"Correlación promedio muy alta ({corr_promedio:.3f}). Los modelos predicen muy similar."
        )
        recomendacion['razonamiento'].append(
            f"Sugerencia: Usar solo el mejor calibrado ({mejor_calibrado}) o el de mayor ganancia ({mejor_ganancia})."
        )
    elif corr_promedio > 0.85:
        recomendacion['estrategia_principal'] = 'ENSEMBLE_PESADO'
        recomendacion['razonamiento'].append(
            f"Correlación promedio alta ({corr_promedio:.3f}). Modelos similares pero con diferencias."
        )
        recomendacion['razonamiento'].append(
            "Sugerencia: Ensemble con pesos según calibración (ECE inverso)."
        )
    else:
        recomendacion['estrategia_principal'] = 'ENSEMBLE_DIVERSO'
        recomendacion['razonamiento'].append(
            f"Correlación promedio moderada ({corr_promedio:.3f}). Modelos complementarios."
        )
        recomendacion['razonamiento'].append(
            "Sugerencia: Incluir todos los modelos, pesados por calibración."
        )
        if len(pares_compl) > 0:
            par = pares_compl[0]
            recomendacion['razonamiento'].append(
                f"Par más complementario: {par['estudio_1']} + {par['estudio_2']} (corr={par['correlacion']:.3f})"
            )
    
    return recomendacion


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
        
        # Guardar ensemble (sin probabilidades para ahorrar espacio)
        resultado_sin_probs = {k: v for k, v in estudio['resultado_ensemble'].items() 
                               if k != 'probabilidades'}
        with open(estudio_path / "resultados_ensemble.json", 'w') as f:
            json.dump(resultado_sin_probs, f, indent=2)
        
        # Guardar config
        with open(estudio_path / "config.json", 'w') as f:
            json.dump(estudio['config'], f, indent=2)
    
    # Calcular correlaciones entre ensembles
    correlaciones = calcular_correlaciones_entre_ensembles(resultados)
    
    with open(local_temp / "correlaciones_ensembles.json", 'w') as f:
        json.dump(correlaciones, f, indent=2)
    
    # Sugerir estrategia
    sugerencias = sugerir_estrategia_ensemble(resultados, correlaciones)
    
    with open(local_temp / "estrategia_sugerida.json", 'w') as f:
        json.dump(sugerencias, f, indent=2)
    
    # Reporte comparativo
    comparacion = {
        'fecha': datetime.now().isoformat(),
        'estudios_evaluados': [r['study_name'] for r in resultados],
        'comparacion': [
            {
                'study_name': r['study_name'],
                'ganancia_ensemble': r['resultado_ensemble']['ganancia'],
                'confidence_mean': r['resultado_ensemble']['confidence_mean'],
                'calibration_ece': r['resultado_ensemble']['calibration_ece'],
                'variabilidad_std_mean': r['resultado_ensemble']['variabilidad_interna']['std_mean'],
                'threshold': r['resultado_ensemble']['threshold'],
                'envios': r['resultado_ensemble']['envios'],
                'n_features': r['config']['n_features']
            }
            for r in resultados
        ],
        'estrategia_sugerida': sugerencias['recomendacion']
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
    
    return sugerencias


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
    
    # Guardar resultados y obtener sugerencias
    sugerencias = guardar_resultados(resultados)
    
    # Imprimir resumen
    logger.info("\n" + "="*80)
    logger.info("RESUMEN COMPARATIVO")
    logger.info("="*80)
    
    for r in sorted(resultados, key=lambda x: x['resultado_ensemble']['ganancia'], reverse=True):
        logger.info(f"\n{r['study_name']}:")
        logger.info(f"  Ganancia:        ${r['resultado_ensemble']['ganancia']:>15,.0f}")
        logger.info(f"  Confianza media: {r['resultado_ensemble']['confidence_mean']:>15.4f}")
        logger.info(f"  ECE:             {r['resultado_ensemble']['calibration_ece']:>15.4f}")
        logger.info(f"  Var. interna:    {r['resultado_ensemble']['variabilidad_interna']['std_mean']:>15.4f}")
        logger.info(f"  Threshold:       {r['resultado_ensemble']['threshold']:>15.6f}")
        logger.info(f"  Features:        {r['config']['n_features']:>15,}")
    
    # Imprimir estrategia sugerida
    logger.info("\n" + "="*80)
    logger.info("ESTRATEGIA DE ENSEMBLE SUGERIDA")
    logger.info("="*80)
    logger.info(f"\nEstrategia: {sugerencias['recomendacion']['estrategia_principal']}")
    for razon in sugerencias['recomendacion']['razonamiento']:
        logger.info(f"  • {razon}")
    
    logger.info("\n" + "-"*80)
    logger.info("RANKING POR CALIBRACIÓN (menor ECE = mejor):")
    for i, top in enumerate(sugerencias['ranking_por_calibracion'], 1):
        logger.info(f"  {i}. {top['study_name']}: ECE={top['ece']:.4f}, Ganancia=${top['ganancia']:,.0f}")
    
    logger.info("\n" + "-"*80)
    logger.info("PESOS SUGERIDOS (basados en calibración):")
    pesos_sorted = sorted(sugerencias['pesos_sugeridos_por_calibracion'], 
                         key=lambda x: x['peso'], reverse=True)
    for p in pesos_sorted:
        logger.info(f"  {p['study_name']}: {p['peso']:.3f} (ECE={p['ece']:.4f})")
    
    logger.info("\n" + "-"*80)
    logger.info("PARES MÁS COMPLEMENTARIOS (menor correlación):")
    for par in sugerencias['pares_mas_complementarios']:
        logger.info(f"  {par['estudio_1']} + {par['estudio_2']}: corr={par['correlacion']:.4f}")
    
    logger.info("\n✓ Pipeline maestro completado")


if __name__ == "__main__":
    main()