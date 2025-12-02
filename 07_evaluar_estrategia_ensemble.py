import duckdb
import logging
import numpy as np
import json
import os
import gc
import yaml
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
import lightgbm as lgb
import optuna

# Configuración
ESTUDIOS = ['2511_2', '2611_2', '2711_1', '2711_2', '2811_1', '2911_1', '3011_1']

UMBRAL_POR_ESTUDIO = {
    '2511_2': None,
    '2611_2': '50pct',
    '2711_1': '80pct',
    '2711_2': 'union',
    '2811_1': None,
    '2911_1': '50pct',
    '3011_1': 'union',
}

SEMILLAS = [600011, 600043, 600053, 600071, 600073,
            600091, 600107, 600109, 600113, 600137,
            600169, 600179, 600191, 600193, 600197,
            600209, 600211, 600221, 600227, 600253,
            600257, 600259, 600263, 600281, 600293]

# Logging
os.makedirs("logs", exist_ok=True)
fecha = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"logs/log_estrategias_{fecha}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def refrescar_credenciales_gcs():
    """Refresca las credenciales de GCS."""
    try:
        from google.auth import default
        from google.auth.transport.requests import Request
        
        credentials, project = default()
        credentials.refresh(Request())
        
        # Exportar token para gsutil
        os.environ['CLOUDSDK_AUTH_ACCESS_TOKEN'] = credentials.token
        
        return credentials.token
    except Exception as e:
        logger.error(f"Error refrescando credenciales: {e}")
        raise


def cargar_config_estudio(study_name):
    """Carga configuración de un estudio."""
    conf_file = Path(f"~/sky_dmeyf/{study_name}/conf.yaml").expanduser()
    with open(conf_file, 'r') as f:
        config = yaml.safe_load(f)['configuracion']
    return config


def cargar_features_seleccionadas(study_name, bucket_name, umbral):
    """Carga features seleccionadas si aplica."""
    if umbral is None:
        return None
    
    try:
        # Refrescar credenciales antes de acceder a GCS
        refrescar_credenciales_gcs()
        
        if umbral == 'union':
            gcs_pattern = f"{bucket_name}resultados/union_features_{study_name}_*.json"
            tipo = 'union'
        else:
            gcs_pattern = f"{bucket_name}resultados/features_{umbral}_{study_name}_*.json"
            tipo = 'features'
        
        # gsutil ls con timeout
        result = subprocess.run(
            ['gsutil', 'ls', gcs_pattern], 
            capture_output=True, 
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            logger.error(f"No se encontró archivo de features para {study_name}")
            raise Exception(f"gsutil ls falló")
        
        archivos = result.stdout.strip().split('\n')
        if not archivos or archivos[0] == '':
            raise Exception("No se encontraron archivos")
        
        archivo = sorted(archivos)[-1]
        logger.info(f"{study_name}: Archivo encontrado: {archivo.split('/')[-1]}")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # gsutil cp con timeout
            subprocess.run(
                ['gsutil', 'cp', archivo, tmp_path], 
                capture_output=True, 
                text=True,
                timeout=60,
                check=True
            )
            
            with open(tmp_path, 'r') as f:
                data = json.load(f)
            
            if tipo == 'union':
                features = data['union_features']['lista_completa']
            else:
                features = data['features']
            
            return features
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except Exception as e:
        logger.error(f"Error crítico cargando features para {study_name}: {e}")
        raise


def cargar_hiperparametros(study_name, bucket_name):
    """Carga hiperparámetros de Optuna."""
    # Refrescar credenciales antes de acceder a GCS
    refrescar_credenciales_gcs()
    
    local_db_dir = Path.home() / "optuna_db"
    local_db_dir.mkdir(exist_ok=True)
    
    db_file = local_db_dir / f"{study_name}.db"
    gcs_path = f"{bucket_name}optuna_db/{study_name}.db"
    
    subprocess.run(
        ['gsutil', 'cp', gcs_path, str(db_file)], 
        check=True, 
        capture_output=True,
        timeout=60
    )
    
    storage = f"sqlite:///{db_file}"
    study = optuna.load_study(study_name=study_name, storage=storage)
    
    return study.best_params, study.best_trial.user_attrs.get('best_iteration', 1000)


def entrenar_y_predecir_estudio(study_name):
    """Entrena 25 modelos de un estudio y retorna predicciones en 202107."""
    logger.info(f"\n{'='*80}")
    logger.info(f"PROCESANDO: {study_name}")
    logger.info(f"{'='*80}")
    
    conn = None
    try:
        # Cargar config
        config = cargar_config_estudio(study_name)
        bucket_name = config['BUCKET_NAME']
        data_path = config['DATA_PATH_OPT']
        ganancia_acierto = config['GANANCIA_ACIERTO']
        costo_estimulo = config['COSTO_ESTIMULO']
        undersampling_ratio = config.get('UNDERSAMPLING_RATIO', 0.075)
        
        # Features
        features_sel = cargar_features_seleccionadas(study_name, bucket_name, UMBRAL_POR_ESTUDIO[study_name])
        
        # Hiperparámetros
        best_params, best_iteration = cargar_hiperparametros(study_name, bucket_name)
        logger.info(f"{study_name}: best_iteration={best_iteration}")
        
        # Conectar DuckDB
        conn = duckdb.connect(database=':memory:')
        
        # Refrescar credenciales para DuckDB
        token = refrescar_credenciales_gcs()
        
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
        if features_sel:
            cols = features_sel + ['target_binario', 'target_ternario', 'foto_mes']
            cols_str = ', '.join(cols)
            conn.execute(f"CREATE TABLE datos AS SELECT {cols_str} FROM read_parquet('{data_path}')")
            logger.info(f"{study_name}: {len(features_sel)} features seleccionadas")
        else:
            conn.execute(f"CREATE TABLE datos AS SELECT * FROM read_parquet('{data_path}')")
            logger.info(f"{study_name}: Todas las features")
        
        # TRAIN hasta 202105
        query_train = f"""
            WITH clase_0 AS (
                SELECT * FROM datos
                WHERE foto_mes < 202106 AND target_binario = 0
                USING SAMPLE {undersampling_ratio * 100} PERCENT (bernoulli, {SEMILLAS[0]})
            ),
            clase_1 AS (
                SELECT * FROM datos
                WHERE foto_mes < 202106 AND target_binario = 1
            )
            SELECT * FROM clase_0 UNION ALL SELECT * FROM clase_1
        """
        
        train_data = conn.execute(query_train).fetchnumpy()
        feature_cols = [c for c in train_data.keys() if c not in ['target_binario', 'target_ternario', 'foto_mes']]
        
        X_train = np.column_stack([train_data[c] for c in feature_cols])
        y_train = train_data['target_binario']
        
        logger.info(f"{study_name}: Training con {len(y_train):,} registros, {len(feature_cols)} features")
        
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
                'n_jobs': 4,  # Más cores por modelo ya que hay menos estudios en paralelo
                'seed': semilla,
                **best_params
            }
            
            train_set = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
            model = lgb.train(params, train_set, num_boost_round=best_iteration, callbacks=[lgb.log_evaluation(period=0)])
            models.append(model)
            del train_set
            gc.collect()
        
        logger.info(f"{study_name}: ✓ 25 modelos entrenados")
        
        del X_train, y_train, train_data
        gc.collect()
        
        # TEST 202107 - NO cargar y_test acá, viene inyectado
        query_test = f"SELECT * FROM datos WHERE foto_mes = 202107"
        test_data = conn.execute(query_test).fetchnumpy()
        
        X_test = np.column_stack([test_data[c] for c in feature_cols])
        
        logger.info(f"{study_name}: Test con {len(test_data['target_ternario']):,} registros")
        
        # Predecir ensemble (promedio) - SIN MODELOS INDIVIDUALES
        all_probs = [model.predict(X_test) for model in models]
        probs_ensemble = np.mean(all_probs, axis=0)
        
        logger.info(f"{study_name}: ✓ Predicciones ensemble completadas")
        
        conn.close()
        gc.collect()
        
        return {
            'study_name': study_name,
            'predicciones_ensemble': probs_ensemble.tolist(),
            # y_test se inyecta desde main
            'n_features': len(feature_cols)
        }
        
    except Exception as e:
        logger.error(f"Error en {study_name}: {e}", exc_info=True)
        if conn:
            conn.close()
        return None


def encontrar_threshold_optimo(y_true, probs, ganancia_acierto, costo_estimulo):
    """Encuentra threshold óptimo."""
    y_true = np.array(y_true)
    probs = np.array(probs)
    
    sorted_idx = np.argsort(probs)[::-1]
    y_sorted = y_true[sorted_idx]
    
    ganancias = np.where(y_sorted == 1, ganancia_acierto, -costo_estimulo)
    ganancia_acum = np.cumsum(ganancias)
    
    idx_max = np.argmax(ganancia_acum)
    
    return {
        'threshold': float(probs[sorted_idx[idx_max]]),
        'ganancia': float(ganancia_acum[idx_max]),
        'envios': int(idx_max + 1)
    }


def evaluar_estrategias(resultados):
    """Evalúa todas las estrategias posibles - SOLO ENSEMBLES."""
    logger.info("\n" + "="*80)
    logger.info("EVALUANDO ESTRATEGIAS")
    logger.info("="*80)
    
    # Asumir que todos tienen mismo y_test y ganancias
    y_test = np.array(resultados[0]['y_test'])
    ganancia_acierto = resultados[0]['ganancia_acierto']
    costo_estimulo = resultados[0]['costo_estimulo']
    
    estrategias = []
    
    # 1. MEJOR ENSEMBLE POR ESTUDIO
    logger.info("\n1. Evaluando mejor ensemble por estudio...")
    mejor_ensemble = None
    for r in resultados:
        resultado = encontrar_threshold_optimo(
            y_test,
            r['predicciones_ensemble'],
            ganancia_acierto,
            costo_estimulo
        )
        if mejor_ensemble is None or resultado['ganancia'] > mejor_ensemble['ganancia']:
            mejor_ensemble = {
                **resultado,
                'estrategia': 'ENSEMBLE_INDIVIDUAL',
                'estudio': r['study_name']
            }
    
    estrategias.append(mejor_ensemble)
    logger.info(f"   Mejor: {mejor_ensemble['estudio']}")
    logger.info(f"   Ganancia: ${mejor_ensemble['ganancia']:,.0f}")
    
    # 2. SUPER-ENSEMBLE PESOS IGUALES (TODOS)
    logger.info("\n2. Evaluando super-ensemble pesos iguales (7 estudios)...")
    probs_iguales = np.mean([np.array(r['predicciones_ensemble']) for r in resultados], axis=0)
    resultado_iguales = encontrar_threshold_optimo(y_test, probs_iguales, ganancia_acierto, costo_estimulo)
    estrategias.append({
        **resultado_iguales,
        'estrategia': 'SUPER_ENSEMBLE_IGUALES_7',
        'n_estudios': len(resultados),
        'estudios': [r['study_name'] for r in resultados],
        'peso_por_estudio': 1.0 / len(resultados)
    })
    logger.info(f"   Ganancia: ${resultado_iguales['ganancia']:,.0f}")
    
    # 3. SUPER-ENSEMBLE PESOS POR GANANCIA (TODOS)
    logger.info("\n3. Evaluando super-ensemble pesos por ganancia (7 estudios)...")
    ganancias = []
    for r in resultados:
        resultado = encontrar_threshold_optimo(y_test, r['predicciones_ensemble'], ganancia_acierto, costo_estimulo)
        ganancias.append(resultado['ganancia'])
    
    total_ganancia = sum(ganancias)
    pesos_ganancia = [g / total_ganancia for g in ganancias]
    
    probs_ganancia = np.average(
        [np.array(r['predicciones_ensemble']) for r in resultados],
        axis=0,
        weights=pesos_ganancia
    )
    
    resultado_ganancia = encontrar_threshold_optimo(y_test, probs_ganancia, ganancia_acierto, costo_estimulo)
    estrategias.append({
        **resultado_ganancia,
        'estrategia': 'SUPER_ENSEMBLE_GANANCIA_7',
        'n_estudios': len(resultados),
        'estudios': [r['study_name'] for r in resultados],
        'pesos': {r['study_name']: float(p) for r, p in zip(resultados, pesos_ganancia)}
    })
    logger.info(f"   Ganancia: ${resultado_ganancia['ganancia']:,.0f}")
    
    # 4. TODAS LAS COMBINACIONES DE 2, 3, 4, 5 Y 6 ESTUDIOS
    from itertools import combinations
    
    for n_comb in [2, 3, 4, 5, 6]:
        logger.info(f"\n4.{n_comb}. Evaluando todas las combinaciones de {n_comb} estudios...")
        
        n_total_combs = len(list(combinations(range(len(resultados)), n_comb)))
        logger.info(f"   Total de combinaciones: {n_total_combs}")
        
        mejor_comb = None
        comb_count = 0
        
        for indices in combinations(range(len(resultados)), n_comb):
            comb_count += 1
            
            estudios_comb = [resultados[i] for i in indices]
            nombres_comb = [resultados[i]['study_name'] for i in indices]
            
            # Promediar predicciones
            probs_comb = np.mean([np.array(r['predicciones_ensemble']) for r in estudios_comb], axis=0)
            
            resultado_comb = encontrar_threshold_optimo(y_test, probs_comb, ganancia_acierto, costo_estimulo)
            
            if mejor_comb is None or resultado_comb['ganancia'] > mejor_comb['ganancia']:
                mejor_comb = {
                    **resultado_comb,
                    'estrategia': f'COMBO_{n_comb}_ESTUDIOS',
                    'n_estudios': n_comb,
                    'estudios': nombres_comb,
                    'indices': list(indices)
                }
            
            # Log progreso cada 10 combinaciones
            if comb_count % 10 == 0 or comb_count == n_total_combs:
                logger.info(f"     Procesadas {comb_count}/{n_total_combs} combinaciones...")
        
        estrategias.append(mejor_comb)
        logger.info(f"   Mejor combinación: {', '.join(mejor_comb['estudios'])}")
        logger.info(f"   Ganancia: ${mejor_comb['ganancia']:,.0f}")
    
    # 5. TOP-N POR GANANCIA INDIVIDUAL
    logger.info("\n5. Evaluando top-N por ganancia individual...")
    
    # Calcular ganancia individual de cada estudio
    ganancias_estudios = []
    for r in resultados:
        resultado = encontrar_threshold_optimo(y_test, r['predicciones_ensemble'], ganancia_acierto, costo_estimulo)
        ganancias_estudios.append((r, resultado['ganancia']))
    
    ganancias_estudios.sort(key=lambda x: x[1], reverse=True)
    
    for n in [3, 5]:
        if n >= len(resultados):
            continue
        
        top_resultados = [r for r, _ in ganancias_estudios[:n]]
        probs_top = np.mean([np.array(r['predicciones_ensemble']) for r in top_resultados], axis=0)
        
        resultado_top = encontrar_threshold_optimo(y_test, probs_top, ganancia_acierto, costo_estimulo)
        estrategias.append({
            **resultado_top,
            'estrategia': f'TOP_{n}_POR_GANANCIA',
            'estudios': [r['study_name'] for r in top_resultados],
            'n_estudios': n
        })
        logger.info(f"   Top-{n}: {', '.join([r['study_name'] for r in top_resultados])}")
        logger.info(f"   Ganancia: ${resultado_top['ganancia']:,.0f}")
    
    return estrategias


def main():
    logger.info("="*80)
    logger.info("EVALUACIÓN DE ESTRATEGIAS DE ENSEMBLE")
    logger.info("="*80)
    logger.info(f"Estudios: {ESTUDIOS}")
    logger.info(f"Configuración: 96GB RAM + 12 vCPU")
    
    # Cargar y_test UNA SOLA VEZ antes de paralelizar
    logger.info("\nCargando y_test (202107) una vez...")
    primer_estudio = ESTUDIOS[0]
    config = cargar_config_estudio(primer_estudio)
    
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
    
    query = f"SELECT target_ternario FROM read_parquet('{config['DATA_PATH_OPT']}') WHERE foto_mes = 202107"
    y_test_global = conn.execute(query).fetchnumpy()['target_ternario'].tolist()
    ganancia_acierto_global = config['GANANCIA_ACIERTO']
    costo_estimulo_global = config['COSTO_ESTIMULO']
    
    conn.close()
    
    logger.info(f"✓ y_test cargado: {len(y_test_global):,} registros")
    
    # Entrenar y predecir SECUENCIALMENTE (evitar saturar GCS)
    logger.info("\nEntrenando y prediciendo estudios SECUENCIALMENTE...")
    logger.info("(Evita saturar conexiones a GCS)")
    
    resultados = []
    
    for estudio in ESTUDIOS:
        try:
            logger.info(f"\n{'='*80}")
            logger.info(f"Procesando {estudio}...")
            resultado = entrenar_y_predecir_estudio(estudio)
            if resultado:
                # Inyectar y_test global
                resultado['y_test'] = y_test_global
                resultado['ganancia_acierto'] = ganancia_acierto_global
                resultado['costo_estimulo'] = costo_estimulo_global
                resultados.append(resultado)
                logger.info(f"✓ {estudio} completado")
        except Exception as e:
            logger.error(f"Error procesando {estudio}: {e}", exc_info=True)
    
    if len(resultados) == 0:
        logger.error("No se pudo procesar ningún estudio")
        return
    
    logger.info(f"\n✓ {len(resultados)} estudios procesados correctamente")
    
    # Evaluar estrategias
    estrategias = evaluar_estrategias(resultados)
    
    # Ordenar por ganancia
    estrategias_ordenadas = sorted(estrategias, key=lambda x: x['ganancia'], reverse=True)
    
    # Imprimir resultados
    logger.info("\n" + "="*80)
    logger.info("RANKING DE ESTRATEGIAS")
    logger.info("="*80)
    
    for i, est in enumerate(estrategias_ordenadas, 1):
        logger.info(f"\n{i}. {est['estrategia']}")
        logger.info(f"   Ganancia: ${est['ganancia']:,.0f}")
        logger.info(f"   Threshold: {est['threshold']:.6f}")
        logger.info(f"   Envíos: {est['envios']:,}")
        
        if 'estudio' in est:
            logger.info(f"   Estudio: {est['estudio']}")
        if 'pesos' in est:
            logger.info(f"   Pesos:")
            for estudio, peso in sorted(est['pesos'].items(), key=lambda x: x[1], reverse=True):
                logger.info(f"     {estudio}: {peso:.3f}")
        if 'estudios' in est:
            logger.info(f"   Estudios: {', '.join(est['estudios'])}")
    
    # Guardar resultados
    output_path = Path("resultados_estrategias")
    output_path.mkdir(exist_ok=True)
    
    with open(output_path / f"estrategias_{fecha}.json", 'w') as f:
        json.dump({
            'fecha': datetime.now().isoformat(),
            'estrategias': estrategias_ordenadas,
            'mejor_estrategia': estrategias_ordenadas[0]
        }, f, indent=2)
    
    # Subir a GCS
    subprocess.run([
        'gsutil', 'cp', 
        str(output_path / f"estrategias_{fecha}.json"),
        f"gs://sra_electron_bukito3/estrategias_finales/"
    ])
    
    logger.info(f"\n✓ Resultados guardados en {output_path}")
    logger.info("\n" + "="*80)
    logger.info(f"ESTRATEGIA GANADORA: {estrategias_ordenadas[0]['estrategia']}")
    logger.info(f"GANANCIA: ${estrategias_ordenadas[0]['ganancia']:,.0f}")
    logger.info("="*80)


if __name__ == "__main__":
    main()