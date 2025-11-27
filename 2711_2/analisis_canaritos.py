"""
Script de análisis de feature importance con canaritos.
Analiza los top 5 trials de Optuna y evalúa la posición de features artificiales (canaritos).
"""

import duckdb
import optuna
import lightgbm as lgb
import numpy as np
import pandas as pd
import logging
import json
import subprocess
import os
from datetime import datetime
from collections import Counter

# Configuración
BUCKET_NAME = "gs://sra_electron_bukito3/"
STUDY_NAME = "2511_2"
SQL_TABLE_NAME = "dataset_competencia"
DATA_PATH = "gs://sra_electron_bukito3/datasets/competencia_03_FE_v4.parquet"
PERIODOS_TRAIN = list(range(202101, 202107))  # Hasta 202106 inclusive
N_CANARITOS = 10
UNDERSAMPLING_RATIO = 0.1
SEMILLA_BASE = 102191

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def configurar_gcs(conn):
    """Configura autenticación para GCS."""
    from google.auth import default
    from google.auth.transport.requests import Request
    
    credentials, _ = default()
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
    logger.info("✓ GCS configurado")

def cargar_dataset_con_canaritos(conn):
    """Carga el dataset y agrega columnas de canaritos."""
    logger.info(f"Cargando dataset desde {DATA_PATH}...")
    
    # Cargar dataset original
    conn.execute(f"""
        CREATE TABLE {SQL_TABLE_NAME} AS 
        SELECT * FROM read_parquet('{DATA_PATH}')
    """)
    
    n_rows = conn.execute(f"SELECT COUNT(*) FROM {SQL_TABLE_NAME}").fetchone()[0]
    logger.info(f"✓ Dataset cargado: {n_rows:,} filas")
    
    # Agregar canaritos
    logger.info(f"Agregando {N_CANARITOS} canaritos...")
    np.random.seed(SEMILLA_BASE)
    
    for i in range(1, N_CANARITOS + 1):
        canarito_col = f"canarito_{i}"
        valores = np.random.uniform(-1, 1, n_rows)
        
        # Crear tabla temporal con los valores
        df_temp = pd.DataFrame({
            'row_num': range(n_rows),
            canarito_col: valores
        })
        
        conn.register('temp_canarito', df_temp)
        
        # Agregar columna y actualizar
        conn.execute(f"""
            CREATE TEMP TABLE numbered_rows AS
            SELECT ROW_NUMBER() OVER () - 1 as row_num, *
            FROM {SQL_TABLE_NAME}
        """)
        
        conn.execute(f"DROP TABLE {SQL_TABLE_NAME}")
        
        conn.execute(f"""
            CREATE TABLE {SQL_TABLE_NAME} AS
            SELECT n.* EXCLUDE(row_num), t.{canarito_col}
            FROM numbered_rows n
            LEFT JOIN temp_canarito t ON n.row_num = t.row_num
        """)
        
        conn.execute("DROP TABLE numbered_rows")
        conn.unregister('temp_canarito')
        
        logger.info(f"  ✓ {canarito_col} agregado")
    
    logger.info(f"✓ {N_CANARITOS} canaritos agregados exitosamente")

def descargar_db_optuna():
    """Descarga la base de datos de Optuna desde GCS."""
    local_db_dir = os.path.expanduser("~/optuna_db")
    os.makedirs(local_db_dir, exist_ok=True)
    
    db_file = os.path.join(local_db_dir, f"{STUDY_NAME}.db")
    gcs_path = f"{BUCKET_NAME}optuna_db/{STUDY_NAME}.db"
    
    logger.info(f"Descargando DB de Optuna desde {gcs_path}...")
    
    result = subprocess.run(
        ['gsutil', 'cp', gcs_path, db_file],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        logger.info(f"✓ DB descargada: {db_file}")
        return db_file
    else:
        raise Exception(f"Error descargando DB: {result.stderr}")

def cargar_estudio_optuna(db_file):
    """Carga el estudio de Optuna."""
    storage = f"sqlite:///{db_file}"
    study = optuna.load_study(study_name=STUDY_NAME, storage=storage)
    logger.info(f"✓ Estudio cargado: {len(study.trials)} trials")
    return study

def entrenar_modelo_con_params(conn, params, semilla):
    """Entrena un modelo con parámetros específicos."""
    logger.info(f"  Entrenando modelo con semilla {semilla}...")
    
    # Query de entrenamiento
    periodos_str = ','.join(map(str, PERIODOS_TRAIN))
    
    query_train = f"""
        WITH clase_0_sample AS (
            SELECT * FROM {SQL_TABLE_NAME}
            WHERE foto_mes IN ({periodos_str}) 
              AND target_binario = 0
            USING SAMPLE {UNDERSAMPLING_RATIO * 100} PERCENT (bernoulli, {semilla})
        ),
        clase_1_completa AS (
            SELECT * FROM {SQL_TABLE_NAME}
            WHERE foto_mes IN ({periodos_str}) 
              AND target_binario = 1
        )
        SELECT * FROM clase_0_sample
        UNION ALL
        SELECT * FROM clase_1_completa
    """
    
    train_data = conn.execute(query_train).fetchnumpy()
    
    # Preparar features
    feature_cols = [col for col in train_data.keys() 
                   if col not in ['target_binario', 'target_ternario', 'foto_mes']]
    
    X_train = np.column_stack([train_data[col] for col in feature_cols])
    y_train = train_data['target_binario']
    
    # Parámetros del modelo
    train_params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'max_bin': 31,
        'is_unbalance': True,
        'boost_from_average': True,
        'feature_pre_filter': True,
        'bagging_freq': 1,
        'n_jobs': -1,
        'seed': semilla,
        'verbose': -1,
        **params
    }
    
    # Entrenar
    train_set = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    
    model = lgb.train(
        train_params,
        train_set,
        num_boost_round=params['best_iteration'],
        callbacks=[lgb.log_evaluation(period=0)]
    )
    
    # Feature importance
    importance = model.feature_importance(importance_type='gain')
    feature_names = model.feature_name()
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    logger.info(f"  ✓ Modelo entrenado. Features: {len(feature_names)}")
    
    del X_train, y_train, train_data, train_set, model
    
    return importance_df

def analizar_canaritos(importance_df):
    """Analiza posiciones de canaritos en el ranking."""
    canaritos = importance_df[importance_df['feature'].str.startswith('canarito_')].copy()
    
    if len(canaritos) == 0:
        return None
    
    primer_canarito_idx = canaritos.index[0]
    features_hasta_primer = int(primer_canarito_idx)
    
    if len(canaritos) >= 10:
        decimo_canarito_idx = canaritos.index[9]
        features_hasta_decimo = int(decimo_canarito_idx)
    else:
        decimo_canarito_idx = None
        features_hasta_decimo = None
    
    return {
        'primer_canarito': canaritos.iloc[0]['feature'],
        'primer_canarito_pos': int(primer_canarito_idx),
        'primer_canarito_importance': float(canaritos.iloc[0]['importance']),
        'features_hasta_primer_canarito': features_hasta_primer,
        'decimo_canarito': canaritos.iloc[9]['feature'] if len(canaritos) >= 10 else None,
        'decimo_canarito_pos': int(decimo_canarito_idx) if decimo_canarito_idx is not None else None,
        'decimo_canarito_importance': float(canaritos.iloc[9]['importance']) if len(canaritos) >= 10 else None,
        'features_hasta_decimo_canarito': features_hasta_decimo,
        'total_canaritos': len(canaritos),
        'todos_canaritos': canaritos[['feature', 'importance']].to_dict('records')
    }

def analizar_overlap_features(lista_features_info):
    """Analiza overlap entre features de diferentes modelos."""
    # Extraer solo las listas de features
    listas_features = [info['features'] for info in lista_features_info]
    
    sets_features = [set(features) for features in listas_features]
    
    # Features comunes a TODOS
    features_comunes = set.intersection(*sets_features)
    
    # Features únicas por modelo
    features_unicas = []
    for i, features_set in enumerate(sets_features):
        otras = set.union(*[s for j, s in enumerate(sets_features) if j != i])
        unicas = features_set - otras
        features_unicas.append(list(unicas))
    
    # Frecuencia de aparición
    todas_features = [f for features in listas_features for f in features]
    contador = Counter(todas_features)
    
    # Análisis por frecuencia
    features_en_5 = [f for f, c in contador.items() if c == 5]
    features_en_4 = [f for f, c in contador.items() if c == 4]
    features_en_3 = [f for f, c in contador.items() if c == 3]
    
    return {
        'features_comunes_todos_5': list(features_comunes),
        'n_features_comunes_5': len(features_comunes),
        'features_unicas_por_modelo': features_unicas,
        'n_features_unicas_por_modelo': [len(u) for u in features_unicas],
        'features_en_5_modelos': features_en_5,
        'features_en_4_modelos': features_en_4,
        'features_en_3_modelos': features_en_3,
        'n_features_en_5': len(features_en_5),
        'n_features_en_4': len(features_en_4),
        'n_features_en_3': len(features_en_3),
        'distribucion_completa': {f: c for f, c in contador.most_common(50)}
    }

def guardar_resultados_gcs(resultados, archivo_local):
    """Guarda resultados localmente y sube a GCS."""
    # Guardar local
    os.makedirs("resultados", exist_ok=True)
    ruta_local = f"resultados/{archivo_local}"
    
    with open(ruta_local, 'w') as f:
        json.dump(resultados, f, indent=2)
    
    logger.info(f"✓ Resultados guardados localmente: {ruta_local}")
    
    # Subir a GCS
    gcs_path = f"{BUCKET_NAME}resultados/{archivo_local}"
    
    result = subprocess.run(
        ['gsutil', 'cp', ruta_local, gcs_path],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        logger.info(f"✓ Resultados subidos a GCS: {gcs_path}")
    else:
        logger.warning(f"Error subiendo a GCS: {result.stderr}")

def main():
    logger.info("="*70)
    logger.info("ANÁLISIS DE FEATURE IMPORTANCE CON CANARITOS")
    logger.info("="*70)
    
    conn = None
    
    try:
        # 1. Configurar DuckDB y GCS
        conn = duckdb.connect(database=':memory:')
        configurar_gcs(conn)
        
        # 2. Cargar dataset con canaritos
        cargar_dataset_con_canaritos(conn)
        
        # 3. Cargar estudio de Optuna
        db_file = descargar_db_optuna()
        study = cargar_estudio_optuna(db_file)
        
        # 4. Obtener top 5 trials
        trials_completos = [t for t in study.trials if t.value is not None]
        top_5_trials = sorted(trials_completos, key=lambda t: t.value, reverse=True)[:5]
        
        logger.info("\n" + "="*70)
        logger.info("TOP 5 TRIALS")
        logger.info("="*70)
        for i, trial in enumerate(top_5_trials, 1):
            logger.info(f"{i}. Trial {trial.number}: Ganancia = {trial.value:,.0f}")
        
        # 5. Analizar cada trial
        resultados_trials = []
        features_hasta_primer_canarito = []
        features_hasta_decimo_canarito = []
        
        for i, trial in enumerate(top_5_trials, 1):
            logger.info("\n" + "-"*70)
            logger.info(f"ANALIZANDO TRIAL {trial.number} ({i}/5)")
            logger.info("-"*70)
            
            params = trial.params.copy()
            params['best_iteration'] = trial.user_attrs.get('best_iteration', 100)
            
            # Entrenar modelo
            importance_df = entrenar_modelo_con_params(conn, params, SEMILLA_BASE)
            
            # Analizar canaritos
            analisis = analizar_canaritos(importance_df)
            
            # Extraer features hasta primer canarito
            n_hasta_primer = analisis['features_hasta_primer_canarito']
            features_primer = importance_df.iloc[:n_hasta_primer]['feature'].tolist()
            features_hasta_primer_canarito.append({
                'trial': trial.number,
                'features': features_primer
            })
            
            # Extraer features hasta décimo canarito
            if analisis['features_hasta_decimo_canarito']:
                n_hasta_decimo = analisis['features_hasta_decimo_canarito']
                features_decimo = importance_df.iloc[:n_hasta_decimo]['feature'].tolist()
                features_hasta_decimo_canarito.append({
                    'trial': trial.number,
                    'features': features_decimo
                })
            
            # Guardar resultado
            resultado_trial = {
                'trial_number': trial.number,
                'ganancia': float(trial.value),
                'params': params,
                'analisis_canaritos': analisis,
                'top_20_features': importance_df.head(20)[['feature', 'importance']].to_dict('records'),
                'ranking_completo': importance_df[['feature', 'importance']].to_dict('records')
            }
            
            resultados_trials.append(resultado_trial)
            
            logger.info(f"  Primer canarito: {analisis['primer_canarito']} en posición {analisis['primer_canarito_pos']}")
            logger.info(f"  Features hasta 1er canarito: {n_hasta_primer}")
            if analisis['decimo_canarito']:
                logger.info(f"  Décimo canarito: {analisis['decimo_canarito']} en posición {analisis['decimo_canarito_pos']}")
                logger.info(f"  Features hasta 10mo canarito: {analisis['features_hasta_decimo_canarito']}")
        
        # 6. Análisis de overlap
        logger.info("\n" + "="*70)
        logger.info("ANÁLISIS DE OVERLAP - FEATURES HASTA PRIMER CANARITO")
        logger.info("="*70)
        
        overlap_primer = analizar_overlap_features(features_hasta_primer_canarito)
        
        logger.info(f"Features comunes a los 5 modelos: {overlap_primer['n_features_comunes_5']}")
        logger.info(f"Features en 4 modelos: {overlap_primer['n_features_en_4']}")
        logger.info(f"Features en 3 modelos: {overlap_primer['n_features_en_3']}")
        logger.info(f"Features únicas por modelo: {overlap_primer['n_features_unicas_por_modelo']}")
        
        overlap_decimo = None
        if features_hasta_decimo_canarito:
            logger.info("\n" + "="*70)
            logger.info("ANÁLISIS DE OVERLAP - FEATURES HASTA DÉCIMO CANARITO")
            logger.info("="*70)
            
            overlap_decimo = analizar_overlap_features(features_hasta_decimo_canarito)
            
            logger.info(f"Features comunes a los 5 modelos: {overlap_decimo['n_features_comunes_5']}")
            logger.info(f"Features en 4 modelos: {overlap_decimo['n_features_en_4']}")
            logger.info(f"Features únicas por modelo: {overlap_decimo['n_features_unicas_por_modelo']}")
        
        # 7. Compilar resultados finales
        resultado_final = {
            'metadata': {
                'fecha_analisis': datetime.now().isoformat(),
                'study_name': STUDY_NAME,
                'n_canaritos': N_CANARITOS,
                'periodos_entrenamiento': PERIODOS_TRAIN,
                'undersampling_ratio': UNDERSAMPLING_RATIO,
                'semilla_base': SEMILLA_BASE
            },
            'top_5_trials': resultados_trials,
            'overlap_hasta_primer_canarito': overlap_primer,
            'overlap_hasta_decimo_canarito': overlap_decimo,
            'features_por_trial': {
                'hasta_primer_canarito': features_hasta_primer_canarito,
                'hasta_decimo_canarito': features_hasta_decimo_canarito
            }
        }
        
        # 8. Guardar resultados
        archivo = f"analisis_canaritos_{STUDY_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        guardar_resultados_gcs(resultado_final, archivo)
        
        # 9. Resumen final
        logger.info("\n" + "="*70)
        logger.info("RESUMEN FINAL")
        logger.info("="*70)
        
        for res in resultados_trials:
            logger.info(f"\nTrial {res['trial_number']} (Ganancia: {res['ganancia']:,.0f}):")
            logger.info(f"  Features hasta 1er canarito: {res['analisis_canaritos']['features_hasta_primer_canarito']}")
            if res['analisis_canaritos']['features_hasta_decimo_canarito']:
                logger.info(f"  Features hasta 10mo canarito: {res['analisis_canaritos']['features_hasta_decimo_canarito']}")
        
        logger.info(f"\n{'='*70}")
        logger.info("ANÁLISIS COMPLETADO EXITOSAMENTE")
        logger.info(f"{'='*70}")
        
    except Exception as e:
        logger.error(f"Error durante el análisis: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    finally:
        if conn:
            conn.close()
            logger.info("Conexión a DuckDB cerrada")

if __name__ == "__main__":
    main()