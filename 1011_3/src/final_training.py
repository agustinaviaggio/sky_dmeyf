import optuna
import lightgbm as lgb
import duckdb
import numpy as np
import logging
import json
import os
from datetime import datetime
from .config import *
from .gain_function import *

def evaluar_en_test(conn, tabla: str, study: optuna.Study, mes_test: str) -> dict:
    """
    Evalúa el modelo con los mejores hiperparámetros en test.
    Args:
        conn: Conexión a DuckDB
        tabla: Nombre de la tabla
        study
    
    Returns:
        dict: Resultados de evaluación en test
    """
    logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")
    logger.info(f"Período de test: {mes_test}")

    mejores_params = study.best_params
    best_iteration = study.best_trial.user_attrs['best_iteration']
    
    # Queries para train completo (train + validación) y test
    if isinstance(MESES_TRAIN, list):
        periodos_train_str = ','.join(map(str, MESES_TRAIN + [MES_VALIDACION]))
    else:
        periodos_train_str = f"{MESES_TRAIN},{MES_VALIDACION}"
    
    query_train_completo = f"SELECT * FROM {tabla} WHERE foto_mes IN ({periodos_train_str})"
    query_test = f"SELECT * FROM {tabla} WHERE foto_mes = {mes_test}"
    
    # Obtener datos con fetchnumpy
    train_data = conn.execute(query_train_completo).fetchnumpy()
    test_data = conn.execute(query_test).fetchnumpy()
    
    # Preparar features y target
    feature_cols = [col for col in train_data.keys() 
                    if col not in ['target_binario', 'target_ternario']]
    
    X_train_completo = np.column_stack([train_data[col] for col in feature_cols])
    y_train_completo = train_data['target_binario']
    
    X_test = np.column_stack([test_data[col] for col in feature_cols])
    y_test = test_data['target_ternario']
    
    logger.info(f"Train completo: {len(y_train_completo)} registros")
    logger.info(f"Test: {len(y_test)} registros")

    models = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    y_pred_futuro = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    
    # Entrenar con mejores parámetros
    for i in range(len(SEMILLAS)):       
        params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'first_metric_only': True,
            'boost_from_average': True,
            'feature_pre_filter': False,
            'max_bin': 31,
            'verbose': 0,
            'seed': SEMILLAS[i],

            'num_leaves': mejores_params['num_leaves'],
            'learning_rate': mejores_params['learning_rate'],
            'min_data_in_leaf': mejores_params['min_data_in_leaf'],
            'feature_fraction': mejores_params['feature_fraction'],
            'bagging_fraction': mejores_params['bagging_fraction'],
            
            'reg_alpha': mejores_params['reg_alpha'],
            'reg_lambda': mejores_params['reg_lambda'],
            'max_depth': mejores_params['max_depth'] 
        }

        train_set = lgb.Dataset(
            X_train_completo,
            label=y_train_completo,
            feature_name=feature_cols
        )

        models[i] = lgb.train(
            params,
            train_set,
            num_boost_round=best_iteration,
            callbacks=[lgb.log_evaluation(period=100)]
        )
    
        # Predecir en test
        y_pred_futuro[i] = models[i].predict(X_test)
        logger.info(f"Entrenamiento con semilla {i} de {len(SEMILLAS)} completadas")
        _, ganancia_test_semilla, _ = ganancia_evaluator(y_pred_futuro[i], lgb.Dataset(X_test, label=y_test))
        logger.info(f"Ganancia en test para esta semilla: {ganancia_test_semilla:,.0f}")
    
    pred_matrix = np.column_stack(y_pred_futuro)
    y_pred_promedio = pred_matrix.mean(axis=1) 
    
    # Calcular ganancia usando tu función que no necesita threshold
    _, ganancia_test, _ = ganancia_evaluator(y_pred_promedio, lgb.Dataset(X_test, label=y_test))
    
    # Estadísticas: contar cuántos estarían en el corte óptimo
    # (opcional, si querés saber cuántos enviarías)
    df_eval = pl.DataFrame({
        'y_true': y_test,
        'y_pred_proba': y_pred_promedio
    })
    df_ordenado = df_eval.sort('y_pred_proba', descending=True)
    df_ordenado = df_ordenado.with_columns([
        pl.when(pl.col('y_true') == 1)
          .then(GANANCIA_ACIERTO)
          .otherwise(-COSTO_ESTIMULO)
          .cast(pl.Int64)
          .alias('ganancia_individual')
    ])
    df_ordenado = df_ordenado.with_columns([
        pl.col('ganancia_individual').cum_sum().alias('ganancia_acumulada')
    ])
    
    # Encontrar el índice donde está la ganancia máxima
    idx_max = df_ordenado.select(
        pl.col('ganancia_acumulada').arg_max()
    ).item()
    
    predicciones_positivas = idx_max + 1  # +1 porque es índice 0-based
    total_predicciones = len(y_test)
    porcentaje_positivas = (predicciones_positivas / total_predicciones) * 100
    
    resultados = {
        'ganancia_test': float(ganancia_test),
        'total_predicciones': int(total_predicciones),
        'predicciones_positivas': int(predicciones_positivas),
        'porcentaje_positivas': float(porcentaje_positivas),
        'parametros_usados': mejores_params
    }
    
    logger.info(f"Ganancia en test: {ganancia_test:,.0f}")
    logger.info(f"Enviarías estímulo a: {predicciones_positivas} clientes ({porcentaje_positivas:.2f}%)")
    
    return resultados
