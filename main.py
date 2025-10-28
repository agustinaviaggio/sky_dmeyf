import pandas as pd
import os
from datetime import datetime
import logging
from src.loader import cargar_datos
from src.features import feature_engineering_lag

os.makedirs("logs", exist_ok=True)

fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
monbre_log = f"log_{fecha}.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/{monbre_log}", mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    logger.info("Inicio de ejecucion.")

    #00 Cargar datos
    os.makedirs("data", exist_ok=True)
    path = "data/competencia_01_crudo.csv"
    df = cargar_datos(path)   

    #01 Feature Engineering
    atributos = ["ctrx_quarter"]
    cant_lag = 2
    df = feature_engineering_lag(df, columnas=atributos, cant_lag=cant_lag)
  
    #02 Guardar datos
    path = "data/competencia_01_lag.csv"
    df.to_csv(path, index=False)
  
    logger.info(f">>> Ejecución finalizada. Revisar logs para mas detalles.{monbre_log}")

if __name__ == "__main__":
    main()