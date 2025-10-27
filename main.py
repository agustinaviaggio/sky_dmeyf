import pandas as pd
import os
from datetime import datetime

def main():
    print("Inicio de ejecución")

    os.makedirs("logs", exist_ok=True)

    # Cargar dataset
    try:
        df = pd.read_csv("data/competencia_01_crudo.csv")
    except FileNotFoundError:
        print("No se encontró el archivo data/competencia_01_crudo.csv")
        return
    print(df.head())
    filas, columnas = df.shape
    mensaje = f"[{datetime.now()}] Dataset cargado con {filas} filas y {columnas} columnas\n"

    with open("logs/logs.txt", "a", encoding="utf-8") as f:
        f.write(mensaje)
    
    print(">>> Ejecución finalizada. Revisa logs/logs.txt")

if __name__ == "__main__":
    main()