"""
Genera un Excel con drift artificial a partir de la base histórica,
para poder probar el pipeline de model_monitoring.py.

"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(36)

INPUT_PATH = "Base_de_datos.xlsx"
OUTPUT_PATH = "Base_de_datos_drifted.xlsx"

df = pd.read_excel(INPUT_PATH)
print("Columnas detectadas:", list(df.columns))
print(df.dtypes)

drifted = df.copy()

numeric_cols = drifted.select_dtypes(include=np.number).columns.tolist()
categorical_cols = drifted.select_dtypes(include=["object", "category"]).columns.tolist()

# --- Elegimos un subconjunto de columnas para introducir drift --- #
n_numeric_drift = min(2, len(numeric_cols))
n_categorical_drift = min(1, len(categorical_cols))

numeric_drift_cols = np.random.choice(numeric_cols, n_numeric_drift, replace=False) if numeric_cols else []
categorical_drift_cols = np.random.choice(categorical_cols, n_categorical_drift, replace=False) if categorical_cols else []

print("\nVariables con drift inyectado:")
print("Numéricas:", list(numeric_drift_cols))
print("Categóricas:", list(categorical_drift_cols))

# --- Drift en numéricas: shift de media + aumento de varianza --- #
for col in numeric_drift_cols:
    std = drifted[col].std()
    shift = std * 0.8  # corrimiento de ~0.8 desvíos estándar
    noise = np.random.normal(loc=shift, scale=std * 0.3, size=len(drifted))
    drifted[col] = drifted[col] + noise

# --- Drift en categóricas: cambiamos las proporciones --- #
for col in categorical_drift_cols:
    categories = drifted[col].dropna().unique()
    if len(categories) > 1:
        # Sobre-representamos una categoría en el 30% de las filas
        target_category = categories[0]
        mask = np.random.rand(len(drifted)) < 0.3
        drifted.loc[mask, col] = target_category

drifted.to_excel(OUTPUT_PATH, index=False)
print(f"\nArchivo generado: {OUTPUT_PATH}")