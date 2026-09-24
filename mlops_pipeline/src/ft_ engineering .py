"""
Feature Engineering para el modelo de riesgo de crédito
Preprocesamiento de datos
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, one_hot_encoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Parametros
TARGET_COLUMN = "Pago_atiempo"
RADOM_STATE = 42
TEST_SIZE = 0.2

#Columnas
NUMERIC_COLUMNS = [
    "edad_cliente",
    "capital_prestado",
    "promedio_ingresos_datacredito",
    "total_otros_prestamos",
    "cuota_pactada",
    "saldo_total",
    "saldo_principal",
    "puntaje",
    "puntaje_datacredito",
    "huella_consulta",
    "plazo_meses",
    "salario_cliente",
    "cant_creditosvigentes",
    "saldo_mora",
    "saldo_mora_codeudor",
    "creditos_sectorFinanciero",
    "creditos_sectorCooperativo",
    "creditos_sectorReal",
]

CATEGORICAL_COLUMNS = [
    "tipo_laboral",
]

CATEGORICAL_ORDINAL_COLUMNS = [
    "tipo_credito",
]
ordinal_categories = [[4,6,7,9,10,68]]

df = pd.read_excel("C:\\Users\\Nico\\Desktop\\Henry\\Data Science\\Modulo 5\\ProyectoM5_NicolásAllende\\Henry-MLOps\\Base_de_datos.xlsx")