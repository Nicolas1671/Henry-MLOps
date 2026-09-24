"""
Feature Engineering para el modelo de riesgo de crédito
Preprocesamiento de datos
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Parametros
TARGET_COLUMN = "Pago_atiempo"
RANDOM_STATE = 42
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

# NOTA: se descarta "puntaje" (score interno) por alta correlación con el target,
# confirmado en comprension_eda.ipynb (sección 5). Es muy probable que se calcule
# usando información posterior al desembolso (ej. comportamiento de pago), lo cual
# constituye data leakage: el modelo "vería" el futuro al entrenar.
# "puntaje_datacredito" SÍ se mantiene: es un score de buró externo, disponible
# al momento de originar el crédito.

CATEGORICAL_COLUMNS = [
    "tipo_laboral", 
    #"tendencia_ingresos",
]

CATEGORICAL_ORDINAL_COLUMNS = [
    "tipo_credito",
]
ordinal_categories = [[4,6,7,9,10,68]]

# Límites de negocio para edad_cliente (evita valores imposibles: negativos, >100, etc.)
EDAD_MIN = 18
EDAD_MAX = 90

#Cargamos los datos
def load_data():    
    data = pd.read_excel(Path(__file__).parent.parent.parent / 'Base_de_datos.xlsx')    
    # Corregimos outliers/errores de captura en edad_cliente con un recorte (clip)
    # en vez de eliminar filas: mantiene el volumen de datos y evita perder señal
    # de otras variables del mismo registro.
    data["edad_cliente"] = data["edad_cliente"].clip(lower=EDAD_MIN, upper=EDAD_MAX)
    return data

def build_features_pipeline(
        NUMERIC_COLUMNS = NUMERIC_COLUMNS, 
        CATEGORICAL_COLUMNS = CATEGORICAL_COLUMNS,
        CATEGORICAL_ORDINAL_COLUMNS = CATEGORICAL_ORDINAL_COLUMNS,
        ordinal_categories = ordinal_categories,
    ):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    categorical_ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(categories=ordinal_categories, handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_COLUMNS),
            ('cat', categorical_transformer, CATEGORICAL_COLUMNS),
            ('cat_ordinal', categorical_ordinal_transformer, CATEGORICAL_ORDINAL_COLUMNS)
        ]
    )
    return Pipeline(steps=[('preprocessor', preprocessor)])

def split_features_target(
        data = None, 
        target_column=TARGET_COLUMN, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
    ):
    if data is None:
        data = load_data()
    X = data.drop(columns=[target_column])
    y = data[target_column]

    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def fit_transform_datasets(
        pipeline, 
        X_train, 
        X_test
):
    X_train_transformed = pipeline.fit_transform(X_train)
    X_test_transformed = pipeline.transform(X_test)
    return X_train_transformed, X_test_transformed

if __name__ == "__main__":
    print("=== Feature engineering ===")
    data = load_data()
    X_train, X_test, y_train, y_test = split_features_target(data)
    pipeline = build_features_pipeline()
    X_train_transformed, X_test_transformed = fit_transform_datasets(pipeline, X_train, X_test)
    print("Feature engineering pipeline applied successfully.")
    
    print(f"X_train_transformed: {X_train_transformed.shape}")
    print(f"X_test_transformed:  {X_test_transformed.shape}")
    print(f"y_train: {y_train.shape} | y_test: {y_test.shape}")