"""
Entrenamiento y evaluación de modelos de clasificación para riesgo de crédito.

Compara 7 modelos candidatos bajo el mismo preprocesamiento (ft_engineering.py):
LogisticRegression, RandomForest, GradientBoosting, ExtraTrees, SVC, KNN y XGBoost.

El target está desbalanceado (~95% / 5%), por lo que cada modelo compensa las clases
con el mecanismo que soporta:
    - class_weight='balanced': LogisticRegression, RandomForest, ExtraTrees, SVC.
    - sample_weight balanceado en fit(): GradientBoosting.
    - scale_pos_weight: XGBoost.
    - KNN: sin compensación (no soporta ponderación de clases).

Selecciona el modelo con mejor ROC-AUC en test y lo retorna.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from ft_engineering import (
    load_data,
    split_features_target,
    build_features_pipeline,
    fit_transform_datasets,
    RANDOM_STATE,
)


def get_scale_pos_weight(y_train):
    """Calcula el ratio negativos/positivos para XGBoost."""
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    return n_neg / n_pos if n_pos > 0 else 1.0


def build_models(y_train):
    """
    Construye el diccionario de modelos candidatos con su respectivo
    mecanismo de compensación de clases desbalanceadas.
    """    

    models = {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            class_weight="balanced",
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            class_weight="balanced",
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "SVC": SVC(
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_STATE,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            scale_pos_weight=get_scale_pos_weight(y_train),
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    return models


def fit_model(name, model, X_train, y_train):
    """
    Entrena un modelo. GradientBoosting no soporta class_weight,
    por lo que se le pasa sample_weight balanceado calculado manualmente.
    """
    if name == "GradientBoosting":
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)
    return model


def evaluate_model(name, model, X_test, y_test):
    """Evalúa el modelo con ROC-AUC, classification_report y matriz de confusión."""
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)

    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"\n--- {name} ---")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(classification_report(y_test, y_pred, digits=3))
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

    return roc_auc


def train_and_evaluate_all(X_train, X_test, y_train, y_test):
    """
    Entrena y evalúa todos los modelos candidatos.
    Retorna un dict {nombre: (modelo_entrenado, roc_auc)}.
    """
    models = build_models(y_train)
    results = {}

    for name, model in models.items():
        fitted = fit_model(name, model, X_train, y_train)
        roc_auc = evaluate_model(name, fitted, X_test, y_test)
        results[name] = (fitted, roc_auc)

    return results


def select_best_model(results):
    """
    Selecciona el modelo con mejor ROC-AUC en test.

    Nota de producción: además de la métrica, conviene considerar:
    - Consistencia temporal: validar performance en ventanas de tiempo
      distintas (backtesting), no solo en un split aleatorio.
    - Escalabilidad: modelos como KNN y SVC no escalan bien con grandes
      volúmenes de datos en inferencia (KNN es O(n) por predicción, SVC
      con kernel no lineal es costoso). RandomForest/GradientBoosting/
      XGBoost suelen ser más apropiados para producción por su balance
      entre performance y costo de inferencia.
    """
    best_name = max(results, key=lambda k: results[k][1])
    best_model, best_auc = results[best_name]
    print(f"\n=== Mejor modelo: {best_name} (ROC-AUC={best_auc:.4f}) ===")
    return best_name, best_model, best_auc


def build_full_pipeline(preprocessor_pipeline, model):
    """Combina el preprocesamiento y el modelo en un único Pipeline."""
    return Pipeline(steps=[
        ("preprocessor", preprocessor_pipeline.named_steps.get("preprocessor", preprocessor_pipeline)),
        ("model", model),
    ])


def run_training_evaluation():
    """Flujo completo: carga datos, preprocesa, entrena, evalúa y selecciona el mejor modelo."""
    print("=== Entrenamiento y evaluación de modelos ===")

    data = load_data()
    X_train, X_test, y_train, y_test = split_features_target(data)

    preprocessor_pipeline = build_features_pipeline()
    X_train_t, X_test_t = fit_transform_datasets(preprocessor_pipeline, X_train, X_test)

    results = train_and_evaluate_all(X_train_t, X_test_t, y_train, y_test)
    best_name, best_model, best_auc = select_best_model(results)

    full_pipeline = build_full_pipeline(preprocessor_pipeline, best_model)

    return {
        "best_model_name": best_name,
        "best_model": best_model,
        "best_auc": best_auc,
        "full_pipeline": full_pipeline,
        "all_results": results,
    }


if __name__ == "__main__":
    output = run_training_evaluation()
    print(f"\nModelo final seleccionado: {output['best_model_name']} "
          f"con ROC-AUC={output['best_auc']:.4f}")
    print("\nResultados de todos los modelos:")
    for name, (model, roc_auc) in output['all_results'].items():
        print(f"- {name}: ROC-AUC={roc_auc:.4f}")