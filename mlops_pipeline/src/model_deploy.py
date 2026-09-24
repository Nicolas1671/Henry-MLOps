"""
Disponibilización del modelo de riesgo crediticio como API HTTP.

Expone el modelo entrenado por model_training_evaluation.py en un endpoint /predict,
para que cualquier sistema pueda pedir una predicción sin saber Python ni tener el dataset.

El archivo best_model_pipeline.joblib ya trae dentro el preprocesamiento de ft_engineering.py
ajustado durante el entrenamiento, así que aquí no se repite ninguna transformación: se
reutiliza la misma pipeline, que es la única forma de que el modelo reciba los datos
como los aprendió.

Target original: Pago_atiempo (1 = pagó a tiempo, 0 = no pagó a tiempo / default).

Uso:
    uvicorn mlops_pipeline.src.model_deploy:app --reload --port 8000
"""

from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model_pipeline.joblib"


# --------------------------------------------------------------------------- #
# Carga del modelo (una sola vez, al levantar la API)
# --------------------------------------------------------------------------- #
_artifact = None


def get_artifact():
    global _artifact
    if _artifact is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"No se encontró el modelo en {MODEL_PATH}. "
                "Corré primero model_training_evaluation.py."
            )
        _artifact = joblib.load(MODEL_PATH)
    return _artifact


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_artifact()
    yield

app = FastAPI(
    title="API de Riesgo Crediticio",
    description="Predicción de probabilidad de pago a tiempo a partir del modelo entrenado.",
    version="1.0.0",
    lifespan=lifespan,
)

# --------------------------------------------------------------------------- #
# Esquema de entrada: mismas columnas crudas del dataset (sin el target)
# --------------------------------------------------------------------------- #
class CreditoRequest(BaseModel):
    tipo_credito: int
    capital_prestado: float
    plazo_meses: int
    edad_cliente: int
    tipo_laboral: str
    salario_cliente: int
    total_otros_prestamos: int
    cuota_pactada: int
    puntaje_datacredito: Optional[float] = None
    cant_creditosvigentes: int
    huella_consulta: int
    saldo_mora: Optional[float] = None
    saldo_total: Optional[float] = None
    saldo_principal: Optional[float] = None
    saldo_mora_codeudor: Optional[float] = None
    creditos_sectorFinanciero: int
    creditos_sectorCooperativo: int
    creditos_sectorReal: int
    promedio_ingresos_datacredito: Optional[float] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "tipo_credito": 7,
                "capital_prestado": 3692160.0,
                "plazo_meses": 10,
                "edad_cliente": 42,
                "tipo_laboral": "Independiente",
                "salario_cliente": 8000000,
                "total_otros_prestamos": 2500000,
                "cuota_pactada": 341296,
                "puntaje_datacredito": 695.0,
                "cant_creditosvigentes": 10,
                "huella_consulta": 5,
                "saldo_mora": 0.0,
                "saldo_total": 51258.0,
                "saldo_principal": 51258.0,
                "saldo_mora_codeudor": 0.0,
                "creditos_sectorFinanciero": 5,
                "creditos_sectorCooperativo": 0,
                "creditos_sectorReal": 0,
                "promedio_ingresos_datacredito": 908526.0,
            }
        }
    }


class PredictBatchRequest(BaseModel):
    records: List[CreditoRequest] = Field(
        ...,
        description="Lista de créditos a evaluar en una sola llamada.",
    )


class PredictResponse(BaseModel):
    prediction: int = Field(..., description="1 = paga a tiempo, 0 = riesgo de no pago")
    probability_pago_atiempo: float
    model_name: str


class PredictBatchResponse(BaseModel):
    predictions: List[PredictResponse]


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    """Chequeo simple de que la API está viva y el modelo cargado."""
    try:
        artifact = get_artifact()
        return {
            "status": "ok",
            "model_name": artifact["model_name"],
            "roc_auc_train": artifact["roc_auc"],
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/model-info")
def model_info():
    """Metadata del modelo actualmente en producción."""
    artifact = get_artifact()
    return {
        "model_name": artifact["model_name"],
        "roc_auc_test": artifact["roc_auc"],
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: CreditoRequest):
    """
    Recibe un crédito crudo, lo pasa por el pipeline (preprocesamiento + modelo)
    y devuelve la predicción de clase y la probabilidad de pago a tiempo.
    """
    artifact = get_artifact()
    pipeline = artifact["pipeline"]

    try:
        df = pd.DataFrame([request.model_dump()])
        prediction = int(pipeline.predict(df)[0])
        probability = float(pipeline.predict_proba(df)[:, 1][0])
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Error al procesar el registro: {exc}",
        )

    return PredictResponse(
        prediction=prediction,
        probability_pago_atiempo=probability,
        model_name=artifact["model_name"],
    )


@app.post("/predict-batch", response_model=PredictBatchResponse)
def predict_batch(request: PredictBatchRequest):
    """Versión en lote de /predict, para predecir varios créditos en una sola llamada."""
    artifact = get_artifact()
    pipeline = artifact["pipeline"]

    if not request.records:
        raise HTTPException(status_code=400, detail="La lista de registros está vacía.")

    try:
        df = pd.DataFrame([r.model_dump() for r in request.records])
        predictions = pipeline.predict(df)
        probabilities = pipeline.predict_proba(df)[:, 1]
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Error al procesar el lote: {exc}",
        )

    results = [
        PredictResponse(
            prediction=int(pred),
            probability_pago_atiempo=float(prob),
            model_name=artifact["model_name"],
        )
        for pred, prob in zip(predictions, probabilities)
    ]

    return PredictBatchResponse(predictions=results)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("model_deploy:app", host="0.0.0.0", port=8000, reload=True)