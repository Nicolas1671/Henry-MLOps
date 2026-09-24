# 💳 Proyecto MLOps - Riesgo Crediticio

Pipeline end-to-end de Machine Learning para predecir si un solicitante de crédito
**pagará a tiempo** (`Pago_atiempo = 1`) o presenta **riesgo de no pago**
(`Pago_atiempo = 0`), incluyendo entrenamiento, disponibilización como API,
interfaz de usuario y monitoreo de data drift en producción.

---

## 📂 Estructura del proyecto

```
Henry-MLOps/
├── Base_de_datos.xlsx              # Dataset histórico de entrenamiento
├── Base_de_datos_drifted.xlsx      # Dataset simulado con drift (generado)
└── mlops_pipeline/
    ├── src/
    │   ├── ft_engineering.py           # Carga de datos + pipeline de preprocesamiento
    │   ├── model_training_evaluation.py# Entrenamiento, comparación y selección del mejor modelo
    │   ├── model_deploy.py             # API FastAPI que expone el modelo (/predict, /predict-batch)
    │   ├── app_predict_credito.py      # UI Streamlit para consumir la API
    │   ├── model_monitoring.py         # Cálculo de data drift (KS, PSI, JS, Chi²)
    │   ├── generate_drifted_data.py    # Generador de dataset con drift artificial (testing)
    │   └── app_monitoreo_drift.py      # UI Streamlit para visualizar reportes de drift
    ├── models/
    │   └── best_model_pipeline.joblib  # Pipeline serializado (preprocesador + mejor modelo)
    └── reports/
        ├── drift_report.json           # Última corrida de monitoreo (se sobrescribe)
        └── drift_history.csv           # Historial acumulado de corridas
```

---

## 🔁 Flujo de trabajo (pipeline completo)

```
Base_de_datos.xlsx
        │
        ▼
ft_engineering.py ──────────► preprocesamiento (imputación + encoding)
        │
        ▼
model_training_evaluation.py ─► entrena 7 modelos, elige el mejor por ROC-AUC
        │                       y lo guarda en best_model_pipeline.joblib
        ▼
model_deploy.py (FastAPI) ────► expone /predict y /predict-batch
        │
        ▼
app_predict_credito.py (Streamlit) ─► UI para cargar créditos y ver predicciones
        │
        ▼
model_monitoring.py ──────────► compara distribución histórica vs. lotes recientes
        │
        ▼
app_monitoreo_drift.py (Streamlit) ─► dashboard de drift + recomendaciones
```

---

## 🧬 Variables del modelo

El target es **`Pago_atiempo`** (1 = paga a tiempo, 0 = riesgo de no pago). El dataset
está desbalanceado (~95% / 5%).

### Variables numéricas (`NUMERIC_COLUMNS`)
Imputadas con la **mediana**:

`edad_cliente`, `capital_prestado`, `promedio_ingresos_datacredito`,
`total_otros_prestamos`, `cuota_pactada`, `saldo_total`, `saldo_principal`,
`puntaje_datacredito`, `huella_consulta`, `plazo_meses`,
`salario_cliente`, `cant_creditosvigentes`, `saldo_mora`, `saldo_mora_codeudor`,
`creditos_sectorFinanciero`, `creditos_sectorCooperativo`, `creditos_sectorReal`

### Variable categórica nominal (`CATEGORICAL_COLUMNS`)
Imputada con la **moda** + **One-Hot Encoding**:

- `tipo_laboral`

### Variable categórica ordinal (`CATEGORICAL_ORDINAL_COLUMNS`)
Imputada con la **moda** + **Ordinal Encoding** con categorías fijas:

- `tipo_credito` → orden `[4, 6, 7, 9, 10, 68]` (valores no vistos se mapean a `-1`)

> ⚠️ **Nota:** `fecha_prestamo` y `tendencia_ingresos` existen en el dataset original
> pero **no se usan** como input del modelo (la segunda está deshabilitada
> explícitamente en `ft_engineering.py`). No es necesario enviarlas a la API,
> aunque no rompe nada si se incluyen (se ignoran).

---

## 🔍 Decisiones de calidad de datos (EDA → Feature Engineering)

Durante el análisis exploratorio (`comprension_eda.ipynb`) se detectaron 3 problemas
que impactan directamente en la validez del modelo. Se documentan acá las decisiones
tomadas y su justificación:

### 1. Exclusión de `puntaje` por data leakage

En la sección 5 del EDA (`comprension_eda.ipynb`), `puntaje` resultó ser la variable
con **mayor correlación absoluta** con el target `Pago_atiempo`. Un análisis posterior
mostró indicios de que este score interno se calcula usando información **posterior**
al desembolso del crédito (comportamiento de pago ya observado), lo cual constituye
**data leakage**: el modelo "vería el futuro" al entrenar, generando una métrica de
ROC-AUC artificialmente alta que no se replicaría en producción con datos reales.

**Decisión:** se descarta `puntaje` como feature. Se mantiene `puntaje_datacredito`,
que es un score de buró **externo**, disponible legítimamente al momento de originar
el crédito.

### 2. Corrección de outliers en `edad_cliente`

El EDA detectó valores imposibles (mayores a 90 años), producto de
errores de captura. En vez de eliminar esos registros (lo que perdería información
válida del resto de las columnas), se aplicó un **recorte (`clip`)** de la variable al
rango `[18, 90]` dentro de `load_data()` en `ft_engineering.py`. Esto garantiza que el
mismo tratamiento se aplique tanto en entrenamiento como en cualquier dato nuevo que
reciba la API en producción.

### 3. Exclusión de `tendencia_ingresos`

Esta columna categórica presentó valores inconsistentes (números, texto libre y
categorías sin sentido de negocio), producto de errores de carga/captura. Dado el
costo de limpieza vs. el beneficio incierto, se decidió **excluirla del modelo** por
ahora. Queda documentada como mejora futura si se logra sanear la fuente de datos.

> 📌 Estas decisiones impactan únicamente en `ft_engineering.py`. Gracias a que el
> pipeline completo (preprocesamiento + modelo) se serializa como una sola unidad en
> `best_model_pipeline.joblib`, **no fue necesario modificar** la API, la UI de
> Streamlit ni el Dockerfile: todos consumen el pipeline como caja negra.

---

## 🤖 Modelos evaluados

`model_training_evaluation.py` entrena y compara 7 modelos bajo el mismo
preprocesamiento, seleccionando el de mejor **ROC-AUC** en test:

| Modelo | Manejo de clases desbalanceadas |
|---|---|
| Logistic Regression | `class_weight='balanced'` |
| Random Forest | `class_weight='balanced'` |
| Extra Trees | `class_weight='balanced'` |
| SVC | `class_weight='balanced'` |
| Gradient Boosting | `sample_weight` balanceado en `fit()` |
| XGBoost | `scale_pos_weight` |
| KNN | sin compensación (no soporta ponderación) |

El pipeline ganador (preprocesador + modelo) se serializa en:
`mlops_pipeline/models/best_model_pipeline.joblib`

---

## 🚀 Cómo correr el proyecto

### 1. Instalar dependencias

```powershell
pip install pandas numpy scikit-learn xgboost joblib fastapi uvicorn streamlit requests plotly openpyxl scipy
```

### 2. Entrenar el modelo

```powershell
python mlops_pipeline/src/model_training_evaluation.py
```

Esto genera `mlops_pipeline/models/best_model_pipeline.joblib` con el mejor modelo
y su ROC-AUC.

### 3. Levantar la API

```powershell
uvicorn mlops_pipeline.src.model_deploy:app --reload --port 8000
```

Documentación interactiva: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

**Endpoints:**

| Endpoint | Método | Descripción |
|---|---|---|
| `/health` | GET | Estado de la API y metadata del modelo cargado |
| `/model-info` | GET | Nombre del modelo, ROC-AUC y path del `.joblib` |
| `/predict` | POST | Predicción para un crédito individual |
| `/predict-batch` | POST | Predicción para una lista de créditos |

### 4. Levantar la UI de predicción

```powershell
streamlit run mlops_pipeline/src/app_predict_credito.py
```

Permite:
- Cargar un crédito manualmente vía formulario y ver la predicción.
- Subir un Excel/CSV con varios créditos y descargar los resultados.

### 5. Generar datos con drift (para testing del monitoreo)

```powershell
python mlops_pipeline/src/generate_drifted_data.py
```

Genera `Base_de_datos_drifted.xlsx`, aplicando corrimientos artificiales en
variables numéricas y cambios de proporción en variables categóricas.

### 6. Correr el monitoreo de drift

```powershell
python mlops_pipeline/src/model_monitoring.py --reference Base_de_datos.xlsx --current Base_de_datos_drifted.xlsx
```

Genera:
- `mlops_pipeline/reports/drift_report.json` (foto de la última corrida)
- `mlops_pipeline/reports/drift_history.csv` (historial acumulado)

**Métricas utilizadas:**

| Métrica | Aplica a | Umbral de alerta |
|---|---|---|
| Kolmogorov-Smirnov (KS) | Numéricas | p-value < 0.05 |
| Population Stability Index (PSI) | Numéricas y categóricas | PSI ≥ 0.25 (moderado ≥ 0.10) |
| Jensen-Shannon Divergence | Numéricas y categóricas | JS > 0.1 |
| Chi-cuadrado | Categóricas | p-value < 0.05 |

Una variable se marca con `drift_detected=True` si **al menos 2 métricas**
coinciden (regla de mayoría, evita falsos positivos de una sola métrica).

### 7. Levantar el dashboard de drift

```powershell
streamlit run mlops_pipeline/src/app_monitoreo_drift.py
```

Incluye 3 pestañas: métricas de la última corrida, evolución histórica y
recomendaciones automáticas de acción.

---

## 📊 Resultados esperados

Al finalizar el flujo completo vas a tener:

- ✅ Un modelo entrenado y versionado (`.joblib`) con su ROC-AUC documentado.
- ✅ Una API productiva para servir predicciones en tiempo real o en lote.
- ✅ Una interfaz visual para que usuarios no técnicos consulten el modelo.
- ✅ Un sistema de monitoreo que alerta automáticamente si el modelo necesita
  reentrenarse por cambios en la distribución de los datos.

---

## 🛠️ Stack tecnológico

- **Procesamiento y modelado:** `pandas`, `numpy`, `scikit-learn`, `xgboost`
- **Serialización:** `joblib`
- **API:** `FastAPI`, `uvicorn`, `pydantic`
- **UI:** `Streamlit`, `plotly`
- **Monitoreo estadístico:** `scipy` (KS test, Chi², Jensen-Shannon)