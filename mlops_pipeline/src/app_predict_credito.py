"""
App de Streamlit para consumir la API de riesgo crediticio (model_deploy.py).

Permite:
    - Evaluar un crédito individual con un formulario.
    - Evaluar un lote de créditos subiendo un Excel/CSV.

Requiere que la API esté corriendo:
    uvicorn mlops_pipeline.src.model_deploy:app --reload --port 8000

Uso:
    streamlit run mlops_pipeline/src/app_predict_credito.py
"""

import json
from datetime import date

import pandas as pd
import requests
import streamlit as st

API_URL = st.sidebar.text_input("URL de la API", value="http://127.0.0.1:8000")

st.set_page_config(page_title="Predicción de Riesgo Crediticio", layout="wide")
st.title("💳 Predicción de Riesgo Crediticio")

TIPOS_LABORALES = ["Independiente", "Empleado", "Pensionado", "Otro"]
TENDENCIAS_INGRESOS = ["Estable", "Creciente", "Decreciente"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def check_api_health():
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


def call_predict(payload):
    resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
    if resp.status_code != 200:
        raise RuntimeError(resp.json().get("detail", resp.text))
    return resp.json()


def call_predict_batch(records):
    resp = requests.post(f"{API_URL}/predict-batch", json={"records": records}, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(resp.json().get("detail", resp.text))
    return resp.json()


# --------------------------------------------------------------------------- #
# Sidebar: estado de la API
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.subheader("Estado de la API")
    health = check_api_health()
    if health.get("status") == "ok":
        st.success("✅ API conectada")
        st.caption(f"Modelo: **{health['model_name']}**")
        st.caption(f"ROC-AUC (train): **{health['roc_auc_train']:.4f}**")
    else:
        st.error("❌ No se pudo conectar con la API")
        st.caption(health.get("detail", ""))


tab1, tab2 = st.tabs(["📝 Crédito individual", "📂 Carga masiva (Excel/CSV)"])

# --------------------------------------------------------------------------- #
# TAB 1: Formulario individual
# --------------------------------------------------------------------------- #
with tab1:
    st.subheader("Datos del crédito")

    with st.form("form_credito"):
        col1, col2, col3 = st.columns(3)

        with col1:
            tipo_credito = st.number_input("Tipo de crédito", min_value=0, value=7, step=1)
            fecha_prestamo = st.date_input("Fecha del préstamo", value=date.today())
            capital_prestado = st.number_input("Capital prestado", min_value=0.0, value=3_000_000.0, step=1000.0)
            plazo_meses = st.number_input("Plazo (meses)", min_value=1, value=12, step=1)
            edad_cliente = st.number_input("Edad del cliente", min_value=18, max_value=100, value=35, step=1)
            tipo_laboral = st.selectbox("Tipo laboral", TIPOS_LABORALES)
            salario_cliente = st.number_input("Salario del cliente", min_value=0, value=3_000_000, step=100000)

        with col2:
            total_otros_prestamos = st.number_input("Total otros préstamos", min_value=0, value=0, step=100000)
            cuota_pactada = st.number_input("Cuota pactada", min_value=0, value=250000, step=10000)
            puntaje_datacredito = st.number_input("Puntaje Datacrédito", min_value=0.0, max_value=1000.0, value=700.0)
            cant_creditosvigentes = st.number_input("Créditos vigentes", min_value=0, value=2, step=1)
            huella_consulta = st.number_input("Huella de consulta", min_value=0, value=2, step=1)
            saldo_mora = st.number_input("Saldo en mora", min_value=0.0, value=0.0, step=1000.0)

        with col3:
            saldo_total = st.number_input("Saldo total", min_value=0.0, value=10000.0, step=1000.0)
            saldo_principal = st.number_input("Saldo principal", min_value=0.0, value=10000.0, step=1000.0)
            saldo_mora_codeudor = st.number_input("Saldo en mora (codeudor)", min_value=0.0, value=0.0, step=1000.0)
            creditos_sectorFinanciero = st.number_input("Créditos sector financiero", min_value=0, value=1, step=1)
            creditos_sectorCooperativo = st.number_input("Créditos sector cooperativo", min_value=0, value=0, step=1)
            creditos_sectorReal = st.number_input("Créditos sector real", min_value=0, value=0, step=1)
            promedio_ingresos_datacredito = st.number_input(
                "Promedio ingresos Datacrédito", min_value=0.0, value=900000.0, step=10000.0
            )

        submitted = st.form_submit_button("🔮 Predecir")

    if submitted:
        payload = {
            "tipo_credito": int(tipo_credito),
            "fecha_prestamo": fecha_prestamo.isoformat(),
            "capital_prestado": float(capital_prestado),
            "plazo_meses": int(plazo_meses),
            "edad_cliente": int(edad_cliente),
            "tipo_laboral": tipo_laboral,
            "salario_cliente": int(salario_cliente),
            "total_otros_prestamos": int(total_otros_prestamos),
            "cuota_pactada": int(cuota_pactada),
            "puntaje_datacredito": float(puntaje_datacredito),
            "cant_creditosvigentes": int(cant_creditosvigentes),
            "huella_consulta": int(huella_consulta),
            "saldo_mora": float(saldo_mora),
            "saldo_total": float(saldo_total),
            "saldo_principal": float(saldo_principal),
            "saldo_mora_codeudor": float(saldo_mora_codeudor),
            "creditos_sectorFinanciero": int(creditos_sectorFinanciero),
            "creditos_sectorCooperativo": int(creditos_sectorCooperativo),
            "creditos_sectorReal": int(creditos_sectorReal),
            "promedio_ingresos_datacredito": float(promedio_ingresos_datacredito),
        }

        with st.spinner("Consultando modelo..."):
            try:
                result = call_predict(payload)
            except Exception as exc:
                st.error(f"Error al predecir: {exc}")
                result = None

        if result:
            st.divider()
            st.subheader("Resultado")

            pred = result["prediction"]
            prob = result["probability_pago_atiempo"]

            col_a, col_b = st.columns(2)
            with col_a:
                if pred == 1:
                    st.success(f"✅ Predicción: **Paga a tiempo**")
                else:
                    st.error(f"⚠️ Predicción: **Riesgo de no pago**")
            with col_b:
                st.metric("Probabilidad de pago a tiempo", f"{prob * 100:.1f}%")

            st.progress(prob)
            st.caption(f"Modelo utilizado: {result['model_name']}")

            with st.expander("Ver payload enviado a la API"):
                st.json(payload)

# --------------------------------------------------------------------------- #
# TAB 2: Carga masiva
# --------------------------------------------------------------------------- #
with tab2:
    st.subheader("Cargar archivo con varios créditos")
    st.caption(
        "El archivo debe tener las mismas columnas que el dataset original "
        "(sin la columna `Pago_atiempo`, que es lo que se predice)."
    )

    uploaded_file = st.file_uploader("Subí un Excel (.xlsx) o CSV", type=["xlsx", "csv"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as exc:
            st.error(f"No se pudo leer el archivo: {exc}")
            df = None

        if df is not None:
            if "Pago_atiempo" in df.columns:
                df = df.drop(columns=["Pago_atiempo"])

            st.write(f"**{len(df)}** registros cargados.")
            st.dataframe(df.head(10), use_container_width=True)

            if st.button("🔮 Predecir lote completo"):
                # Convertir fechas a string ISO para que sean serializables en JSON
                df_json = df.copy()
                for col in df_json.select_dtypes(include=["datetime64[ns]", "datetime64[us]"]).columns:
                    df_json[col] = df_json[col].astype(str)

                records = json.loads(df_json.to_json(orient="records"))

                with st.spinner(f"Prediciendo {len(records)} registros..."):
                    try:
                        result = call_predict_batch(records)
                    except Exception as exc:
                        st.error(f"Error al predecir el lote: {exc}")
                        result = None

                if result:
                    predictions_df = pd.DataFrame(result["predictions"])
                    final_df = pd.concat([df.reset_index(drop=True), predictions_df], axis=1)

                    st.divider()
                    st.subheader("Resultados")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total evaluados", len(final_df))
                    col2.metric("Pagan a tiempo", int((final_df["prediction"] == 1).sum()))
                    col3.metric("Riesgo de no pago", int((final_df["prediction"] == 0).sum()))

                    def resaltar_riesgo(row):
                        color = "background-color: #ffcccc" if row["prediction"] == 0 else "background-color: #ccffcc"
                        return [color] * len(row)

                    st.dataframe(
                        final_df.style.apply(resaltar_riesgo, axis=1),
                        use_container_width=True,
                        hide_index=True,
                    )

                    csv_output = final_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Descargar resultados (CSV)",
                        data=csv_output,
                        file_name="predicciones_riesgo_crediticio.csv",
                        mime="text/csv",
                    )