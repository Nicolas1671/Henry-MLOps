"""
Tablero de Streamlit para monitoreo de data drift.

Lee los reportes generados por model_monitoring.py:
    - drift_report.json  -> foto de la última corrida
    - drift_history.csv  -> historial acumulado

"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

REPORTS_DIR = Path("mlops_pipeline/reports")
REPORT_PATH = REPORTS_DIR / "drift_report.json"
HISTORY_PATH = REPORTS_DIR / "drift_history.csv"

st.set_page_config(page_title="Monitoreo de Data Drift", layout="wide")
st.title("📊 Monitoreo de Data Drift - Riesgo Crediticio")


@st.cache_data
def load_report(path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_history(path):
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


report = load_report(REPORT_PATH)
history = load_history(HISTORY_PATH)

if report is None:
    st.warning(
        "No se encontró `drift_report.json`. Corré primero "
        "`python mlops_pipeline/src/model_monitoring.py` para generar los reportes."
    )
    st.stop()

detail_df = pd.DataFrame(report["detalle"])

tab1, tab2, tab3 = st.tabs(["📌 Última corrida", "📈 Evolución en el tiempo", "🚦 Recomendaciones"])

# --------------------------------------------------------------------------- #
# TAB 1: Métricas de la última corrida
# --------------------------------------------------------------------------- #
with tab1:
    st.subheader("Resumen general")
    col1, col2, col3 = st.columns(3)
    col1.metric("Variables analizadas", report["n_features_analizadas"])
    col2.metric("Variables con drift", report["n_features_con_drift"])
    col3.metric(
        "¿Requiere revisión?",
        "SÍ ⚠️" if report["requiere_revision_modelo"] else "NO ✅",
    )
    st.caption(f"Última corrida: {report['timestamp']}")

    st.divider()
    st.subheader("Detalle por variable")

    solo_drift = st.checkbox("Mostrar solo variables con drift", value=False)
    tabla = detail_df[detail_df["drift_detected"]] if solo_drift else detail_df

    def resaltar_drift(row):
        color = "background-color: #ffcccc" if row["drift_detected"] else ""
        return [color] * len(row)

    st.dataframe(
        tabla.style.apply(resaltar_drift, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader("PSI por variable")
    fig_psi = px.bar(
        detail_df.sort_values("psi", ascending=False),
        x="feature",
        y="psi",
        color="drift_detected",
        color_discrete_map={True: "#e74c3c", False: "#2ecc71"},
        labels={"feature": "Variable", "psi": "PSI", "drift_detected": "Drift"},
    )
    fig_psi.add_hline(y=0.25, line_dash="dash", line_color="red", annotation_text="Umbral crítico (0.25)")
    fig_psi.add_hline(y=0.10, line_dash="dot", line_color="orange", annotation_text="Umbral moderado (0.10)")
    st.plotly_chart(fig_psi, use_container_width=True)

    st.subheader("Jensen-Shannon divergence por variable")
    fig_js = px.bar(
        detail_df.sort_values("js_divergence", ascending=False),
        x="feature",
        y="js_divergence",
        color="drift_detected",
        color_discrete_map={True: "#e74c3c", False: "#2ecc71"},
        labels={"feature": "Variable", "js_divergence": "JS Divergence"},
    )
    fig_js.add_hline(y=0.1, line_dash="dash", line_color="red", annotation_text="Umbral (0.1)")
    st.plotly_chart(fig_js, use_container_width=True)

# --------------------------------------------------------------------------- #
# TAB 2: Evolución en el tiempo
# --------------------------------------------------------------------------- #
with tab2:
    if history is None or history.empty:
        st.info("Todavía no hay historial suficiente. Ejecutá el monitoreo varias veces para ver la evolución.")
    else:
        st.subheader("Evolución de PSI en el tiempo")
        variables_disponibles = sorted(history["feature"].unique())
        variables_seleccionadas = st.multiselect(
            "Seleccioná variables a graficar",
            variables_disponibles,
            default=variables_disponibles[: min(5, len(variables_disponibles))],
        )

        if variables_seleccionadas:
            filtrado = history[history["feature"].isin(variables_seleccionadas)]
            fig_evol_psi = px.line(
                filtrado,
                x="timestamp",
                y="psi",
                color="feature",
                markers=True,
                labels={"timestamp": "Fecha", "psi": "PSI", "feature": "Variable"},
            )
            fig_evol_psi.add_hline(y=0.25, line_dash="dash", line_color="red")
            fig_evol_psi.add_hline(y=0.10, line_dash="dot", line_color="orange")
            st.plotly_chart(fig_evol_psi, use_container_width=True)

            st.subheader("Evolución de Jensen-Shannon en el tiempo")
            fig_evol_js = px.line(
                filtrado,
                x="timestamp",
                y="js_divergence",
                color="feature",
                markers=True,
                labels={"timestamp": "Fecha", "js_divergence": "JS Divergence", "feature": "Variable"},
            )
            fig_evol_js.add_hline(y=0.1, line_dash="dash", line_color="red")
            st.plotly_chart(fig_evol_js, use_container_width=True)

            st.subheader("Cantidad de corridas con drift por variable")
            resumen = (
                filtrado.groupby("feature")["drift_detected"]
                .agg(["sum", "count"])
                .rename(columns={"sum": "corridas_con_drift", "count": "total_corridas"})
                .reset_index()
            )
            resumen["porcentaje_drift"] = (resumen["corridas_con_drift"] / resumen["total_corridas"] * 100).round(1)
            st.dataframe(resumen, use_container_width=True, hide_index=True)
        else:
            st.info("Seleccioná al menos una variable para graficar.")

        st.divider()
        st.subheader("Historial completo")
        st.dataframe(history.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------- #
# TAB 3: Recomendaciones automáticas
# --------------------------------------------------------------------------- #
with tab3:
    st.subheader("Diagnóstico automático")

    if report["requiere_revision_modelo"]:
        st.error(
            f"⚠️ Se detectó drift en **{report['n_features_con_drift']}** de "
            f"**{report['n_features_analizadas']}** variables. Se recomienda revisar el modelo."
        )
    else:
        st.success("✅ No se detectó drift relevante. El modelo puede seguir operando sin cambios.")

    drifted_vars = detail_df[detail_df["drift_detected"]]

    if not drifted_vars.empty:
        st.markdown("### Variables a revisar")
        for _, row in drifted_vars.iterrows():
            with st.expander(f"🔴 {row['feature']} ({row['tipo']})"):
                st.write(f"**PSI:** {row['psi']:.4f} → *{row['psi_level']}*")
                st.write(f"**Jensen-Shannon:** {row['js_divergence']:.4f}")
                if row["tipo"] == "numerica":
                    st.write(f"**KS p-value:** {row['ks_pvalue']:.4f}")
                else:
                    chi2_p = row.get("chi2_pvalue")
                    st.write(f"**Chi² p-value:** {chi2_p:.4f}" if pd.notna(chi2_p) else "**Chi² p-value:** N/A")

                recomendaciones = []
                if row["psi_level"] == "cambio_significativo":
                    recomendaciones.append(
                        "El PSI indica un cambio significativo en la distribución. "
                        "Evaluar reentrenamiento del modelo con datos recientes."
                    )
                elif row["psi_level"] == "cambio_moderado":
                    recomendaciones.append(
                        "El PSI indica un cambio moderado. Monitorear de cerca en las próximas corridas."
                    )

                if row.get("drift_detected_js"):
                    recomendaciones.append(
                        "La divergencia Jensen-Shannon confirma un cambio en la forma de la distribución."
                    )

                if row["tipo"] == "numerica" and row.get("drift_detected_ks"):
                    recomendaciones.append(
                        "El test KS confirma que la distribución numérica cambió de forma estadísticamente significativa."
                    )

                if row["tipo"] == "categorica" and row.get("drift_detected_chi2"):
                    recomendaciones.append(
                        "El test Chi-cuadrado indica un cambio en las proporciones de las categorías."
                    )

                for rec in recomendaciones:
                    st.write(f"- {rec}")
    else:
        st.info("No hay variables que requieran atención en esta corrida.")

    st.divider()
    st.subheader("Próximos pasos sugeridos")
    if report["requiere_revision_modelo"]:
        st.markdown(
            """
            1. **Priorizar** las variables marcadas con `cambio_significativo`.
            2. Analizar si el cambio se debe a un **problema de calidad de datos** (bug en el pipeline de ingesta)
               o a un **cambio real en el comportamiento** de los solicitantes de crédito.
            3. Si el cambio es real, evaluar **reentrenar el modelo** con datos más recientes.
            4. Volver a correr `model_monitoring.py` tras cada actualización de datos para confirmar que el drift se resolvió.
            """
        )
    else:
        st.markdown(
            """
            - Mantener la cadencia actual de monitoreo (ej. semanal/mensual).
            - No se requieren acciones sobre el modelo en este momento.
            """
        )