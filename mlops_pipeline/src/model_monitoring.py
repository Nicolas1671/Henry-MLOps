"""
Monitoreo de data drift para el modelo de riesgo crediticio.

Compara la distribución histórica de cada variable (con la que se entrenó el modelo)
contra la de un lote reciente, y reporta si alguna se movió lo suficiente como para
revisar el modelo.

Metricas utilizadas:
    - Kolmogorov-Smirnov (KS test)      -> variables numéricas
    - Population Stability Index (PSI)  -> variables numéricas y categóricas
    - Jensen-Shannon divergence         -> variables numéricas y categóricas
    - Chi-cuadrado                      -> variables categóricas

Cada corrida genera:
    drift_report.json  -> foto de la última corrida (se sobrescribe)
    drift_history.csv  -> historial acumulado (se agrega una fila por variable)

"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp
from scipy.spatial.distance import jensenshannon

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

THRESHOLDS = {
    "ks_pvalue": 0.05,
    "psi": {"low": 0.1, "moderate": 0.25},
    "js_divergence": 0.1,
    "chi2_pvalue": 0.05,
}

N_BINS = 10


def ks_test(reference, current):
    ref = reference.dropna()
    cur = current.dropna()
    stat, p_value = ks_2samp(ref, cur)
    return {
        "ks_statistic": float(stat),
        "ks_pvalue": float(p_value),
        "drift_detected_ks": bool(p_value < THRESHOLDS["ks_pvalue"]),
    }


def _get_bins(reference, n_bins=N_BINS):
    quantiles = np.linspace(0, 1, n_bins + 1)
    bins = np.unique(reference.quantile(quantiles).values)
    if len(bins) < 3:
        bins = np.linspace(reference.min(), reference.max(), n_bins + 1)
    return bins


def _psi_numeric(reference, current):
    ref, cur = reference.dropna(), current.dropna()
    bins = _get_bins(ref)
    bins[0], bins[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(ref, bins=bins)
    cur_counts, _ = np.histogram(cur, bins=bins)

    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-6, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _psi_categorical(reference, current):
    ref, cur = reference.dropna(), current.dropna()
    categories = pd.Index(ref.unique()).union(cur.unique())

    ref_pct = ref.value_counts(normalize=True).reindex(categories, fill_value=1e-6).clip(lower=1e-6)
    cur_pct = cur.value_counts(normalize=True).reindex(categories, fill_value=1e-6).clip(lower=1e-6)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def psi(reference, current, is_categorical):
    value = _psi_categorical(reference, current) if is_categorical else _psi_numeric(reference, current)

    if value < THRESHOLDS["psi"]["low"]:
        level = "sin_cambio"
    elif value < THRESHOLDS["psi"]["moderate"]:
        level = "cambio_moderado"
    else:
        level = "cambio_significativo"

    return {
        "psi": value,
        "psi_level": level,
        "drift_detected_psi": bool(value >= THRESHOLDS["psi"]["moderate"]),
    }


def js_divergence(reference, current, is_categorical):
    ref, cur = reference.dropna(), current.dropna()

    if is_categorical:
        categories = pd.Index(ref.unique()).union(cur.unique())
        ref_dist = ref.value_counts(normalize=True).reindex(categories, fill_value=0.0).values
        cur_dist = cur.value_counts(normalize=True).reindex(categories, fill_value=0.0).values
    else:
        bins = _get_bins(ref)
        ref_counts, _ = np.histogram(ref, bins=bins)
        cur_counts, _ = np.histogram(cur, bins=bins)
        ref_dist = ref_counts / max(ref_counts.sum(), 1)
        cur_dist = cur_counts / max(cur_counts.sum(), 1)

    js = jensenshannon(ref_dist, cur_dist, base=2)
    js = float(js) if not np.isnan(js) else 0.0

    return {"js_divergence": js, "drift_detected_js": bool(js > THRESHOLDS["js_divergence"])}


def chi_square_test(reference, current):
    ref, cur = reference.dropna(), current.dropna()
    categories = pd.Index(ref.unique()).union(cur.unique())

    ref_counts = ref.value_counts().reindex(categories, fill_value=0)
    cur_counts = cur.value_counts().reindex(categories, fill_value=0)

    contingency = pd.DataFrame({"reference": ref_counts, "current": cur_counts}).T
    contingency = contingency.loc[:, contingency.sum(axis=0) > 0]

    try:
        stat, p_value, _, _ = chi2_contingency(contingency)
    except ValueError:
        stat, p_value = np.nan, np.nan

    return {
        "chi2_statistic": None if np.isnan(stat) else float(stat),
        "chi2_pvalue": None if np.isnan(p_value) else float(p_value),
        "drift_detected_chi2": bool(p_value < THRESHOLDS["chi2_pvalue"]) if not np.isnan(p_value) else False,
    }


def _is_categorical(series):
    return series.dtype == object or str(series.dtype) == "category" or series.nunique() <= 15


def analyze_column(reference, current, column_name):
    is_cat = _is_categorical(reference)
    result = {
        "feature": column_name,
        "tipo": "categorica" if is_cat else "numerica",
        "n_reference": int(reference.dropna().shape[0]),
        "n_current": int(current.dropna().shape[0]),
    }

    if is_cat:
        result.update(psi(reference, current, is_categorical=True))
        result.update(js_divergence(reference, current, is_categorical=True))
        result.update(chi_square_test(reference, current))
        drift_flags = [result["drift_detected_psi"], result["drift_detected_js"], result["drift_detected_chi2"]]
    else:
        result.update(ks_test(reference, current))
        result.update(psi(reference, current, is_categorical=False))
        result.update(js_divergence(reference, current, is_categorical=False))
        drift_flags = [result["drift_detected_ks"], result["drift_detected_psi"], result["drift_detected_js"]]

    result["drift_detected"] = bool(sum(drift_flags) >= 2)
    return result


def run_drift_analysis(reference_df, current_df):
    common_columns = [c for c in reference_df.columns if c in current_df.columns]
    if not common_columns:
        raise ValueError("No hay columnas en común entre reference y current.")

    results = []
    for column in common_columns:
        try:
            result = analyze_column(reference_df[column], current_df[column], column)
            results.append(result)
            flag = "DRIFT" if result["drift_detected"] else "ok"
            logger.info("Columna '%s' -> %s", column, flag)
        except Exception as exc:
            logger.warning("No se pudo analizar la columna '%s': %s", column, exc)
    return results


def save_reports(results, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()

    n_drifted = sum(r["drift_detected"] for r in results)
    report = {
        "timestamp": timestamp,
        "n_features_analizadas": len(results),
        "n_features_con_drift": n_drifted,
        "requiere_revision_modelo": n_drifted > 0,
        "detalle": results,
    }

    report_path = output_dir / "drift_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Reporte JSON guardado en: %s", report_path)

    history_path = output_dir / "drift_history.csv"
    history_rows = pd.DataFrame(results)
    history_rows.insert(0, "timestamp", timestamp)

    if history_path.exists():
        history_rows.to_csv(history_path, mode="a", header=False, index=False)
    else:
        history_rows.to_csv(history_path, mode="w", header=True, index=False)
    logger.info("Historial CSV actualizado en: %s", history_path)

    if n_drifted > 0:
        logger.warning("⚠️  %s/%s variables muestran drift. Se recomienda revisar el modelo.", n_drifted, len(results))
    else:
        logger.info("✅ No se detectó drift relevante en ninguna variable.")


def parse_args():
    parser = argparse.ArgumentParser(description="Monitoreo de data drift - riesgo crediticio")
    parser.add_argument("--reference", default="Base_de_datos.xlsx", help="Excel/CSV con el dataset histórico")
    parser.add_argument("--current", default="Base_de_datos_drifted.xlsx", help="Excel/CSV con el lote reciente")
    parser.add_argument("--output-dir", default="mlops_pipeline/reports", help="Carpeta de salida de los reportes")
    return parser.parse_args()


def _read_any(path):
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def main():
    args = parse_args()

    logger.info("Cargando dataset de referencia: %s", args.reference)
    reference_df = _read_any(args.reference)

    logger.info("Cargando dataset actual: %s", args.current)
    current_df = _read_any(args.current)

    results = run_drift_analysis(reference_df, current_df)
    save_reports(results, Path(args.output_dir))


if __name__ == "__main__":
    main()