# -*- coding: utf-8 -*-
"""
12_protocolo_oe4.py

Objetivo Especifico 4: definir el protocolo de operacion del modelo predictivo
en la Red Integrada de Servicios de Salud Metropolitana E.S.E.

Produce los artefactos que una oficina de analitica necesita para poner el modelo
en produccion y mantenerlo:

  1. Modelo final reentrenado con todas las observaciones disponibles, en formato
     binario UBJ, con su manifiesto de versiones, semilla y huellas SHA-256.
  2. Tamano del artefacto, numero de arboles y de nodos.
  3. Latencia de inferencia por lote (4, 52, 400 y 5.000 registros), en ms y en
     microsegundos por registro.
  4. Volumen de datos de entrada, de entrenamiento y de la actualizacion semanal.
  5. Indice de estabilidad poblacional (PSI) de cada predictor entre 2024 y 2025,
     con los umbrales habituales de 0,10 (alerta) y 0,25 (reentrenar).
  6. Regla de reentrenamiento y calendario de monitoreo.

La latencia se mide con time.perf_counter sobre el modelo ya cargado en memoria;
no incluye la construccion de variables ni la consulta a la base de datos.
"""
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from panel_features import PARAMS_XGB, columnas_modelo, construir_panel

BASE = Path(__file__).resolve().parents[1]
DATOS = BASE / "01_datos"
RES = BASE / "03_resultados"
MODELOS = BASE / "03_resultados" / "modelos"
MODELOS.mkdir(parents=True, exist_ok=True)

LOTE_LATENCIA = (4, 52, 400, 5000)
REPETICIONES = 300
UMBRAL_ALERTA = 0.10
UMBRAL_REENTRENAR = 0.25
# Variables de calendario y tendencia: su distribucion cambia por diseno entre
# anos, por lo que el PSI no es informativo y se excluyen del monitoreo.
EXCLUIR_PSI = {"t", "woy_sin", "woy_cos"}


def sha256(ruta, bloque=1 << 20):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for trozo in iter(lambda: f.read(bloque), b""):
            h.update(trozo)
    return h.hexdigest()


def psi(ref, act, bins=10):
    """Indice de estabilidad poblacional entre dos muestras."""
    ref = np.asarray(ref, float)
    act = np.asarray(act, float)
    ref = ref[~np.isnan(ref)]
    act = act[~np.isnan(act)]
    if len(ref) == 0 or len(act) == 0:
        return None
    if np.unique(ref).size <= 2:
        p = np.array([(ref == v).mean() for v in (0, 1)])
        q = np.array([(act == v).mean() for v in (0, 1)])
    else:
        cortes = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        if len(cortes) < 3:
            return None
        cortes[0], cortes[-1] = -np.inf, np.inf
        p = np.histogram(ref, bins=cortes)[0] / len(ref)
        q = np.histogram(act, bins=cortes)[0] / len(act)
    p = np.clip(p, 1e-4, None)
    q = np.clip(q, 1e-4, None)
    p, q = p / p.sum(), q / q.sum()
    return round(float(np.sum((p - q) * np.log(p / q))), 4)


def medir_latencia(modelo, X, cols):
    resultados = {}
    for n in LOTE_LATENCIA:
        base = X[cols].iloc[: min(n, len(X))].copy()
        if len(base) < n:
            base = pd.concat([base] * int(np.ceil(n / len(base))), ignore_index=True).iloc[:n]
        modelo.predict(base)  # calentamiento
        tiempos = []
        for _ in range(REPETICIONES):
            t0 = time.perf_counter()
            modelo.predict(base)
            tiempos.append((time.perf_counter() - t0) * 1000)
        tiempos = np.array(tiempos)
        resultados[str(n)] = {
            "p50_ms": round(float(np.percentile(tiempos, 50)), 4),
            "p95_ms": round(float(np.percentile(tiempos, 95)), 4),
            "max_ms": round(float(tiempos.max()), 4),
            "us_por_registro": round(float(np.percentile(tiempos, 50) * 1000 / n), 2),
        }
    return resultados


def main():
    p, X = construir_panel()
    cols = columnas_modelo(X)

    modelo = xgb.XGBRegressor(**PARAMS_XGB).fit(X[cols], X["y"])
    ruta_ubj = MODELOS / "modelo_panel_xgboost.ubj"
    modelo.save_model(ruta_ubj)
    booster = modelo.get_booster()
    arboles = booster.get_dump(fmap="", with_stats=False)
    n_nodos = sum(line.count("[") for t in arboles for line in t.splitlines() if "leaf=" not in line)

    X["anio"] = X["semana"].dt.year
    tabla_psi = []
    for c in cols:
        if c in EXCLUIR_PSI:
            continue
        v = psi(X.loc[X["anio"] == 2024, c], X.loc[X["anio"] == 2025, c])
        if v is not None:
            tabla_psi.append({"variable": c, "psi": v,
                              "estado": ("reentrenar" if v >= UMBRAL_REENTRENAR
                                         else "alerta" if v >= UMBRAL_ALERTA else "estable")})
    tabla_psi = sorted(tabla_psi, key=lambda d: d["psi"], reverse=True)
    pd.DataFrame(tabla_psi).to_csv(RES / "tabla_psi_deriva.csv", index=False, encoding="utf-8-sig")

    crudo = DATOS / "demanda_ambulatoria_sintetica_2024_2025_100k.csv"
    depurado = DATOS / "dataset_depurado_OE1.csv"
    panel_csv = DATOS / "panel_zona_semana_capacidad.csv"
    pqrs_csv = DATOS / "pqrs_sinteticas_2024_2025.csv"

    n_crudo = int(len(pd.read_csv(crudo, usecols=[0], encoding="utf-8-sig")))
    n_dep = int(len(pd.read_csv(depurado, usecols=[0], encoding="utf-8-sig")))
    n_pqrs = int(len(pd.read_csv(pqrs_csv, usecols=[0], encoding="utf-8-sig")))
    semanales = p.groupby("semana")["demanda"].sum()

    latencia = medir_latencia(modelo, X, cols)
    alerta = [d for d in tabla_psi if d["estado"] == "alerta"]
    reentrenar = [d for d in tabla_psi if d["estado"] == "reentrenar"]

    res = {
        "modelo": {
            "algoritmo": "XGBoost regresion",
            "version_xgboost": xgb.__version__,
            "python": platform.python_version(),
            "n_estimadores": int(PARAMS_XGB["n_estimators"]),
            "profundidad_maxima": int(PARAMS_XGB["max_depth"]),
            "semilla": int(PARAMS_XGB["random_state"]),
            "n_predictores": len(cols),
            "predictores": cols,
            "n_arboles": len(arboles),
            "n_nodos_internos": n_nodos,
            "tamano_ubj_bytes": ruta_ubj.stat().st_size,
            "tamano_ubj_kb": round(ruta_ubj.stat().st_size / 1024, 1),
            "sha256_ubj": sha256(ruta_ubj),
        },
        "latencia_inferencia": latencia,
        "volumen_datos": {
            "crudo_registros": n_crudo,
            "crudo_mb": round(crudo.stat().st_size / (1024 ** 2), 1),
            "depurado_registros": n_dep,
            "depurado_mb": round(depurado.stat().st_size / (1024 ** 2), 1),
            "panel_registros": int(len(p)),
            "pqrs_registros": n_pqrs,
            "semanas_panel": int(p["semana"].nunique()),
            "demanda_semanal_media": int(round(float(semanales.mean()))),
            "demanda_semanal_sd": int(round(float(semanales.std()))),
            "registros_nuevos_por_semana": int(round(float(semanales.mean()))),
            "observaciones_entrenamiento": int(len(X)),
            "memoria_modelo_kb": round(booster.save_raw("ubj").__len__() / 1024, 1),
        },
        "deriva_psi": {
            "metodo": "PSI con 10 bins por cuantiles de la referencia 2024",
            "excluidas": sorted(EXCLUIR_PSI),
            "umbral_alerta": UMBRAL_ALERTA,
            "umbral_reentrenar": UMBRAL_REENTRENAR,
            "n_variables_evaluadas": len(tabla_psi),
            "n_alerta": len(alerta),
            "n_reentrenar": len(reentrenar),
            "variables_alerta": [d["variable"] for d in alerta],
            "variables_reentrenar": [d["variable"] for d in reentrenar],
            "psi_maximo": tabla_psi[0]["psi"] if tabla_psi else None,
            "psi_mediana": round(float(np.median([d["psi"] for d in tabla_psi])), 4),
            "detalle": tabla_psi,
        },
        "regla_reentrenamiento": {
            "monitoreo_semanal": ("calcular PSI de los predictores sustantivos y el error de la "
                                  "ultima semana contra el valor observado"),
            "alerta": f"PSI >= {UMBRAL_ALERTA} en cualquier predictor o MAPE mensual 20 % superior a la linea base",
            "reentrenamiento_programado": "trimestral con ventana movil de 24 meses",
            "reentrenamiento_extraordinario": f"PSI >= {UMBRAL_REENTRENAR} o cambio de politica de agenda",
            "validacion": ("comparar challenger contra champion en las 8 semanas previas; "
                           "promover solo si reduce el MAE al menos 5 %"),
            "versionado": "registro de fecha, semilla, hash del modelo y del conjunto de datos",
            "reversion": "conservar la version anterior del artefacto UBJ durante 6 meses",
        },
        "huellas_datos": {
            "depurado_sha256": sha256(depurado),
            "panel_sha256": sha256(panel_csv),
            "pqrs_sha256": sha256(pqrs_csv),
            "script_features_sha256": sha256(Path(__file__).resolve().parent / "panel_features.py"),
        },
    }
    (RES / "resultados_protocolo_oe4.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "deriva_psi"}, ensure_ascii=False, indent=1))
    print("\nPSI (top 12):")
    for d in tabla_psi[:12]:
        print(f"  {d['variable']:<28} {d['psi']:>7.4f}  {d['estado']}")


if __name__ == "__main__":
    main()
