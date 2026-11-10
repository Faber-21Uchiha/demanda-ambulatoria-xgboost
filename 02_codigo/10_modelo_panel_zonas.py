# -*- coding: utf-8 -*-
"""
10_modelo_panel_zonas.py

Extiende la Fase 3 al nivel zona-semana (OE3) y agrega los referentes que
faltaban en la version anterior:

  - XGBoost con panel zona-semana (420 observaciones potenciales).
  - Referentes por zona: media historica, naive t-1, naive estacional t-52 y
    regresion lineal multiple.
  - SARIMA sobre la serie agregada de la red (referente estadistico clasico
    declarado en la seccion 7.7).

Salidas en 03_resultados/: resultados_modelo_panel.json, predicciones_panel.csv
y resultados_sarima.json.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

from panel_features import PARAMS_XGB, columnas_modelo, construir_panel

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[1]
DATOS = BASE / "01_datos"
RES = BASE / "03_resultados"
SEED = 2026
N_TEST_SEMANAS = 18


def metricas(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    return {
        "MAE": round(float(mean_absolute_error(y, p)), 1),
        "RMSE": round(float(np.sqrt(mean_squared_error(y, p))), 1),
        "MAPE": round(float(np.mean(np.abs((y - p) / y)) * 100), 2),
        "R2": round(float(r2_score(y, p)), 3),
    }


def main():
    p, X = construir_panel()
    semanas = sorted(X["semana"].unique())
    test_semanas = set(semanas[-N_TEST_SEMANAS:])
    test = X["semana"].isin(test_semanas)
    tr, te = X[~test].copy(), X[test].copy()
    cols = columnas_modelo(X)

    modelo = xgb.XGBRegressor(**PARAMS_XGB).fit(tr[cols], tr["y"])
    pred = modelo.predict(te[cols])

    # regresion lineal multiple con el mismo conjunto de predictores
    lin = LinearRegression().fit(tr[cols], tr["y"])
    pred_lin = lin.predict(te[cols])

    te = te.copy()
    te["pred_xgb"] = pred
    te["pred_lin"] = pred_lin

    res = {
        "n_obs_panel": int(len(X)),
        "n_semanas": int(X["semana"].nunique()),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "train_ini": str(pd.Timestamp(tr["semana"].min()).date()),
        "train_fin": str(pd.Timestamp(tr["semana"].max()).date()),
        "test_ini": str(pd.Timestamp(te["semana"].min()).date()),
        "test_fin": str(pd.Timestamp(te["semana"].max()).date()),
        "n_predictores": len(cols),
        "xgboost": metricas(te["y"], te["pred_xgb"]),
        "regresion": metricas(te["y"], te["pred_lin"]),
        "media_historica": metricas(te["y"], te["media_historica"]),
        "naive_1": metricas(te["y"], te["naive_1"]),
        "naive_52": metricas(te["y"], te["naive_52"]),
        "por_zona": {},
    }
    for z in sorted(te["zona"].unique()):
        s = te[te["zona"] == z]
        res["por_zona"][z] = {
            "n": int(len(s)),
            "xgboost": metricas(s["y"], s["pred_xgb"]),
            "naive_52": metricas(s["y"], s["naive_52"]),
        }
    # mejora porcentual de MAE/MAPE frente al mejor referente por zona
    res["mejora_vs_naive52_pct"] = {
        "MAE": round(100 * (1 - res["xgboost"]["MAE"] / res["naive_52"]["MAE"]), 1),
        "MAPE": round(100 * (1 - res["xgboost"]["MAPE"] / res["naive_52"]["MAPE"]), 1),
    }

    (RES / "resultados_modelo_panel.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    salida = te[["semana", "zona", "y", "pred_xgb", "pred_lin", "naive_52"]].copy()
    salida["semana"] = salida["semana"].dt.strftime("%Y-%m-%d")
    salida.to_csv(RES / "predicciones_panel.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------ SARIMA
    agregada = p.groupby("semana")["demanda"].sum().sort_index()
    y = agregada.reindex(pd.date_range(agregada.index.min(), agregada.index.max(), freq="7D"))
    tr_y, te_y = y.iloc[:-N_TEST_SEMANAS], y.iloc[-N_TEST_SEMANAS:]
    naive52_agregado = y.shift(52).iloc[-N_TEST_SEMANAS:].values
    sarima = {"estado": "no estimado"}
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        best = None
        for orden, est in [((1, 1, 1), (0, 1, 0, 52)), ((1, 0, 1), (1, 0, 0, 52)),
                           ((0, 1, 1), (0, 1, 1, 52)), ((1, 1, 0), (0, 1, 1, 52)),
                           ((1, 1, 1), (0, 1, 1, 52))]:
            try:
                m = SARIMAX(tr_y, order=orden, seasonal_order=est,
                            enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                pr = m.forecast(N_TEST_SEMANAS)
                met = metricas(te_y.values, pr.values)
                if best is None or met["MAE"] < best["metricas"]["MAE"]:
                    best = {"orden": list(orden), "orden_estacional": list(est),
                            "metricas": met, "aic": round(float(m.aic), 1),
                            "pred": [round(float(v), 1) for v in pr.values]}
            except Exception as exc:  # noqa: BLE001
                import traceback
                sarima.setdefault("errores", []).append(f"{orden}: {exc}\n{traceback.format_exc()}")
        if best:
            sarima = {"estado": "estimado", **best,
                      "naive_52": metricas(te_y.values, naive52_agregado),
                      "errores": sarima.get("errores", [])}
    except Exception as exc:  # noqa: BLE001
        sarima = {"estado": "error", "detalle": str(exc)}

    sarima.update({
        "n_train": int(len(tr_y)), "n_test": int(len(te_y)),
        "test_ini": str(te_y.index.min().date()), "test_fin": str(te_y.index.max().date()),
        "pred": sarima.get("pred", []),
        "fechas_test": [str(d.date()) for d in te_y.index],
        "y_test": [float(v) for v in te_y.values],
    })
    (RES / "resultados_sarima.json").write_text(
        json.dumps(sarima, ensure_ascii=False, indent=1), encoding="utf-8")

    print(json.dumps(res, ensure_ascii=False, indent=1))
    print("\nSARIMA:", json.dumps({k: v for k, v in sarima.items() if k not in ("pred",)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
