# -*- coding: utf-8 -*-
"""Fase 3 (OE3): validacion temporal detallada, importancia, residuos e intervalos."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from statsmodels.stats.diagnostic import acorr_ljungbox

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "03_resultados"

df = pd.read_csv(BASE / "01_datos" / "dataset_depurado_OE1.csv", encoding="utf-8-sig")
df["fecha_atencion"] = pd.to_datetime(df["fecha_atencion"])

g = df.groupby(pd.Grouper(key="fecha_atencion", freq="W-SUN"))
w = pd.DataFrame({
    "y": g.size(),
    "medicina_general": g["servicio"].apply(lambda s: (s == "Medicina general").mean()),
    "enfermeria": g["servicio"].apply(lambda s: (s == "Enfermería").mean()),
    "odontologia": g["servicio"].apply(lambda s: (s == "Odontología").mean()),
    "pediatria": g["servicio"].apply(lambda s: (s == "Pediatría").mean()),
    "apoyo_diagnostico": g["servicio"].apply(lambda s: s.isin(["Laboratorio", "Imágenes diagnósticas"]).mean()),
    "edad_media": g["edad"].mean(),
    "prop_femenino": g["sexo"].apply(lambda s: (s == "F").mean()),
    "prop_mayor_60": g["grupo_edad"].apply(lambda s: s.isin(["60-74", "75+"]).mean()),
    "prop_contributivo": g["regimen"].apply(lambda s: (s == "Contributivo").mean()),
    "prop_prioridad_pref": g["prioridad"].apply(lambda s: (s == "Preferente").mean()),
    "prop_remision": g["remision"].apply(lambda s: (s == "Sí").mean()),
    "espera_media": g["tiempo_espera_min"].mean(),
})
w = w[w["y"] > 0].asfreq("W-SUN")
w = w[w.index <= "2025-12-28"]
y = w["y"].astype(float)

exo = ["medicina_general", "enfermeria", "odontologia", "pediatria", "apoyo_diagnostico",
       "edad_media", "prop_femenino", "prop_mayor_60", "prop_contributivo",
       "prop_prioridad_pref", "prop_remision", "espera_media"]
X = pd.DataFrame(index=w.index)
for lag in (1, 2, 3, 4):
    X[f"lag{lag}"] = y.shift(lag)
for win in (4, 8, 13):
    X[f"roll{win}"] = y.shift(1).rolling(win).mean()
X["roll_std8"] = y.shift(1).rolling(8).std()
X = X.join(w[exo].shift(1).add_suffix("_lag1"))
woy = X.index.isocalendar().week.astype(int)
X["woy_sin"] = np.sin(2 * np.pi * woy / 52)
X["woy_cos"] = np.cos(2 * np.pi * woy / 52)
X["t"] = np.arange(len(X))
X["y"] = y
X = X.dropna()

n = len(X)
corte = int(round(n * 0.8))
tr, te = X.iloc[:corte], X.iloc[corte:]
cols = [c for c in X.columns if c != "y"]


def metrics(ytrue, ypred):
    ytrue = np.asarray(ytrue, float)
    ypred = np.asarray(ypred, float)
    return {
        "MAE": round(float(mean_absolute_error(ytrue, ypred)), 1),
        "RMSE": round(float(np.sqrt(mean_squared_error(ytrue, ypred))), 1),
        "MAPE": round(float(np.mean(np.abs((ytrue - ypred) / ytrue)) * 100), 2),
        "R2": round(float(r2_score(ytrue, ypred)), 3),
    }


params = dict(max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
              min_child_weight=2, reg_lambda=1.0, random_state=2026, n_jobs=2)

Xtr, ytr = tr[cols], tr["y"]
fold_rows = []
tscv = TimeSeriesSplit(n_splits=4)
for k, (tr_idx, va_idx) in enumerate(tscv.split(Xtr), start=1):
    m = xgb.XGBRegressor(n_estimators=600, early_stopping_rounds=40, eval_metric="mae", **params)
    m.fit(Xtr.iloc[tr_idx], ytr.iloc[tr_idx], eval_set=[(Xtr.iloc[va_idx], ytr.iloc[va_idx])], verbose=False)
    p = m.predict(Xtr.iloc[va_idx])
    met = metrics(ytr.iloc[va_idx], p)
    fold_rows.append({
        "fold": k,
        "n_train": int(len(tr_idx)),
        "n_val": int(len(va_idx)),
        "v_ini": str(Xtr.index[va_idx[0]].date()),
        "v_fin": str(Xtr.index[va_idx[-1]].date()),
        "rounds": int(m.best_iteration) + 1,
        **met,
    })

nr = int(np.median([f["rounds"] for f in fold_rows]))
xg = xgb.XGBRegressor(n_estimators=max(nr, 50), **params).fit(Xtr, ytr)
pred = xg.predict(te[cols])
met_xgb = metrics(te["y"], pred)

# intervalos de prediccion al 90 % con XGBoost cuantilico
qmodels = {}
for alpha in (0.05, 0.95):
    qm = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=alpha,
                          n_estimators=max(nr, 50), **params).fit(Xtr, ytr)
    qmodels[alpha] = qm
lo = qmodels[0.05].predict(te[cols])
hi = qmodels[0.95].predict(te[cols])
cover = float(np.mean((te["y"].values >= lo) & (te["y"].values <= hi)))

# residuos
res = te["y"].values - pred
lb = acorr_ljungbox(res, lags=[4], return_df=True)
dw = float(np.sum(np.diff(res) ** 2) / np.sum(res ** 2))
acf1 = float(np.corrcoef(res[:-1], res[1:])[0, 1])

# importancia por ganancia
gain = xg.get_booster().get_score(importance_type="gain")
tot = sum(gain.values())
imp = sorted(((k, round(v / tot * 100, 2)) for k, v in gain.items()), key=lambda t: -t[1])

out = {
    "n_total": int(n), "n_train": int(len(tr)), "n_test": int(len(te)),
    "train_ini": str(tr.index.min().date()), "train_fin": str(tr.index.max().date()),
    "test_ini": str(te.index.min().date()), "test_fin": str(te.index.max().date()),
    "n_features": len(cols),
    "params": {k: v for k, v in params.items()},
    "n_estimators_final": int(max(nr, 50)),
    "folds": fold_rows,
    "metricas_xgb": met_xgb,
    "cobertura_pi90": round(cover * 100, 1),
    "ancho_pi90_medio": round(float(np.mean(hi - lo)), 1),
    "residuos": {
        "media": round(float(res.mean()), 1),
        "desv": round(float(res.std(ddof=1)), 1),
        "min": round(float(res.min()), 1),
        "max": round(float(res.max()), 1),
        "ljungbox_p": round(float(lb["lb_pvalue"].iloc[0]), 3),
        "dw": round(dw, 2),
        "acf1": round(acf1, 3),
    },
    "importancias": imp[:12],
    "pred": [round(float(v), 1) for v in pred],
    "pi_lo": [round(float(v), 1) for v in lo],
    "pi_hi": [round(float(v), 1) for v in hi],
    "residuos_serie": [round(float(v), 1) for v in res],
    "fechas_test": [str(d.date()) for d in te.index],
    "y_test": [float(v) for v in te["y"]],
}
(OUT / "resultados_modelo_xgboost.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
xg.save_model(str(OUT / "modelo_xgboost.json"))
print(json.dumps(out, ensure_ascii=False, indent=1))
