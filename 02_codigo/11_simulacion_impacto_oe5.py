# -*- coding: utf-8 -*-
"""
11_simulacion_impacto_oe5.py

Objetivo Especifico 5: estimar el impacto de usar el pronostico del modelo en la
planificacion de capacidad sobre las PQRS por falta de oportunidad.

Relacion PQRS ~ citas no atendidas
----------------------------------
Se estima por minimos cuadrados ordinarios con errores robustos HC1 sobre las 104
semanas del panel y efectos fijos de zona:

    pqrs_op(z,t) = a + b * no_atendidas(z,t) + efectos_zona + e(z,t)

La forma lineal es la que corresponde al generador de la capa sintetica (cada
cita no atendida genera en esperanza 0,28 PQRS) y evita el problema del GLM
Poisson con enlace identidad, cuya verosimilitud colapsa cuando la media lineal
se aproxima a cero. El coeficiente estimado se contrasta con el parametro del
generador.

Escenarios (18 semanas de prueba, misma ventana que el OE3)
-----------------------------------------------------------
  base          capacidad historica, sin intervencion.
  reasignacion  mismo total de cupos, redistribuido entre zonas segun el
                pronostico, con un limite operativo del 10 % por zona y semana.
  dirigido      se agrega un presupuesto de cupos extraordinarios equivalente al
                3 % de la capacidad semanal, asignado segun el deficit pronosticado.
  proporcional  el mismo presupuesto del 3 % repartido segun la capacidad
                instalada (regla reactiva de referencia).

La incertidumbre de las PQRS esperadas se obtiene remuestreando la distribucion
asintotica de los coeficientes estimados.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[1]
DATOS = BASE / "01_datos"
RES = BASE / "03_resultados"
SEED = 2026
N_TEST = 18
PRESUPUESTO = 0.03
LIMITE_REASIGNACION = 0.10


def cargar():
    panel = pd.read_csv(DATOS / "panel_zona_semana_capacidad.csv", encoding="utf-8-sig")
    panel["semana"] = pd.to_datetime(panel["semana"])
    pqrs = pd.read_csv(DATOS / "pqrs_sinteticas_2024_2025.csv", encoding="utf-8-sig")
    pqrs["semana_hecho"] = pd.to_datetime(pqrs["semana_hecho"])
    pred = pd.read_csv(RES / "predicciones_panel.csv", encoding="utf-8-sig")
    pred["semana"] = pd.to_datetime(pred["semana"])
    return panel, pqrs, pred


def diseno(d):
    X = pd.get_dummies(d[["zona"]], columns=["zona"], drop_first=True, dtype=float)
    X["no_atendidas"] = d["no_atendidas"].values
    X = sm.add_constant(X)
    return X


def ajustar_ols(base):
    X = diseno(base)
    modelo = sm.OLS(base["pqrs_op"], X).fit(cov_type="HC1")
    return modelo, list(modelo.params.index)


def agregar_capacidad(deficit, cap_base, presupuesto, modo):
    total = cap_base.sum()
    extra = presupuesto * total
    d = np.clip(deficit, 0, None)
    if modo == "proporcional" or d.sum() <= 1e-9:
        return cap_base + extra * cap_base / total
    return cap_base + extra * d / d.sum()


def reasignar_existente(pron, cap_base, limite):
    """Redistribuye el total semanal de cupos segun el pronostico, con tope por zona."""
    total = cap_base.sum()
    objetivo = total * pron / pron.sum()
    lo, hi = cap_base * (1 - limite), cap_base * (1 + limite)
    cap = np.clip(objetivo, lo, hi)
    for _ in range(60):
        dif = total - cap.sum()
        if abs(dif) < 1e-6:
            break
        ajustables = (cap > lo + 1e-9) if dif < 0 else (cap < hi - 1e-9)
        if ajustables.sum() == 0:
            break
        cap[ajustables] = np.clip(cap[ajustables] + dif / ajustables.sum(),
                                  lo[ajustables], hi[ajustables])
    return cap


def simular(test, pred, presupuesto, modo, sigma_error=0.0, rng=None,
            reasignar=False, limite=LIMITE_REASIGNACION):
    filas = []
    for semana, g in test.groupby("semana"):
        g = g.sort_values("zona")
        f = pred[pred["semana"] == semana].set_index("zona").reindex(g["zona"])["pred_xgb"].values
        if sigma_error > 0:
            f = f + rng.normal(0, sigma_error, len(f))
        f = np.clip(f, 1, None)
        cap = g["capacidad"].values.astype(float)
        if reasignar:
            nueva = reasignar_existente(f, cap, limite)
        else:
            nueva = agregar_capacidad(f - cap, cap, presupuesto, modo)
        dem = g["demanda"].values.astype(float)
        for z, d_, c_, u_ in zip(g["zona"], dem, nueva, np.clip(dem - nueva, 0, None)):
            filas.append({"semana": semana, "zona": z, "demanda": d_,
                          "capacidad": c_, "no_atendidas": u_})
    return pd.DataFrame(filas)


def simular_oraculo(test, modo):
    """Cota superior: asigna con la demanda real en lugar del pronostico."""
    filas = []
    for semana, g in test.groupby("semana"):
        g = g.sort_values("zona")
        d = g["demanda"].values.astype(float)
        c = g["capacidad"].values.astype(float)
        if modo == "reasignacion":
            nueva = c.sum() * d / d.sum()
        else:
            extra = PRESUPUESTO * c.sum()
            defi = np.clip(d - c, 0, None)
            nueva = c + extra * defi / defi.sum() if defi.sum() > 0 else c + extra * c / c.sum()
        for z, d_, c_, u_ in zip(g["zona"], d, nueva, np.clip(d - nueva, 0, None)):
            filas.append({"semana": semana, "zona": z, "demanda": d_,
                          "capacidad": c_, "no_atendidas": u_})
    return pd.DataFrame(filas)


def error_participacion(test, pred):
    """Error L1 medio entre la participacion por zona y la demanda real."""
    err_f, err_c = [], []
    for _, g in test.groupby("semana"):
        g = g.sort_values("zona")
        f = pred[pred["semana"] == g["semana"].iloc[0]].set_index("zona").reindex(g["zona"])["pred_xgb"].values
        d = g["demanda"].values.astype(float)
        c = g["capacidad"].values.astype(float)
        err_f.append(float(np.abs(f / f.sum() - d / d.sum()).sum()))
        err_c.append(float(np.abs(c / c.sum() - d / d.sum()).sum()))
    return float(np.mean(err_f)), float(np.mean(err_c))


def esperadas(sim, beta, cols):
    X = diseno(sim).reindex(columns=cols, fill_value=0.0)
    return float(np.clip(X.values @ beta, 0, None).sum())


def main():
    rng = np.random.default_rng(SEED)
    panel, pqrs, pred = cargar()
    q = (pqrs[pqrs["vinculada_falta_oportunidad"]]
         .groupby(["zona", "semana_hecho"]).size().rename("pqrs_op").reset_index()
         .rename(columns={"semana_hecho": "semana"}))
    base = panel.merge(q, on=["zona", "semana"], how="left").fillna({"pqrs_op": 0})
    base["pqrs_op"] = base["pqrs_op"].astype(int)

    modelo, cols = ajustar_ols(base)
    beta = modelo.params.values
    cov = modelo.cov_params().values + np.eye(len(beta)) * 1e-12
    coef = float(modelo.params["no_atendidas"])
    ic = modelo.conf_int().loc["no_atendidas"].tolist()

    test = base[base["semana"] >= base["semana"].max() - pd.Timedelta(weeks=N_TEST - 1)].copy()
    base_sim = test[["semana", "zona", "demanda", "capacidad", "no_atendidas"]].copy()
    rea_sim = simular(test, pred, 0.0, "dirigido", rng=rng, reasignar=True)
    dir_sim = simular(test, pred, PRESUPUESTO, "dirigido", rng=rng)
    prop_sim = simular(test, pred, PRESUPUESTO, "proporcional")

    resid_sd = float((pred["y"] - pred["pred_xgb"]).std())

    def ic_esperado(sim, n=2000):
        e = esperadas(sim, beta, cols)
        draws = np.array([esperadas(sim, rng.multivariate_normal(beta, cov), cols)
                          for _ in range(n)])
        return e, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))

    e_base, lo_b, hi_b = ic_esperado(base_sim)
    e_rea, lo_r, hi_r = ic_esperado(rea_sim)
    e_dir, lo_d, hi_d = ic_esperado(dir_sim)
    e_pro, lo_p, hi_p = ic_esperado(prop_sim)

    # cota superior con informacion perfecta
    ora_rea_sim = simular_oraculo(test, "reasignacion")
    ora_dir_sim = simular_oraculo(test, "dirigido")
    e_ora_rea, _, _ = ic_esperado(ora_rea_sim, n=600)
    e_ora_dir, _, _ = ic_esperado(ora_dir_sim, n=600)
    err_pron, err_cap = error_participacion(test, pred)

    sens = {"reasignacion": {}, "presupuesto": {}, "error_pronostico": {}}
    for lim in (0.10, 0.25, 0.50, 1.00):
        s = simular(test, pred, 0.0, "dirigido", rng=rng, reasignar=True, limite=lim)
        e, _, _ = ic_esperado(s, n=600)
        sens["reasignacion"][str(lim)] = {
            "no_atendidas": round(float(s["no_atendidas"].sum()), 0),
            "pqrs_esperadas": round(e, 1),
            "reduccion_pct": round(float(100 * (1 - e / e_base)), 2),
        }
    for pct in (0.01, 0.02, 0.03, 0.05):
        s = simular(test, pred, pct, "dirigido", rng=rng)
        e, _, _ = ic_esperado(s, n=600)
        sens["presupuesto"][str(pct)] = {
            "cupos_extra": int(round(pct * base_sim["capacidad"].sum())),
            "no_atendidas": round(float(s["no_atendidas"].sum()), 0),
            "pqrs_esperadas": round(e, 1),
            "reduccion_pct": round(float(100 * (1 - e / e_base)), 2),
        }
    for k, mult in (("0.5x", 0.5), ("1x", 1.0), ("1.5x", 1.5), ("2x", 2.0)):
        s = simular(test, pred, PRESUPUESTO, "dirigido", sigma_error=resid_sd * mult, rng=rng)
        e, _, _ = ic_esperado(s, n=600)
        sens["error_pronostico"][k] = {
            "sigma": round(resid_sd * mult, 2),
            "no_atendidas": round(float(s["no_atendidas"].sum()), 0),
            "pqrs_esperadas": round(e, 1),
            "reduccion_pct": round(float(100 * (1 - e / e_base)), 2),
        }

    res = {
        "ventana": {"semanas": int(test["semana"].nunique()),
                    "ini": str(test["semana"].min().date()),
                    "fin": str(test["semana"].max().date()),
                    "observaciones": int(len(test))},
        "modelo_pqrs": {"familia": "OLS", "errores": "HC1",
                        "forma": "pqrs_op = a + b*no_atendidas + efectos_zona",
                        "coef_no_atendidas": round(coef, 4),
                        "ic95": [round(float(ic[0]), 4), round(float(ic[1]), 4)],
                        "r2": round(float(modelo.rsquared), 3),
                        "n": int(modelo.nobs),
                        "coef_generador": 0.28,
                        "pqrs_observadas_ventana": int(test["pqrs_op"].sum())},
        "base": {"no_atendidas": round(float(base_sim["no_atendidas"].sum()), 0),
                 "pqrs_esperadas": round(e_base, 1),
                 "ic95": [round(lo_b, 1), round(hi_b, 1)]},
        "reasignacion": {"limite": LIMITE_REASIGNACION,
                         "no_atendidas": round(float(rea_sim["no_atendidas"].sum()), 0),
                         "pqrs_esperadas": round(e_rea, 1),
                         "ic95": [round(lo_r, 1), round(hi_r, 1)],
                         "reduccion_pct": round(float(100 * (1 - e_rea / e_base)), 2)},
        "dirigido": {"presupuesto": PRESUPUESTO,
                     "no_atendidas": round(float(dir_sim["no_atendidas"].sum()), 0),
                     "pqrs_esperadas": round(e_dir, 1),
                     "ic95": [round(lo_d, 1), round(hi_d, 1)],
                     "reduccion_pct": round(float(100 * (1 - e_dir / e_base)), 2)},
        "proporcional": {"presupuesto": PRESUPUESTO,
                         "no_atendidas": round(float(prop_sim["no_atendidas"].sum()), 0),
                         "pqrs_esperadas": round(e_pro, 1),
                         "ic95": [round(lo_p, 1), round(hi_p, 1)],
                         "reduccion_pct": round(float(100 * (1 - e_pro / e_base)), 2)},
        "comparacion": {"diferencia_pp": round(float(100 * (e_pro - e_dir) / e_base), 2),
                        "pqrs_evitadas_extra": round(float(e_pro - e_dir), 1)},
        "oraculo": {"reasignacion_pqrs": round(e_ora_rea, 1),
                    "reasignacion_reduccion_pct": round(float(100 * (1 - e_ora_rea / e_base)), 2),
                    "dirigido_pqrs": round(e_ora_dir, 1),
                    "dirigido_reduccion_pct": round(float(100 * (1 - e_ora_dir / e_base)), 2)},
        "error_participacion_l1": {"pronostico": round(err_pron, 4),
                                   "capacidad_actual": round(err_cap, 4)},
        "sensibilidad": sens,
    }
    (RES / "resultados_simulacion_oe5.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
