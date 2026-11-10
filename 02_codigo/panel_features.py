# -*- coding: utf-8 -*-
"""
panel_features.py

Ingenieria de variables del modelo de panel zona-semana, compartida por el
OE3 (10_modelo_panel_zonas.py) y el protocolo de operacion del OE4
(12_protocolo_oe4.py). Mantener una sola definicion evita que el modelo
desplegado y el modelo evaluado difieran en sus entradas.

Todas las variables explicativas se construyen con informacion disponible hasta
t-1 (rezagos, medias moviles y calendario), de modo que el modelo pueda usarse
para pronosticar la semana siguiente sin fuga de informacion.
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
DATOS = BASE / "01_datos"

EXO = ["medicina_general", "enfermeria", "odontologia", "pediatria", "apoyo_diagnostico",
       "edad_media", "prop_femenino", "prop_mayor_60", "prop_contributivo",
       "prop_prioritaria", "prop_remision", "espera_media"]

NO_PREDICTORAS = ("y", "demanda", "semana", "zona", "no_atendidas", "tasa_no_atencion",
                  "utilizacion", "capacidad", "naive_1", "naive_52", "media_historica")


def construir_panel():
    """Devuelve (panel completo, matriz de modelado con rezagos)."""
    p = pd.read_csv(DATOS / "panel_zona_semana_capacidad.csv", encoding="utf-8-sig")
    p["semana"] = pd.to_datetime(p["semana"])
    p = p.sort_values(["zona", "semana"]).reset_index(drop=True)

    X = p[["zona", "semana", "demanda", "capacidad", "no_atendidas",
           "tasa_no_atencion", "utilizacion"] + EXO].copy()
    for z, idx in X.groupby("zona").groups.items():
        sub = X.loc[idx]
        for lag in (1, 2, 3, 4):
            X.loc[idx, f"lag{lag}"] = sub["demanda"].shift(lag)
        for w in (4, 8, 13):
            X.loc[idx, f"roll{w}"] = sub["demanda"].shift(1).rolling(w).mean()
        X.loc[idx, "roll_std8"] = sub["demanda"].shift(1).rolling(8).std()
        for c in EXO:
            X.loc[idx, f"{c}_lag1"] = sub[c].shift(1)
        woy = sub["semana"].dt.isocalendar().week.astype(int)
        X.loc[idx, "woy_sin"] = np.sin(2 * np.pi * woy / 52)
        X.loc[idx, "woy_cos"] = np.cos(2 * np.pi * woy / 52)
        X.loc[idx, "t"] = np.arange(len(sub))
        # referentes calculados con informacion hasta t-1
        X.loc[idx, "naive_1"] = sub["demanda"].shift(1)
        X.loc[idx, "naive_52"] = sub["demanda"].shift(52)
        X.loc[idx, "media_historica"] = sub["demanda"].shift(1).expanding().mean()

    dummies = pd.get_dummies(X["zona"], prefix="zona", drop_first=True).astype(int)
    X = pd.concat([X, dummies], axis=1)
    X["y"] = X["demanda"]
    X = X.dropna(subset=[f"lag{i}" for i in (1, 2, 3, 4)] + ["roll13", "roll_std8"]
                 + [f"{c}_lag1" for c in EXO])
    return p, X


def columnas_modelo(X):
    """Predictores del modelo: excluye objetivo, identificadores y referentes."""
    return [c for c in X.columns if c not in NO_PREDICTORAS]


PARAMS_XGB = dict(max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                  min_child_weight=2, reg_lambda=1.0, random_state=2026, n_jobs=2,
                  n_estimators=400)
