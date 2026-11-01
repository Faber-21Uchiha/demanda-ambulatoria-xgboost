# -*- coding: utf-8 -*-
"""Analisis ACA2: ACF/PACF, STL, estabilidad temporal, VIF y OE2."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "03_resultados"
OUT.mkdir(parents=True, exist_ok=True)
FIG = OUT / "figuras"
FIG.mkdir(exist_ok=True)

VERDE, VERDE_OSCURO, AZUL, TEAL, GRIS = "#84BD00", "#007A33", "#0C2340", "#00859B", "#898D8D"
AMARILLO = "#FFCD00"
plt.rcParams.update({
    "figure.dpi": 200, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    "font.size": 9, "axes.edgecolor": "#666666", "font.family": "DejaVu Sans",
})

df = pd.read_csv(BASE / "01_datos" / "dataset_depurado_OE1.csv", encoding="utf-8-sig")
df["fecha_atencion"] = pd.to_datetime(df["fecha_atencion"])
R = {}

# ------------------------------------------------------------------ serie
sem = df.groupby(pd.Grouper(key="fecha_atencion", freq="W-SUN")).size()
sem = sem[sem > 0]
sem = sem[sem.index <= "2025-12-28"].asfreq("W-SUN", fill_value=0)
y = sem.astype(float)
R["n_semanas"] = int(len(y))
R["media_semanal"] = round(float(y.mean()), 1)
R["sd_semanal"] = round(float(y.std(ddof=1)), 1)
R["cv_semanal"] = round(float(y.std(ddof=1) / y.mean() * 100), 1)
R["min_semanal"] = int(y.min())
R["max_semanal"] = int(y.max())
R["fecha_min"] = str(y.index.min().date())
R["fecha_max"] = str(y.index.max().date())

adf = adfuller(y, autolag="AIC")
R["adf_stat"] = round(float(adf[0]), 3)
R["adf_p"] = round(float(adf[1]), 4)
R["adf_lags"] = int(adf[2])

lb = acorr_ljungbox(y, lags=[4, 8, 12, 26], return_df=True)
R["ljungbox"] = {int(k): [round(float(r["lb_stat"]), 2), round(float(r["lb_pvalue"]), 5)]
                 for k, r in lb.iterrows()}

# fuerza estacional (STL diario, periodo 7: ciclo intrasemanal)
diaria = df.groupby("fecha_atencion").size().asfreq("D", fill_value=0).astype(float)
stl = STL(diaria, period=7, robust=True).fit()
var_res = float(np.var(stl.resid))
var_st = float(np.var(stl.seasonal + stl.resid))
var_tr = float(np.var(stl.trend + stl.resid))
R["fuerza_estacional"] = round(max(0.0, 1 - var_res / var_st), 6)
R["fuerza_tendencia"] = round(max(0.0, 1 - var_res / var_tr), 6)
R["var_explicada_stl"] = round(float(1 - var_res / np.var(diaria)), 6)
R["n_dias"] = int(len(diaria))
R["media_diaria"] = round(float(diaria.mean()), 1)
R["dia_semana_media"] = {k: round(float(v), 1) for k, v in
                         diaria.groupby(diaria.index.dayofweek).mean().items()}
adf_d = adfuller(np.diff(y), autolag="AIC")
R["adf_diff_stat"] = round(float(adf_d[0]), 3)
R["adf_diff_p"] = round(float(adf_d[1]), 6)

fig, axes = plt.subplots(3, 1, figsize=(7.4, 5.6), sharex=True,
                         gridspec_kw={"height_ratios": [1.1, 1, 1]})
axes[0].plot(diaria.index, stl.observed, color=TEAL, lw=0.9, label="Observado")
axes[0].plot(diaria.index, stl.trend, color=VERDE_OSCURO, lw=2, label="Tendencia")
axes[0].legend(frameon=False, fontsize=7, loc="upper right", ncol=2)
axes[0].set_ylabel("Registros")
axes[1].plot(diaria.index, stl.seasonal, color=AZUL, lw=0.9)
axes[1].set_ylabel("Componente\nsemanal")
axes[2].plot(diaria.index, stl.resid, color=GRIS, lw=0.7)
axes[2].axhline(0, color="#444444", lw=0.8)
axes[2].set_ylabel("Residuo")
axes[2].set_xlabel("Día de atención")
fig.tight_layout()
fig.savefig(FIG / "fig_stl.png", bbox_inches="tight")
plt.close(fig)

# ACF / PACF
nlags = 35
a = acf(y, nlags=nlags, fft=False)
p = pacf(y, nlags=nlags, method="ywm")
ic = 1.96 / np.sqrt(len(y))
R["acf_lag1"] = round(float(a[1]), 3)
R["acf_lag4"] = round(float(a[4]), 3)
R["acf_lag52"] = round(float(a[52]) if len(a) > 52 else a[-1], 3)
R["pacf_lag1"] = round(float(p[1]), 3)
R["pacf_lag2"] = round(float(p[2]), 3)
R["banda_ic"] = round(float(ic), 3)
R["acf_sig"] = [i for i in range(1, nlags + 1) if abs(a[i]) > ic]
R["pacf_sig"] = [i for i in range(1, nlags + 1) if abs(p[i]) > ic]

fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.2))
for ax, vals, titulo in [(axes[0], a, "Autocorrelación (ACF)"),
                         (axes[1], p, "Autocorrelación parcial (PACF)")]:
    ax.vlines(range(len(vals)), 0, vals, color=TEAL, lw=2.2)
    ax.plot(range(len(vals)), vals, "o", color=AZUL, ms=3)
    ax.axhline(0, color="#444444", lw=0.8)
    ax.axhline(ic, color=GRIS, ls="--", lw=0.9)
    ax.axhline(-ic, color=GRIS, ls="--", lw=0.9)
    ax.set_xlim(-0.5, nlags + 0.5)
    ax.set_title(titulo, fontsize=9, loc="left")
    ax.set_xlabel("Rezago (semanas)")
fig.tight_layout()
fig.savefig(FIG / "fig_acf_pacf.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------- estabilidad 2024 vs 2025
def cramers_v(tab):
    chi2 = stats.chi2_contingency(tab, correction=False)[0]
    n = tab.values.sum()
    r, k = tab.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1))))

def tvd(tab):
    a = tab.sum(axis=1) / tab.sum(axis=1).sum()
    b = tab.sum(axis=0) / tab.sum(axis=0).sum()
    return float(0.5 * np.abs(a - b).sum())

est = {}
for var in ["servicio", "diagnostico_categoria", "localidad_residencia",
            "regimen", "prioridad", "tipo_orden"]:
    tab = pd.crosstab(df[var], df["anio"])
    tab = tab[tab.sum(axis=1) > 0]
    # agrupar categorias raras para validez del chi2
    tab = tab.reindex(tab.sum(axis=1).sort_values(ascending=False).index)
    if len(tab) > 10:
        top = tab.iloc[:10]
        resto = tab.iloc[10:].sum().to_frame().T
        resto.index = ["Otras categorías"]
        tab = pd.concat([top, resto])
    chi2, pv, dof, _ = stats.chi2_contingency(tab, correction=False)
    est[var] = {
        "chi2": round(float(chi2), 2), "gl": int(dof), "p": float(f"{pv:.3g}"),
        "v": round(cramers_v(tab), 3), "tvd": round(tvd(tab), 6),
        "categorias": int(len(tab)),
    }
R["estabilidad"] = est

R["servicio_top_2024"] = df[df.anio == 2024]["servicio"].value_counts(normalize=True).mul(100).round(1).head(3).to_dict()
R["servicio_top_2025"] = df[df.anio == 2025]["servicio"].value_counts(normalize=True).mul(100).round(1).head(3).to_dict()
R["cie_top_2024"] = df[df.anio == 2024]["diagnostico_categoria"].value_counts(normalize=True).mul(100).round(1).head(3).to_dict()
R["cie_top_2025"] = df[df.anio == 2025]["diagnostico_categoria"].value_counts(normalize=True).mul(100).round(1).head(3).to_dict()
R["n_2024"] = int((df.anio == 2024).sum())
R["n_2025"] = int((df.anio == 2025).sum())
R["semanas_2024"] = int((sem.index.year == 2024).sum())
R["semanas_2025"] = int((sem.index.year == 2025).sum())
R["media_2024"] = round(float(df[df.anio == 2024].groupby(pd.Grouper(key="fecha_atencion", freq="W-SUN")).size().mean()), 1)
R["media_2025"] = round(float(df[df.anio == 2025].groupby(pd.Grouper(key="fecha_atencion", freq="W-SUN")).size().mean()), 1)

# figura estabilidad: servicio y CIE, 2024 vs 2025
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
for ax, var, titulo in [(axes[0], "servicio", "Tipo de servicio"),
                        (axes[1], "diagnostico_categoria", "Categoría diagnóstica (top 7)")]:
    tab = pd.crosstab(df[var], df["anio"], normalize="columns").mul(100)
    orden = tab.mean(axis=1).sort_values(ascending=False).index[:7]
    tab = tab.loc[orden]
    yy = np.arange(len(tab))
    ax.barh(yy + 0.19, tab[2024], height=0.36, color=TEAL, label="2024")
    ax.barh(yy - 0.19, tab[2025], height=0.36, color=VERDE, label="2025")
    ax.set_yticks(yy)
    ax.set_yticklabels([s if len(s) <= 34 else s[:32] + "…" for s in tab.index], fontsize=6.6)
    ax.invert_yaxis()
    ax.set_xlabel("Porcentaje de registros")
    ax.set_title(titulo, fontsize=8.5, loc="left")
    ax.grid(axis="y", visible=False)
axes[0].legend(frameon=False, fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "fig_estabilidad.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------- VIF
g = df.groupby(pd.Grouper(key="fecha_atencion", freq="W-SUN"))
w = pd.DataFrame({
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
}).dropna()
w = w[w.index.year.isin([2024, 2025])]
w = w.loc[:, w.std() > 0]

def calc_vif(x):
    xc = add_constant(x, has_constant="add")
    vals = []
    for i in range(1, xc.shape[1]):
        vals.append((xc.columns[i], float(variance_inflation_factor(xc.values, i))))
    return sorted(vals, key=lambda t: -t[1])

vif_full = calc_vif(w)
keep = w.copy()
eliminadas = []
while True:
    vals = calc_vif(keep)
    if vals[0][1] <= 5 or keep.shape[1] <= 2:
        break
    eliminadas.append((vals[0][0], round(vals[0][1], 2)))
    keep = keep.drop(columns=[vals[0][0]])
vif_final = calc_vif(keep)
R["vif_inicial"] = [(k, round(v, 2)) for k, v in vif_full]
R["vif_eliminadas"] = eliminadas
R["vif_final"] = [(k, round(v, 2)) for k, v in vif_final]
R["vif_n_obs"] = int(len(w))
R["vif_vars_final"] = list(keep.columns)

# tabla LaTeX-ish no; guardar csv
pd.DataFrame(vif_full, columns=["variable", "vif"]).to_csv(OUT / "vif_inicial.csv", index=False)
pd.DataFrame(vif_final, columns=["variable", "vif"]).to_csv(OUT / "vif_variables_exogenas.csv", index=False)

# ------------------------------------------------------------------- OE2
df["comb"] = df["servicio"] + " | " + df["diagnostico_categoria"]
servicios = sorted(df["servicio"].unique())
categorias = sorted(df["diagnostico_categoria"].unique())
mat = pd.crosstab(df["id_profesional"], df["comb"])
R["oe2_n_prof"] = int(mat.shape[0])
R["oe2_n_comb"] = int(mat.shape[1])
R["oe2_n_celdas"] = int(mat.size)
R["oe2_densidad"] = round(float((mat > 0).sum().sum() / mat.size * 100), 2)
R["oe2_mediana_comb_prof"] = int(mat.gt(0).sum(axis=1).median())
R["oe2_max_comb_prof"] = int(mat.gt(0).sum(axis=1).max())
R["oe2_min_comb_prof"] = int(mat.gt(0).sum(axis=1).min())
freq = df["comb"].value_counts()
R["oe2_top_comb"] = [(k, int(v), round(100 * v / len(df), 2)) for k, v in freq.head(10).items()]
orden = freq.sort_values(ascending=False)
cum = orden.cumsum() / len(df)
R["oe2_comb_80pct"] = int((cum <= 0.8).sum() + 1)
R["oe2_comb_50pct"] = int((cum <= 0.5).sum() + 1)
R["oe2_n_comb_obs"] = int(freq.shape[0])
R["oe2_comb_teoricas"] = int(len(servicios) * len(categorias))

# asociacion
def asoc(a, b):
    tab = pd.crosstab(df[a], df[b])
    chi2, pv, dof, _ = stats.chi2_contingency(tab, correction=False)
    n = tab.values.sum()
    v = float(np.sqrt(chi2 / (n * (min(tab.shape) - 1))))
    return {"chi2": round(float(chi2), 1), "gl": int(dof), "p": float(f"{pv:.3g}"), "v": round(v, 3)}

R["asoc_servicio_cie"] = asoc("servicio", "diagnostico_categoria")
R["asoc_servicio_orden"] = asoc("servicio", "tipo_orden")
R["asoc_cie_orden"] = asoc("diagnostico_categoria", "tipo_orden")

# reglas de asociacion (A -> B)
def reglas(a, b, min_sup=0.01, topn=8):
    n = len(df)
    sa = df[a].value_counts() / n
    sb = df[b].value_counts() / n
    sab = df.groupby([a, b]).size() / n
    out = []
    for (va, vb), sup in sab.items():
        if sup < min_sup:
            continue
        conf = sup / sa[va]
        lift = conf / sb[vb]
        out.append({"antecedente": f"{va}", "consecuente": f"{vb}",
                    "soporte": round(float(sup) * 100, 2),
                    "confianza": round(float(conf) * 100, 1),
                    "lift": round(float(lift), 2)})
    out = [r for r in out if r["lift"] >= 1.2]
    return sorted(out, key=lambda r: (-r["lift"], -r["soporte"]))[:topn]

R["reglas_servicio_orden"] = reglas("servicio", "tipo_orden")
R["reglas_cie_servicio"] = reglas("diagnostico_categoria", "servicio")

# figura matriz: mapa de calor de las 8 combinaciones mas frecuentes por servicio
piv = pd.crosstab(df["servicio"], df["diagnostico_categoria"])
piv = piv.loc[piv.sum(axis=1).sort_values(ascending=False).index]
cols = piv.sum(axis=0).sort_values(ascending=False).index[:10]
sub = piv.loc[:, cols].div(piv.sum(axis=1), axis=0).mul(100)
fig, ax = plt.subplots(figsize=(7.4, 3.6))
im = ax.imshow(sub.values, cmap="YlGn", aspect="auto")
ax.set_xticks(range(len(sub.columns)))
ax.set_xticklabels([c if len(c) <= 20 else c[:18] + "…" for c in sub.columns],
                   rotation=45, ha="right", fontsize=6)
ax.set_yticks(range(len(sub.index)))
ax.set_yticklabels(sub.index, fontsize=6.5)
for i in range(sub.shape[0]):
    for j in range(sub.shape[1]):
        val = sub.values[i, j]
        if val >= 0.5:
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=5.6,
                    color="white" if val > 12 else "#1a1a1a")
ax.grid(False)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
cb.set_label("% dentro del servicio", fontsize=7)
ax.set_title("Concentración diagnóstica por servicio (% de registros del servicio)", fontsize=8, loc="left")
fig.tight_layout()
fig.savefig(FIG / "fig_matriz_prescriptiva.png", bbox_inches="tight")
plt.close(fig)

R["servicios_especialidad_1a1"] = bool((df.groupby("servicio")["especialidad"].nunique() == 1).all())

# feature engineering: tamano de muestra train/test
R["n_train"] = int(round(len(y) * 0.8))
R["n_test"] = int(len(y) - round(len(y) * 0.8))

(OUT / "resultados_series_y_asociacion.json").write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(R, ensure_ascii=False, indent=1))
