# -*- coding: utf-8 -*-
"""Figura de estabilidad 2024-2025 en dos paneles apilados."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "03_resultados"
VERDE, TEAL = "#84BD00", "#00859B"
plt.rcParams.update({
    "figure.dpi": 200, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    "font.size": 8, "axes.edgecolor": "#666666",
})

df = pd.read_csv(BASE / "01_datos" / "dataset_depurado_OE1.csv", encoding="utf-8-sig")
fig, axes = plt.subplots(2, 1, figsize=(5.9, 4.5),
                         gridspec_kw={"height_ratios": [11, 7]})
for ax, var, titulo, topn in [(axes[0], "servicio", "Tipo de servicio", 11),
                              (axes[1], "diagnostico_categoria", "Categoría diagnóstica (siete más frecuentes)", 7)]:
    tab = pd.crosstab(df[var], df["anio"], normalize="columns").mul(100)
    orden = tab.mean(axis=1).sort_values(ascending=False).index[:topn]
    tab = tab.loc[orden]
    yy = np.arange(len(tab))
    ax.barh(yy + 0.19, tab[2024], height=0.36, color=TEAL, label="2024")
    ax.barh(yy - 0.19, tab[2025], height=0.36, color=VERDE, label="2025")
    ax.set_yticks(yy)
    etiquetas = []
    for s in tab.index:
        if len(s) > 34:
            s = s[:33].rsplit(" ", 1)[0] + "…"
        etiquetas.append(s)
    ax.set_yticklabels(etiquetas, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("% de registros", fontsize=7)
    ax.set_title(titulo, fontsize=8, loc="left")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="x", labelsize=7)
axes[0].legend(frameon=False, fontsize=7, loc="lower right")
fig.tight_layout(h_pad=1.0)
fig.savefig(FIG / "fig_estabilidad.png", bbox_inches="tight")
print("ok")
