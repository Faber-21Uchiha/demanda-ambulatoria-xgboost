# -*- coding: utf-8 -*-
"""Figuras de la Fase 3: arquitectura, importancia de variables y residuos."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "03_resultados"
FIG = OUT / "figuras"
FIG.mkdir(parents=True, exist_ok=True)
VERDE, TEAL, AZUL, GRIS = "#84BD00", "#00859B", "#00465A", "#E8EDF0"
plt.rcParams.update({
    "figure.dpi": 200, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    "font.size": 8, "axes.edgecolor": "#666666",
})

# ------------------------------------------------------------------ arquitectura
fig, ax = plt.subplots(figsize=(6.6, 4.9))
ax.set_xlim(0, 100)
ax.set_ylim(8, 100)
ax.axis("off")

capas = [
    ("Capa de datos", GRIS, 76, ["Fuentes\ninstitucionales\ny capa sintética",
                                 "Almacenamiento\nCSV UTF-8 +\ndiccionario",
                                 "Evaluación\nde calidad\n(15 indicadores)"]),
    ("Capa de\npreparación", GRIS, 55, ["Depuración\ntrazable\n(6 criterios)",
                                        "Agregación\nsemanal\n(104 semanas)",
                                        "Ingeniería de\ncaracterísticas\n(23 variables)"]),
    ("Capa de\nmodelado", GRIS, 34, ["Partición\ntemporal 80/20\n(73 / 18)",
                                     "Validación\ncruzada temporal\n(4 pliegues)",
                                     "Entrenamiento\nXGBoost +\nartefacto"]),
    ("Capa de\noperación", GRIS, 13, ["Predicción\nsemanal\nagregada",
                                      "Monitoreo de\nderiva y\ndesempeño",
                                      "Reentrenamiento\ntrimestral"]),
]
colores = [TEAL, TEAL, VERDE, AZUL]
for (nombre, fondo, y, cajas), col in zip(capas, colores):
    ax.add_patch(FancyBboxPatch((1, y - 1), 98, 17, boxstyle="round,pad=0.6,rounding_size=1.2",
                                facecolor=fondo, edgecolor="#C9D2D8", linewidth=0.8))
    ax.text(4.5, y + 7.5, nombre, fontsize=7.5, fontweight="bold", va="center",
            ha="left", color="#33414A")
    xs = [20, 47.5, 75]
    for x, txt in zip(xs, cajas):
        ax.add_patch(FancyBboxPatch((x, y + 2.2), 21.5, 11, boxstyle="round,pad=0.35,rounding_size=0.8",
                                    facecolor=col, edgecolor="none"))
        ax.text(x + 10.75, y + 7.7, txt, fontsize=6.9, color="white", ha="center",
                va="center", linespacing=1.25)
    for x in (41.7, 69.2):
        ax.add_patch(FancyArrowPatch((x, y + 7.7), (x + 5.4, y + 7.7),
                                     arrowstyle="-|>", mutation_scale=8,
                                     color="#5B6A73", linewidth=1.0))
for i in range(3):
    y0 = capas[i][2]
    y1 = capas[i + 1][2]
    mid = (y0 - 1 + y1 + 18) / 2
    ax.plot([85.75, 85.75, 30.75, 30.75], [y0 + 2.2, mid, mid, y1 + 13.2],
            color="#5B6A73", linewidth=1.0, solid_capstyle="round")
    ax.add_patch(FancyArrowPatch((30.75, y1 + 13.9), (30.75, y1 + 13.1),
                                 arrowstyle="-|>", mutation_scale=9,
                                 color="#5B6A73", linewidth=1.0))
ax.text(99, 97, "Flujo de datos y artefactos", fontsize=7, ha="right", color="#5B6A73",
        style="italic")
fig.tight_layout(pad=0.2)
fig.savefig(FIG / "fig_arquitectura.png", bbox_inches="tight")

# ------------------------------------------------------------------ importancia
f3 = json.loads((OUT / "resultados_modelo_xgboost.json").read_text(encoding="utf-8"))
imp = f3["importancias"][:10][::-1]
nombres = {
    "roll4": "Media móvil 4 semanas", "roll13": "Media móvil 13 semanas",
    "roll_std8": "Desviación móvil 8 semanas", "lag1": "Rezago 1 semana",
    "lag2": "Rezago 2 semanas", "lag3": "Rezago 3 semanas", "lag4": "Rezago 4 semanas",
    "woy_sin": "Estacionalidad (seno)", "woy_cos": "Estacionalidad (coseno)",
    "pediatria_lag1": "Participación pediatría (t-1)",
    "espera_media_lag1": "Espera media (t-1)",
    "apoyo_diagnostico_lag1": "Apoyo diagnóstico (t-1)",
    "edad_media_lag1": "Edad media (t-1)",
    "prop_femenino_lag1": "Proporción femenina (t-1)",
    "prop_mayor_60_lag1": "Mayores de 60 (t-1)",
}
etiquetas = [nombres.get(k, k) for k, _ in imp]
valores = [v for _, v in imp]
fig, ax = plt.subplots(figsize=(6.4, 3.3))
colores_b = [VERDE if i >= len(valores) - 4 else TEAL for i in range(len(valores))]
ax.barh(range(len(valores)), valores, color=colores_b, height=0.68)
ax.set_yticks(range(len(valores)))
ax.set_yticklabels(etiquetas, fontsize=7.4)
ax.set_xlabel("Importancia relativa por ganancia (%)", fontsize=7.6)
ax.grid(axis="y", visible=False)
for i, v in enumerate(valores):
    ax.text(v + 0.25, i, f"{v:.1f}".replace(".", ","), va="center", fontsize=6.9,
            color="#33414A")
ax.set_xlim(0, max(valores) * 1.18)
ax.tick_params(axis="x", labelsize=7)
fig.tight_layout()
fig.savefig(FIG / "fig_importancia.png", bbox_inches="tight")

# ------------------------------------------------------------------ residuos
fechas = np.arange(len(f3["y_test"]))
y = np.array(f3["y_test"])
pred = np.array(f3["pred"])
res = np.array(f3["residuos_serie"])
lo = np.array(f3["pi_lo"])
hi = np.array(f3["pi_hi"])
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9),
                         gridspec_kw={"width_ratios": [1.15, 1]})
ax = axes[0]
ax.axhline(0, color="#33414A", linewidth=0.9)
ax.bar(fechas, res, color=[TEAL if r >= 0 else "#C0504D" for r in res], width=0.62)
ax.set_xticks(fechas[::3])
ax.set_xticklabels([f3["fechas_test"][i][5:] for i in range(0, len(fechas), 3)], fontsize=6.6)
ax.set_ylabel("Residuo (registros)", fontsize=7.2)
ax.set_title("a) Residuos del conjunto de prueba", fontsize=7.6, loc="left")
ax.tick_params(axis="y", labelsize=6.8)
ax.grid(axis="x", visible=False)

ax = axes[1]
ax.fill_between(fechas, lo, hi, color=TEAL, alpha=0.16, label="Intervalo 90 %")
ax.plot(fechas, y, color="#33414A", linewidth=1.1, marker="o", markersize=2.2, label="Observado")
ax.plot(fechas, pred, color=VERDE, linewidth=1.4, label="XGBoost")
ax.set_xticks(fechas[::3])
ax.set_xticklabels([f3["fechas_test"][i][5:] for i in range(0, len(fechas), 3)], fontsize=6.6)
ax.set_ylabel("Registros por semana", fontsize=7.2)
ax.set_title("b) Cobertura del intervalo del 90 %", fontsize=7.6, loc="left")
ax.tick_params(axis="y", labelsize=6.8)
ax.legend(frameon=False, fontsize=6.4, loc="lower left", ncol=1, handlelength=1.4)
ax.grid(axis="x", visible=False)
fig.tight_layout(w_pad=1.4)
fig.savefig(FIG / "fig_residuos.png", bbox_inches="tight")
print("ok")
