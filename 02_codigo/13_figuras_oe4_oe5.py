# -*- coding: utf-8 -*-
"""Figuras del panel por zona (OE3), del protocolo (OE4) y de la simulacion (OE5)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
RES = BASE / "03_resultados"
FIG = RES / "figuras"
FIG.mkdir(parents=True, exist_ok=True)
VERDE, TEAL, AZUL, GRIS, NARANJA = "#84BD00", "#00859B", "#00465A", "#E8EDF0", "#E07B39"
plt.rcParams.update({
    "figure.dpi": 200, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    "font.size": 8, "axes.edgecolor": "#666666",
})

panel = pd.read_csv(RES / "predicciones_panel.csv", encoding="utf-8-sig")
panel["semana"] = pd.to_datetime(panel["semana"])
rpanel = json.loads((RES / "resultados_modelo_panel.json").read_text(encoding="utf-8"))
rproto = json.loads((RES / "resultados_protocolo_oe4.json").read_text(encoding="utf-8"))
roe5 = json.loads((RES / "resultados_simulacion_oe5.json").read_text(encoding="utf-8"))

# ------------------------------------------------- 1. observado vs pronosticado
fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.2), sharex=True)
for ax, (zona, g) in zip(axes.ravel(), panel.groupby("zona")):
    ax.plot(g["semana"], g["y"], color=AZUL, linewidth=1.3, marker="o", ms=2.2, label="Observado")
    ax.plot(g["semana"], g["pred_xgb"], color=VERDE, linewidth=1.6, label="XGBoost")
    ax.plot(g["semana"], g["naive_52"], color=NARANJA, linewidth=1.0, linestyle="--",
            label="Naive estacional")
    m = rpanel["por_zona"][zona]["xgboost"]
    ax.set_title(f"{zona}  ·  MAPE {m['MAPE']:.2f} %", fontsize=8, color=AZUL, loc="left")
    ax.tick_params(axis="x", labelrotation=30, labelsize=6.5)
    ax.tick_params(axis="y", labelsize=6.5)
handles, etiquetas = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, etiquetas, frameon=False, fontsize=6.8, loc="lower center", ncol=3)
for ax in axes[1]:
    ax.set_xlabel("Semana", fontsize=7)
for ax in axes[:, 0]:
    ax.set_ylabel("Citas por semana", fontsize=7)
fig.suptitle("Validación del modelo de panel por zona (18 semanas de prueba)",
             fontsize=9, color=AZUL, x=0.02, ha="left", y=0.99)
fig.tight_layout(rect=(0, 0.045, 1, 0.96))
fig.savefig(FIG / "fig_panel_zonas.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------- 2. error por zona
zonas = list(rpanel["por_zona"].keys())
x = np.arange(len(zonas))
mae_x = [rpanel["por_zona"][z]["xgboost"]["MAE"] for z in zonas]
mae_n = [rpanel["por_zona"][z]["naive_52"]["MAE"] for z in zonas]
map_x = [rpanel["por_zona"][z]["xgboost"]["MAPE"] for z in zonas]
map_n = [rpanel["por_zona"][z]["naive_52"]["MAPE"] for z in zonas]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.6))
a1.bar(x - 0.19, mae_x, 0.36, color=VERDE, label="XGBoost")
a1.bar(x + 0.19, mae_n, 0.36, color=TEAL, label="Naive estacional")
a1.set_xticks(x, zonas, fontsize=7)
a1.set_ylabel("MAE (citas/semana)", fontsize=7)
a1.set_title("Error absoluto medio", fontsize=8, loc="left", color=AZUL)
a1.legend(frameon=False, fontsize=6.6)
a2.bar(x - 0.19, map_x, 0.36, color=VERDE)
a2.bar(x + 0.19, map_n, 0.36, color=TEAL)
a2.set_xticks(x, zonas, fontsize=7)
a2.set_ylabel("MAPE (%)", fontsize=7)
a2.set_title("Error porcentual medio", fontsize=8, loc="left", color=AZUL)
fig.tight_layout()
fig.savefig(FIG / "fig_error_por_zona.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------- 3. PSI
det = pd.DataFrame(rproto["deriva_psi"]["detalle"]).sort_values("psi")
colores = [NARANJA if e == "reentrenar" else "#D9A441" if e == "alerta" else TEAL
           for e in det["estado"]]
fig, ax = plt.subplots(figsize=(6.6, 3.9))
ax.barh(det["variable"], det["psi"], color=colores, height=0.72)
ax.axvline(rproto["deriva_psi"]["umbral_alerta"], color="#7A8791", linestyle="--", linewidth=0.9)
ax.axvline(rproto["deriva_psi"]["umbral_reentrenar"], color=NARANJA, linestyle="--", linewidth=0.9)
ax.text(rproto["deriva_psi"]["umbral_alerta"], -0.9, "alerta 0,10", fontsize=6.2,
        color="#5B6A73", ha="center")
ax.text(rproto["deriva_psi"]["umbral_reentrenar"], -0.9, "reentrenar 0,25", fontsize=6.2,
        color=NARANJA, ha="center")
ax.set_xlabel("PSI (2024 como referencia, 2025 como actual)", fontsize=7)
ax.set_title("Deriva de los predictores entre 2024 y 2025", fontsize=9, color=AZUL, loc="left")
ax.tick_params(axis="y", labelsize=6.5)
fig.tight_layout()
fig.savefig(FIG / "fig_psi.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------- 4. escenarios OE5
esc = [("Base", roe5["base"]), ("Reasignación", roe5["reasignacion"]),
       ("+3 % dirigido", roe5["dirigido"]), ("+3 % proporcional", roe5["proporcional"])]
fig, ax = plt.subplots(figsize=(6.6, 2.9))
vals = [e[1]["pqrs_esperadas"] for e in esc]
lo = [e[1]["pqrs_esperadas"] - e[1]["ic95"][0] for e in esc]
hi = [e[1]["ic95"][1] - e[1]["pqrs_esperadas"] for e in esc]
barras = ax.bar([e[0] for e in esc], vals, color=[TEAL, "#A7B4BC", VERDE, "#B7D46A"],
                width=0.62, yerr=[lo, hi], capsize=3, error_kw={"elinewidth": 0.8, "ecolor": "#4A5A64"})
for b, v, (nombre, datos) in zip(barras, vals, esc):
    etiqueta = f"{v:.1f}" if nombre == "Base" else f"{v:.1f}\n(−{datos['reduccion_pct']:.1f} %)"
    ax.text(b.get_x() + b.get_width() / 2, datos["ic95"][1] + 5, etiqueta, ha="center",
            fontsize=6.8, color="#33414A")
ax.set_ylim(0, max([e[1]["ic95"][1] for e in esc]) * 1.24)
ax.set_ylabel("PQRS esperadas por oportunidad", fontsize=7)
ax.set_title("Escenarios de capacidad en las 18 semanas de prueba (IC 95 %)",
             fontsize=9, color=AZUL, loc="left")
ax.tick_params(axis="x", labelsize=7)
fig.tight_layout()
fig.savefig(FIG / "fig_escenarios_oe5.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------- 5. sensibilidad OE5
pres = roe5["sensibilidad"]["presupuesto"]
pct = sorted(float(k) for k in pres)
red = [pres[str(p)]["reduccion_pct"] for p in pct]
cup = [pres[str(p)]["cupos_extra"] for p in pct]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.7))
a1.plot([p * 100 for p in pct], red, color=VERDE, marker="o", linewidth=1.8)
for i, (p, r, c) in enumerate(zip(pct, red, cup)):
    a1.annotate(f"{r:.1f} %\n({c} cupos)", (p * 100, r), textcoords="offset points",
                xytext=(10 if i == 0 else 0, -26 if i == 0 else 9),
                ha="left" if i == 0 else "center",
                fontsize=6.2, color="#33414A")
a1.set_xlabel("Presupuesto adicional (% de la capacidad semanal)", fontsize=7)
a1.set_ylabel("Reducción de PQRS (%)", fontsize=7)
a1.set_title("Tamaño del presupuesto", fontsize=8, loc="left", color=AZUL)
a1.set_ylim(0, max(red) * 1.32)

err = roe5["sensibilidad"]["error_pronostico"]
ks = ["0.5x", "1x", "1.5x", "2x"]
red_e = [err[k]["reduccion_pct"] for k in ks]
ora = roe5["oraculo"]["dirigido_reduccion_pct"]
a2.plot(range(len(ks)), red_e, color=TEAL, marker="s", linewidth=1.8)
a2.axhline(ora, color=NARANJA, linestyle="--", linewidth=1.0)
a2.text(0, ora + 0.8, f"Cota con demanda real ({ora:.1f} %)", fontsize=6.2, color=NARANJA)
a2.text(0.05, min(red_e) - 3.2, "Dirigido con pronóstico", fontsize=6.2,
        color=TEAL, ha="left")
a2.set_xticks(range(len(ks)), ks, fontsize=7)
a2.set_xlabel("Error de pronóstico (múltiplo del residual)", fontsize=7)
a2.set_ylabel("Reducción de PQRS (%)", fontsize=7)
a2.set_title("Robustez al error de pronóstico", fontsize=8, loc="left", color=AZUL)
a2.set_ylim(min(red_e) - 6, ora + 5)
fig.tight_layout()
fig.savefig(FIG / "fig_sensibilidad_oe5.png", bbox_inches="tight")
plt.close(fig)

print("figuras generadas:", [p.name for p in sorted(FIG.glob("fig_*.png"))])
