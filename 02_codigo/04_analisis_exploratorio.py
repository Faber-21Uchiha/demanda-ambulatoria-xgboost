# -*- coding: utf-8 -*-
"""
04_analisis_exploratorio.py
Analisis exploratorio (EDA) del conjunto de datos depurado del OE1 y generacion
de las figuras y tablas que integran la evidencia del ACA 1:
  - graficas/fig_serie_semanal.png   (variable objetivo futura: demanda semanal)
  - graficas/fig_serie_mensual.png
  - graficas/fig_localidad.png
  - graficas/fig_servicio.png
  - graficas/fig_cie10.png
  - graficas/fig_edad_sexo.png
  - graficas/fig_funnel_depuracion.png
  - graficas/fig_pipeline.png
  - tablas/tabla_estadisticas_descriptivas.csv
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
R2 = BASE / "03_resultados"
GRAF = R2 / "graficas"
TABLAS = R2 / "tablas"

VERDE = "#84BD00"
VERDE_OSCURO = "#007A33"
AZUL = "#0C2340"
TEAL = "#00859B"
GRIS = "#898D8D"
AMARILLO = "#FFCD00"
plt.rcParams.update({
    "figure.dpi": 200, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    "font.size": 9, "axes.edgecolor": "#666666",
})

df = pd.read_csv(BASE / "01_datos" / "dataset_depurado_OE1.csv", encoding="utf-8-sig")
df["fecha_atencion"] = pd.to_datetime(df["fecha_atencion"])
m = json.loads((R2 / "metricas.json").read_text(encoding="utf-8"))

# ---------------------------------------------------------------------------
# Figura: serie semanal de demanda (variable objetivo del modelo futuro)
# ---------------------------------------------------------------------------
semana = df.groupby(pd.Grouper(key="fecha_atencion", freq="W-SUN")).size()
semana = semana[semana > 0]
fig, ax = plt.subplots(figsize=(7.5, 3.2))
ax.plot(semana.index, semana.values, color=TEAL, lw=1.4, marker="o", ms=2.5, mfc=TEAL)
media_movil = semana.rolling(8, center=True).mean()
ax.plot(semana.index, media_movil.values, color=VERDE_OSCURO, lw=2.2, label="Media móvil (8 semanas)")
ax.set_ylabel("Registros por semana")
ax.set_xlabel("Semana de atención")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(GRAF / "fig_serie_semanal.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figura: serie mensual
# ---------------------------------------------------------------------------
mensual = df.groupby(df["fecha_atencion"].dt.to_period("M")).size()
fig, ax = plt.subplots(figsize=(7.5, 3.0))
ax.plot(mensual.index.astype(str), mensual.values, color=AZUL, lw=2, marker="o", ms=4)
ax.set_xticks(range(0, len(mensual), 2))
ax.set_xticklabels([str(p) for p in mensual.index[::2]], rotation=45, ha="right", fontsize=7)
ax.set_ylabel("Registros por mes")
ax.set_xlabel("Mes")
fig.tight_layout()
fig.savefig(GRAF / "fig_serie_mensual.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figura: distribucion por localidad
# ---------------------------------------------------------------------------
loc = df["localidad_residencia"].value_counts()
fig, ax = plt.subplots(figsize=(6.8, 2.8))
bars = ax.bar(loc.index, loc.values, color=[VERDE_OSCURO, VERDE, TEAL, GRIS], width=0.58)
for b, v in zip(bars, loc.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 300, f"{v:,}\n({v/len(df)*100:.1f}%)",
            ha="center", fontsize=8)
ax.set_ylabel("Registros")
ax.set_ylim(0, loc.max() * 1.18)
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(GRAF / "fig_localidad.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figura: distribucion por servicio
# ---------------------------------------------------------------------------
serv = df["servicio"].value_counts().sort_values()
fig, ax = plt.subplots(figsize=(7.0, 3.8))
ax.barh(serv.index, serv.values, color=TEAL, height=0.62)
for i, v in enumerate(serv.values):
    ax.text(v + 250, i, f"{v:,} ({v/len(df)*100:.1f}%)", va="center", fontsize=7.5)
ax.set_xlabel("Registros")
ax.set_xlim(0, serv.max() * 1.22)
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(GRAF / "fig_servicio.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figura: top 10 codigos CIE-10
# ---------------------------------------------------------------------------
cie = df["codigo_cie10"].value_counts().head(10).sort_values()
catalogo = pd.read_excel(BASE / "04_documentacion" / "diccionario_y_reglas_OE1.xlsx",
                         sheet_name="Catalogo_CIE10")
mapa_diag = dict(zip(catalogo["codigo_cie10"], catalogo["diagnostico_categoria"]))
etiquetas = [f"{c} — {mapa_diag.get(c, '')[:44]}" for c in cie.index]
fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.barh(etiquetas, cie.values, color=VERDE, height=0.62)
for i, v in enumerate(cie.values):
    ax.text(v + 120, i, f"{v:,}", va="center", fontsize=7.5)
ax.set_xlabel("Registros")
ax.set_xlim(0, cie.max() * 1.14)
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(GRAF / "fig_cie10.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figura: distribucion de edad por sexo
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.0, 3.0))
bins = np.arange(0, 101, 5)
for s, color in [("F", TEAL), ("M", AMARILLO)]:
    sub = df.loc[df["sexo"] == s, "edad"]
    ax.hist(sub, bins=bins, alpha=0.65, color=color, label=f"Sexo {s} (n={len(sub):,})", edgecolor="white", lw=0.4)
ax.set_xlabel("Edad (años)")
ax.set_ylabel("Registros")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(GRAF / "fig_edad_sexo.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figura: embudo de depuracion (trazabilidad visual)
# ---------------------------------------------------------------------------
pasos = [("Registros iniciales", 100000, AZUL)]
acum = 100000
for t in m["limpieza"]["trazabilidad"]:
    pasos.append((f"{t['paso']}: −{t['registros_excluidos']:,}", t["registros_restantes"], GRIS))
pasos.append(("Conjunto depurado", m["limpieza"]["n_final"], VERDE_OSCURO))

fig, ax = plt.subplots(figsize=(7.2, 4.2))
y = np.arange(len(pasos))[::-1]
maxv = pasos[0][1]
for (nombre, v, color), yi in zip(pasos, y):
    ax.barh(yi, v, color=color, height=0.6, alpha=0.92)
    ax.text(v + 1200, yi, f"{v:,}", va="center", fontsize=8, color="#333333")
    ax.text(1300, yi, nombre, va="center", fontsize=8, color="white", fontweight="bold")
ax.set_yticks([])
ax.set_xlim(0, maxv * 1.12)
ax.set_xlabel("Registros")
ax.grid(visible=False)
ax.spines["left"].set_visible(False)
fig.tight_layout()
fig.savefig(GRAF / "fig_funnel_depuracion.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figura: diagrama del proceso tecnico (pipeline OE1)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.6, 4.4))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6.4)
ax.axis("off")

def caja(x, y, texto, color, ancho=2.15, alto=0.95, fs=7.6):
    b = FancyBboxPatch((x, y), ancho, alto, boxstyle="round,pad=0.08",
                       fc=color, ec="none")
    ax.add_patch(b)
    ax.text(x + ancho / 2, y + alto / 2, texto, ha="center", va="center",
            fontsize=fs, color="white", fontweight="bold")

def flecha(x1, y1, x2, y2):
    ax.add_patch(FlechaSimple(x1, y1, x2, y2))

class FlechaSimple(FancyArrowPatch):
    pass

def flecha(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=AZUL, lw=1.6,
                                shrinkA=2, shrinkB=2))

fila1 = ["1. Inventario y\ncaracterización\nde fuentes", "2. Carga y revisión\nde estructura\ny tipos de datos",
         "3. Perfilamiento:\nnulos, únicos,\ndistribuciones", "4. Detección de\nduplicados"]
fila2 = ["5. Validación CIE-10\ncontra catálogo", "6. Validación de\nfechas y campos\nderivados",
         "7. Homologación de\ncategorías de\nservicio", "8. Evaluación de\nrepresentatividad"]
fila3 = ["9. Aplicación de\ncriterios C1-C5\n(inclusión/exclusión)", "10. Tratamientos\nT1-T3 y recálculo\nde campos derivados",
         "11. Conjunto de\ndatos depurado\n(87.832 registros)", "12. Documentación\ny EDA inicial\n(este informe)"]

for i, t in enumerate(fila1):
    caja(0.25 + i * 2.45, 4.9, t, TEAL)
for i, t in enumerate(fila2):
    caja(0.25 + i * 2.45, 2.85, t, VERDE_OSCURO)
for i, t in enumerate(fila3):
    caja(0.25 + i * 2.45, 0.8, t, AZUL)

for i in range(3):
    flecha(0.25 + i * 2.45 + 2.18, 5.375, 0.25 + (i + 1) * 2.45, 5.375)
    flecha(0.25 + i * 2.45 + 2.18, 3.325, 0.25 + (i + 1) * 2.45, 3.325)
    flecha(0.25 + i * 2.45 + 2.18, 1.275, 0.25 + (i + 1) * 2.45, 1.275)
flecha(9.45, 4.9, 9.45, 3.9)   # baja de fila 1 a fila 2 (derecha)
flecha(0.32, 3.8, 0.32, 1.85)  # baja de fila 2 a fila 3 (izquierda)
ax.text(5.0, 6.15, "", ha="center")
fig.tight_layout()
fig.savefig(GRAF / "fig_pipeline.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Tabla: estadisticas descriptivas del conjunto depurado
# ---------------------------------------------------------------------------
stats = pd.DataFrame({
    "variable": ["edad", "tiempo_espera_min", "cantidad_ordenes", "cantidad_servicios"],
    "media": [round(df["edad"].mean(), 1), round(df["tiempo_espera_min"].mean(), 1),
              round(df["cantidad_ordenes"].mean(), 2), round(df["cantidad_servicios"].mean(), 2)],
    "desviacion_estandar": [round(df["edad"].std(), 1), round(df["tiempo_espera_min"].std(), 1),
                            round(df["cantidad_ordenes"].std(), 2), round(df["cantidad_servicios"].std(), 2)],
    "minimo": [df["edad"].min(), df["tiempo_espera_min"].min(),
               df["cantidad_ordenes"].min(), df["cantidad_servicios"].min()],
    "mediana": [df["edad"].median(), df["tiempo_espera_min"].median(),
                df["cantidad_ordenes"].median(), df["cantidad_servicios"].median()],
    "maximo": [df["edad"].max(), df["tiempo_espera_min"].max(),
               df["cantidad_ordenes"].max(), df["cantidad_servicios"].max()],
    "nulos_restantes": [int(df["edad"].isna().sum()), int(df["tiempo_espera_min"].isna().sum()),
                        int(df["cantidad_ordenes"].isna().sum()), int(df["cantidad_servicios"].isna().sum())],
})
stats.to_csv(TABLAS / "tabla_estadisticas_descriptivas.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------------------
# Actualizar metricas con datos del EDA
# ---------------------------------------------------------------------------
eda = {
    "demanda_semanal_media": round(float(semana.mean()), 1),
    "demanda_semanal_min": int(semana.min()),
    "demanda_semanal_max": int(semana.max()),
    "n_semanas": int(len(semana)),
    "mes_pico": str(mensual.idxmax()),
    "mes_pico_registros": int(mensual.max()),
    "mes_valle": str(mensual.idxmin()),
    "mes_valle_registros": int(mensual.min()),
    "localidad_top": str(loc.index[0]),
    "localidad_top_pct": round(100 * loc.iloc[0] / len(df), 1),
    "servicio_top": str(serv.index[-1]),
    "servicio_top_pct": round(100 * serv.iloc[-1] / len(df), 1),
    "cie_top": str(df["codigo_cie10"].value_counts().index[0]),
    "cie_top_n": int(df["codigo_cie10"].value_counts().iloc[0]),
    "edad_media": round(float(df["edad"].mean()), 1),
    "pct_mayores_60": round(100 * float((df["edad"] >= 60).mean()), 1),
    "pct_pediatria_menor_18": round(100 * float((df["edad"] < 18).mean()), 1),
    "pct_cronicos_hta_dm": round(100 * float(df["codigo_cie10"].isin(["I10", "E11.9"]).mean()), 1),
    "n_registros_final": int(len(df)),
}
metricas = json.loads((R2 / "metricas.json").read_text(encoding="utf-8"))
metricas["eda"] = eda
(R2 / "metricas.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

print("EDA COMPLETADO")
print(f"Serie semanal: {eda['n_semanas']} semanas, media {eda['demanda_semanal_media']} registros/semana (min {eda['demanda_semanal_min']}, max {eda['demanda_semanal_max']})")
print(f"Mes pico: {eda['mes_pico']} ({eda['mes_pico_registros']:,}) | mes valle: {eda['mes_valle']} ({eda['mes_valle_registros']:,})")
print(f"Edad media: {eda['edad_media']} | >=60 anios: {eda['pct_mayores_60']}% | <18: {eda['pct_pediatria_menor_18']}%")
print(f"HTA+DM (I10/E11.9): {eda['pct_cronicos_hta_dm']}% del total")
print("Figuras: fig_serie_semanal, fig_serie_mensual, fig_localidad, fig_servicio, fig_cie10, fig_edad_sexo, fig_funnel_depuracion, fig_pipeline")
