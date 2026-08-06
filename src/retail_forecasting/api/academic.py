"""Content for the "Fundamentos matemáticos" modules on the dashboard.

Three expandable cards — Conformal Prediction, Newsvendor and the capacity LP —
each with its formula, derivation, worked example, reference code and the notes
for defending it. This is editorial content, so it lives in one data module
rather than being scattered through templates.

The worked example and the pill readouts depend on the what-if sliders, so they
are callables taking the current parameters and recommendation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import ceil
from typing import Any

from retail_forecasting.api.services.forecast import WhatIfParams

Params = WhatIfParams
Recommendation = dict[str, Any]


@dataclass(frozen=True)
class AcademicModule:
    """One mathematical module: card front, modal body and defence notes."""

    id: str
    icon: str
    color: str
    kicker: str
    title: str
    formula: str
    short: str
    big_formula: str
    intuition: str
    derivation: str
    code: str
    example: Callable[[Params, Recommendation], str]
    pills: Callable[[Params, Recommendation], list[dict[str, Any]]]
    readout: Callable[[Params, Recommendation], str]
    defense: tuple[str, ...]
    refs: tuple[str, ...]


def _fmt_units(value: float) -> str:
    return f"{value:,.0f}"


CONFORMAL = AcademicModule(
    id="conformal",
    icon="shield",
    color="var(--c-conf)",
    kicker="Conformal",
    title="Conformal Prediction",
    formula=(
        r"C(X) = [ \hat{y} - \hat{q}_{1-\alpha}, \; \hat{y} + \hat{q}_{1-\alpha} ] "
        r"\quad \text{donde} \quad P( Y \in C(X) ) \ge 1 - \alpha"
    ),
    short=(
        "Calibramos residuales sobre un holdout y obtenemos un intervalo con "
        "<b>garantía marginal</b> de cobertura — base estadística del Service Level."
    ),
    big_formula=(
        r"\begin{aligned} R_i &= | y_i - \hat{y}_i |, \quad i \in \mathcal{D}_{\text{cal}} \\ "
        r"\hat{q}_{1-\alpha} &= \text{Quantile}\left(\{R_1, \dots, R_n\}, "
        r"\frac{\lceil(n+1)(1-\alpha)\rceil}{n}\right) \\ "
        r"C(X) &= [ \hat{y}(X) - \hat{q}_{1-\alpha}, \; \hat{y}(X) + \hat{q}_{1-\alpha} ] \\ "
        r"P( Y_{n+1} &\in C(X_{n+1}) ) \ge 1 - \alpha \quad \text{(Cobertura Marginal)} "
        r"\end{aligned}"
    ),
    intuition=(
        "La idea es <b>elegante y simple</b>: si nuestros errores históricos en un conjunto de "
        "calibración tienen un cierto cuantil <code>q̂</code>, entonces — bajo el único supuesto "
        "de <i>intercambiabilidad</i> de los datos — la siguiente observación caerá dentro del "
        "intervalo <code>ŷ ± q̂</code> con probabilidad <b>al menos</b> <code>1 − α</code>. "
        "<b>No requiere</b> asumir normalidad ni propiedades del modelo: funciona con LightGBM, "
        "redes neuronales, modelos lineales o cualquier <i>black-box</i>."
    ),
    derivation=(
        "Partimos del conjunto de calibración disjunto del entrenamiento. Calculamos los "
        "<i>scores de no-conformidad</i> <code>R_i = |y_i − ŷ_i|</code>. El teorema clave "
        "(Vovk et al., 2005) prueba que el <i>rank</i> de un nuevo score es uniforme en "
        "<code>{1, …, n+1}</code>, por lo que tomar el cuantil empírico <code>q̂</code> garantiza "
        "la cobertura marginal. Para hacerlo <b>adaptativo</b> usamos <i>Mondrian CP</i> "
        "(cuantiles por segmento de SKU) y <i>weighted CP</i> para corregir el "
        "<i>covariate shift</i> estacional."
    ),
    code="""from sklearn.model_selection import train_test_split
import numpy as np

def conformal_interval(model, X_cal, y_cal, X_new, alpha=0.05):
    # 1. residuales de calibración
    y_hat_cal = model.predict(X_cal)
    scores    = np.abs(y_cal - y_hat_cal)

    # 2. cuantil ajustado por tamaño finito
    n    = len(scores)
    q    = np.quantile(scores, np.ceil((n+1)*(1-alpha))/n,
                       method="higher")

    # 3. intervalo para la nueva predicción
    y_hat = model.predict(X_new)
    return y_hat - q, y_hat, y_hat + q""",
    example=lambda p, r: (
        f"Con <code>SL = {p.service_level:.1f}%</code> trabajamos con "
        f"<code>α = {p.alpha:.3f}</code>. Sobre <code>n = 14</code> muestras de calibración, el "
        f"cuantil empírico requerido es el de orden <code>⌈15 · {1 - p.alpha:.3f}⌉ / 14 = "
        f"{ceil(15 * (1 - p.alpha)) / 14:.3f}</code>. En la última ventana ese <code>q̂</code> se "
        f"sitúa en torno a <b>{round(r['qStar'] * 0.18)} u</b>, generando los intervalos "
        f"translúcidos que ves en el gráfico principal."
    ),
    pills=lambda p, r: [
        {"label": "α", "value": f"{p.alpha:.3f}", "accent": True},
        {"label": "1 − α", "value": f"{p.service_level:.1f}%"},
        {"label": "n_cal", "value": "14"},
        {"label": "score", "value": "|y − ŷ|"},
    ],
    readout=lambda p, r: f"α = {p.alpha:.3f}",
    defense=(
        "Es distribution-free: la garantía vale sin asumir normalidad ni linealidad — clave si "
        "el tribunal pregunta por los supuestos del modelo.",
        "La garantía es marginal, no condicional: hablamos de cobertura promedio en X, no en "
        "cada subgrupo. Por eso aplicamos Mondrian CP por categoría de SKU.",
        "Bajo covariate shift (rebajas, festivos) la intercambiabilidad falla. Mitigamos con "
        "weighted CP y re-calibración semanal automática.",
        "Trade-off natural: bajar α (más cobertura) → intervalos más anchos → política "
        "Newsvendor más conservadora → más coste de almacenamiento.",
    ),
    refs=(
        "Vovk, Gammerman, Shafer (2005) — Algorithmic Learning in a Random World.",
        "Angelopoulos & Bates (2022) — A Gentle Introduction to Conformal Prediction.",
        "Tibshirani et al. (2019) — Conformal Prediction Under Covariate Shift.",
    ),
)


NEWSVENDOR = AcademicModule(
    id="newsvendor",
    icon="function",
    color="var(--c-drift)",
    kicker="Newsvendor",
    title="Newsvendor Model",
    formula=r"q^* = F^{-1}\left( \frac{c_s}{c_s + c_h} \right)",
    short=(
        "Cantidad óptima bajo coste asimétrico. El <b>critical ratio</b> determina el cuantil "
        "de demanda a cubrir."
    ),
    big_formula=(
        r"\begin{aligned} \text{Coste Esperado: } C(q) &= c_h \cdot \mathbb{E}[(q - D)^+] + "
        r"c_s \cdot \mathbb{E}[(D - q)^+] \\ \text{Bajo Leibniz: } \frac{dC}{dq} &= "
        r"c_h \cdot F(q) - c_s \cdot (1 - F(q)) = 0 \\ \Rightarrow F(q^*) &= "
        r"\frac{c_s}{c_s + c_h} = \text{CR}^* \\ \Rightarrow q^* &= F^{-1}(\text{CR}^*) "
        r"\end{aligned}"
    ),
    intuition=(
        "Cada unidad de más nos cuesta <code>c_h</code> (almacenamiento, capital inmovilizado, "
        "obsolescencia). Cada unidad de menos nos cuesta <code>c_s</code> (venta perdida, "
        "penalizaciones, daño reputacional). Si <code>c_s ≫ c_h</code> compensa pedir mucho — "
        "<b>preferimos sobre-stockear a quedarnos cortos</b>. El <i>critical ratio</i> traduce "
        "esa asimetría en el <b>cuantil exacto</b> de demanda que debemos cubrir."
    ),
    derivation=(
        "Modelamos la demanda como variable aleatoria <code>D</code> con distribución "
        "<code>F</code>. El coste esperado es convexo en <code>q</code>, así que la condición de "
        "primer orden basta: la derivada <code>c_h · F(q) − c_s · (1 − F(q))</code> se anula en "
        "el cuantil <code>CR*</code> de la CDF. En este TFG la CDF <b>no es paramétrica</b>: la "
        "construimos a partir del intervalo conformal — los extremos <code>[L, U]</code> definen "
        "el soporte y el <i>spread</i> implícito de la incertidumbre."
    ),
    code="""from scipy.stats import norm
import numpy as np

def newsvendor_q(mu, sigma, c_s, c_h, dist="normal"):
    # critical ratio
    cr = c_s / (c_s + c_h)

    if dist == "normal":
        z = norm.ppf(cr)
        return mu + z * sigma

    # empirical CDF a partir del intervalo conformal
    samples = np.random.normal(mu, sigma, 10_000)
    return np.quantile(samples, cr)


# En el endpoint /api/forecast (campo recommendation.qStar):
q_star = newsvendor_q(mu=avg_pred, sigma=conformal_sigma,
                      c_s=req.shortage_cost, c_h=req.holding_cost)""",
    example=lambda p, r: (
        f"Con <code>c_s = {p.shortage_cost:g}</code> y <code>c_h = {p.holding_cost:.1f}</code>, "
        f"el critical ratio es <code>CR* = {p.critical_ratio:.3f}</code> → debemos cubrir el "
        f"cuantil <code>{p.critical_ratio * 100:.1f}%</code> de la demanda. Aplicando z ≈ "
        f"<code>{r['z']:.2f}</code> sobre la predicción media: <code>q* = μ + z · σ ≈ </code>"
        f"<b>{_fmt_units(r['qStar'])} u</b>. <i>Mueve los sliders y observa cómo q* se reajusta "
        f"en directo.</i>"
    ),
    pills=lambda p, r: [
        {"label": "c_s", "value": f"{p.shortage_cost:g} u.m."},
        {"label": "c_h", "value": f"{p.holding_cost:.1f} u.m."},
        {"label": "CR*", "value": f"{p.critical_ratio:.3f}", "accent": True},
        {"label": "q*", "value": f"{_fmt_units(r['qStar'])} u"},
    ],
    readout=lambda p, r: f"CR* = {p.critical_ratio:.3f}",
    defense=(
        "Es el modelo seminal de inventario para productos de un solo periodo — Arrow, Harris, "
        "Marschak (1951). El comité lo reconoce inmediatamente.",
        "La elegancia: una sola fórmula cerrada captura todo el trade-off rotura/almacén. No hay "
        "hiperparámetros que tunear.",
        "Cuando c_s/c_h se desconocen, los inferimos del margen unitario y el coste financiero. "
        "Documentamos la sensibilidad en el slider del dashboard.",
        "Limitación honesta: asume demanda no estacional. Por eso re-estimamos μ y σ por ventana "
        "móvil de 14 días.",
    ),
    refs=(
        "Arrow, Harris, Marschak (1951) — Optimal Inventory Policy.",
        "Porteus (2002) — Foundations of Stochastic Inventory Theory.",
        "Bertsimas & Thiele (2006) — A Data-Driven Approach to Newsvendor Problems.",
    ),
)


def _capacity_example(p: Params, r: Recommendation) -> str:
    utilization = r["utilization"]
    if utilization >= 100:
        tail = (
            " <b>Restricción activa</b>: el solver está recortando q_i en SKUs de bajo margen. "
            "El precio sombra es estrictamente positivo — ampliar capacidad mejoraría el margen."
        )
    else:
        tail = (
            f" Restricción <b>no vinculante</b>: cada SKU recibe su q* del Newsvendor sin "
            f"recortes. Holgura {100 - utilization:g}%."
        )
    return (
        f"Capacidad actual <code>V = {p.capacity / 1000:.1f}k u</code>. La política agregada "
        f"utiliza <code>{utilization:g}%</code> del espacio.{tail}"
    )


CAPACITY = AcademicModule(
    id="capacity",
    icon="layers",
    color="var(--c-ai)",
    kicker="LP",
    title="Capacity Optimization",
    formula=(
        r"\max \sum_{i=1}^N \pi_i q_i \quad \text{s.t.} \quad \sum_{i=1}^N v_i q_i \le V, "
        r"\; q_i \ge q^*_i, \; q_i \in \mathbb{R}^+"
    ),
    short=(
        "Cuando la capacidad <code>V</code> es vinculante, repartimos espacio entre SKUs "
        "maximizando margen esperado."
    ),
    big_formula=(
        r"\begin{aligned} \max_{q_1, \dots, q_N} \quad &\sum_{i=1}^N \pi_i \cdot q_i \quad "
        r"\text{(Margen Esperado Total)} \\ \text{sujeto a:} \quad &\sum_{i=1}^N v_i \cdot q_i "
        r"\le V \quad \text{(Capacidad Física de Almacén)} \\ &q_i \ge q^{*,\text{news}}_i \quad "
        r"\text{(Mínimo Newsvendor por SKU)} \\ &q_i \le U_i^{\text{conformal}} \quad "
        r"\text{(Cota Conformal por SKU)} \\ &q_i \ge 0 \quad \forall i \end{aligned}"
    ),
    intuition=(
        "El Newsvendor te dice cuánto pedir <i>por SKU aislado</i>. Pero el almacén es "
        "<b>finito</b>: si la suma de los <code>q*_i</code> excede la capacidad, alguien tiene "
        "que ceder. La LP elige qué SKU sacrifica unidades minimizando la pérdida de margen "
        "total. Es una <b>capa de decisión</b> sobre las recomendaciones probabilísticas del "
        "modelo."
    ),
    derivation=(
        "Es un problema de Programación Lineal estándar — convexo, con dualidad fuerte. El dual "
        'nos da los <i>precios sombra</i> <code>λ</code> del recurso "capacidad": cuántos '
        "dólares de margen perdemos por cada unidad de espacio que <i>no</i> tenemos. <b>Es la "
        "métrica clave para decidir cuándo ampliar el almacén</b>. Resoluble con "
        "<code>scipy.optimize.linprog</code> (método HiGHS) en milisegundos para miles de SKUs."
    ),
    code="""from scipy.optimize import linprog
import numpy as np

def allocate_capacity(margins, volumes, q_min, q_max, capacity):
    # linprog minimiza ⇒ negamos el margen para maximizar
    c = -np.array(margins)

    # A_ub @ q ≤ b_ub : restricción de capacidad
    A_ub = [volumes]
    b_ub = [capacity]

    # bounds por SKU: [q*_i_news,  U_i_conformal]
    bounds = list(zip(q_min, q_max))

    res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                  bounds=bounds, method="highs")

    return {
        "q":            res.x,
        "margin":       -res.fun,
        "shadow_price": res.ineqlin.marginals[0],
    }""",
    example=_capacity_example,
    pills=lambda p, r: [
        {"label": "V", "value": f"{p.capacity / 1000:.1f}k u", "accent": True},
        {"label": "utilización", "value": f"{r['utilization']:g}%"},
        {"label": "estado", "value": "binding" if r["utilization"] >= 100 else "slack"},
        {"label": "solver", "value": "HiGHS"},
    ],
    readout=lambda p, r: f"capacity = {p.capacity / 1000:.1f}k u",
    defense=(
        "La LP encadena con el Newsvendor: usamos sus q* como cota inferior, manteniendo "
        "coherencia entre los dos niveles de decisión.",
        "El precio sombra λ del dual da una interpretación económica directa al stakeholder: "
        "'X u.m. de margen perdido por cada unidad de espacio faltante'.",
        "Complejidad polinomial — escala a >10k SKUs en <1s con HiGHS. Listo para producción.",
        "Extensible a programación entera mixta (MILP) si se incluyen costes fijos por SKU o "
        "restricciones de lote mínimo.",
    ),
    refs=(
        "Dantzig (1963) — Linear Programming and Extensions.",
        "Bertsimas & Tsitsiklis (1997) — Introduction to Linear Optimization.",
        "Huangfu & Hall (2018) — Parallelizing the Dual Revised Simplex Method (HiGHS).",
    ),
)


MODULES: tuple[AcademicModule, ...] = (CONFORMAL, NEWSVENDOR, CAPACITY)
MODULES_BY_ID = {module.id: module for module in MODULES}


def card_context(module: AcademicModule, params: Params, rec: Recommendation) -> dict[str, Any]:
    """Front-of-card data, including the live readout driven by the sliders."""
    return {
        "id": module.id,
        "icon": module.icon,
        "color": module.color,
        "kicker": module.kicker,
        "title": module.title,
        "formula": module.formula,
        "short": module.short,
        "readout": module.readout(params, rec),
    }


def modal_context(module: AcademicModule, params: Params, rec: Recommendation) -> dict[str, Any]:
    """Everything the expanded modal renders."""
    return {
        **card_context(module, params, rec),
        "big_formula": module.big_formula,
        "intuition": module.intuition,
        "derivation": module.derivation,
        "code": module.code,
        "example": module.example(params, rec),
        "pills": module.pills(params, rec),
        "defense": module.defense,
        "refs": module.refs,
    }
