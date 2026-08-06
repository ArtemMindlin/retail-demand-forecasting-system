"""SVG renderers for the exploratory-analysis figures.

One function per chart type emitted by :mod:`retail_forecasting.api.services.eda`.
All of them share a viewBox so the figures scale with their container, and all
colours come from the stylesheet's custom properties.

The geometry here is a reimplementation rather than a line-by-line transcription
of the former React renderers: same chart types, same data, same palette, but
the axis and spacing logic is written once and shared instead of being repeated
per component.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from django.utils.html import escape
from django.utils.safestring import SafeString, mark_safe

from retail_forecasting.api.charts import _num, nice_ticks

WIDTH = 760
HEIGHT = 300
PAD = {"left": 54, "right": 18, "top": 18, "bottom": 42}

INNER_W = WIDTH - PAD["left"] - PAD["right"]
INNER_H = HEIGHT - PAD["top"] - PAD["bottom"]

GRID_STROKE = "rgba(148,163,184,0.09)"
AXIS_FONT = 'font-family="var(--font-mono)" font-size="10" fill="var(--text-3)"'

SERIES_COLORS = ("var(--c-conf)", "var(--c-inv)", "var(--c-ai)", "var(--c-drift)")


def _svg(body: str, label: str, height: int = HEIGHT) -> SafeString:
    return mark_safe(  # noqa: S308 - callers interpolate numbers and escaped text only
        f'<svg viewBox="0 0 {WIDTH} {height}" style="width:100%;height:auto;display:block" '
        f'role="img" aria-label="{escape(label)}">{body}</svg>'
    )


def _y_axis(low: float, high: float, height: int = HEIGHT) -> tuple[str, Any]:
    """Horizontal gridlines with labels; returns the markup and a scale function."""
    inner_h = height - PAD["top"] - PAD["bottom"]
    ticks = nice_ticks(low, high, 4)
    low = min(low, ticks[0])
    high = max(high, ticks[-1])
    span = (high - low) or 1.0

    def scale(value: float) -> float:
        return PAD["top"] + (1 - (value - low) / span) * inner_h

    parts = []
    for tick in ticks:
        y = scale(tick)
        label = f"{tick:.2f}".rstrip("0").rstrip(".") if abs(tick) < 10 else f"{tick:,.0f}"
        parts.append(
            f'<line x1="{PAD["left"]}" x2="{WIDTH - PAD["right"]}" y1="{_num(y)}" '
            f'y2="{_num(y)}" stroke="{GRID_STROKE}"/>'
            f'<text x="{PAD["left"] - 8}" y="{_num(y + 3)}" text-anchor="end" {AXIS_FONT}>'
            f"{label}</text>"
        )
    return "".join(parts), scale


def _x_label(x: float, text: str, height: int = HEIGHT, rotate: bool = False) -> str:
    y = height - PAD["bottom"] + 16
    if rotate:
        return (
            f'<text x="{_num(x)}" y="{y}" text-anchor="end" {AXIS_FONT} '
            f'transform="rotate(-35 {_num(x)} {y})">{escape(text)}</text>'
        )
    return f'<text x="{_num(x)}" y="{y}" text-anchor="middle" {AXIS_FONT}>{escape(text)}</text>'


def _legend(entries: Sequence[tuple[str, str]]) -> str:
    """Inline legend swatches drawn at the top-right of the plot."""
    parts = []
    x = WIDTH - PAD["right"]
    for label, color in reversed(entries):
        width = 8 + len(label) * 6
        x -= width + 14
        parts.append(
            f'<rect x="{_num(x)}" y="{PAD["top"] - 12}" width="8" height="8" rx="2" '
            f'fill="{escape(color)}"/>'
            f'<text x="{_num(x + 12)}" y="{PAD["top"] - 4}" {AXIS_FONT}>{escape(label)}</text>'
        )
    return "".join(parts)


def line_dual(data: dict[str, Any]) -> SafeString:
    """Two lines over a categorical x-axis (weekday mean vs median)."""
    rows = data.get("data", [])
    series = data.get("series", [])
    if not rows or not series:
        return _svg("", "Sin datos")

    values = [float(row[s["key"]]) for row in rows for s in series if row.get(s["key"]) is not None]
    if not values:
        return _svg("", "Sin datos")

    grid, scale_y = _y_axis(min(values), max(values))
    divisor = max(1, len(rows) - 1)
    xs = [PAD["left"] + (i / divisor) * INNER_W for i in range(len(rows))]

    parts = [grid]
    for index, spec in enumerate(series):
        color = spec.get("color") or SERIES_COLORS[index % len(SERIES_COLORS)]
        points = " ".join(
            f"{_num(xs[i])},{_num(scale_y(float(row[spec['key']])))}"
            for i, row in enumerate(rows)
            if row.get(spec["key"]) is not None
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{escape(color)}" '
            f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for i, row in enumerate(rows):
            if row.get(spec["key"]) is None:
                continue
            parts.append(
                f'<circle cx="{_num(xs[i])}" cy="{_num(scale_y(float(row[spec["key"]])))}" '
                f'r="3" fill="{escape(color)}"/>'
            )

    first_key = next(iter(rows[0]))
    for i, row in enumerate(rows):
        parts.append(_x_label(xs[i], str(row[first_key])))

    parts.append(
        _legend([(s.get("label", s["key"]), s.get("color", SERIES_COLORS[0])) for s in series])
    )
    return _svg("".join(parts), "Perfil semanal de demanda")


def _bars(centers: Sequence[float], counts: Sequence[float], color: str, scale_y: Any) -> str:
    if not centers:
        return ""
    bar_width = INNER_W / len(centers)
    baseline = HEIGHT - PAD["bottom"]
    parts = []
    for index, count in enumerate(counts):
        y = scale_y(count)
        parts.append(
            f'<rect x="{_num(PAD["left"] + index * bar_width + 0.5)}" y="{_num(y)}" '
            f'width="{_num(max(0.0, bar_width - 1))}" height="{_num(max(0.0, baseline - y))}" '
            f'fill="{escape(color)}" fill-opacity="0.65" rx="1.5"/>'
        )
    return "".join(parts)


def histogram(data: dict[str, Any]) -> SafeString:
    """Single-series histogram with optional median annotation."""
    centers = data.get("centers", [])
    counts = data.get("counts", [])
    if not centers:
        return _svg("", "Sin datos")

    grid, scale_y = _y_axis(0, max(counts))
    parts = [grid, _bars(centers, counts, "var(--c-inv)", scale_y)]

    step = max(1, len(centers) // 8)
    bar_width = INNER_W / len(centers)
    for index in range(0, len(centers), step):
        x = PAD["left"] + index * bar_width + bar_width / 2
        parts.append(_x_label(x, f"{centers[index]:.2f}".rstrip("0").rstrip(".")))

    median = data.get("median")
    if median is not None and centers:
        low, high = centers[0], centers[-1]
        span = (high - low) or 1.0
        x = PAD["left"] + ((median - low) / span) * INNER_W
        parts.append(
            f'<line x1="{_num(x)}" x2="{_num(x)}" y1="{PAD["top"]}" '
            f'y2="{HEIGHT - PAD["bottom"]}" stroke="var(--c-drift)" stroke-dasharray="4 3"/>'
            f'<text x="{_num(x + 5)}" y="{PAD["top"] + 10}" {AXIS_FONT}>'
            f"mediana {median:.2f}</text>"
        )

    parts.append(
        f'<text x="{WIDTH / 2}" y="{HEIGHT - 6}" text-anchor="middle" {AXIS_FONT}>'
        f"{escape(str(data.get('x_label', '')))}</text>"
    )
    return _svg("".join(parts), "Histograma")


def histogram_dual(data: dict[str, Any]) -> SafeString:
    """Histogram with a log10-count overlay, for heavily skewed distributions."""
    centers = data.get("centers", [])
    counts = data.get("counts", [])
    log_counts = data.get("log_counts", [])
    if not centers:
        return _svg("", "Sin datos")

    grid, scale_y = _y_axis(0, max(counts))
    parts = [grid, _bars(centers, counts, "var(--c-inv)", scale_y)]

    if log_counts:
        peak = max(log_counts) or 1.0
        baseline = HEIGHT - PAD["bottom"]
        bar_width = INNER_W / len(centers)
        points = " ".join(
            f"{_num(PAD['left'] + i * bar_width + bar_width / 2)},"
            f"{_num(baseline - (value / peak) * INNER_H)}"
            for i, value in enumerate(log_counts)
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="var(--c-ai)" stroke-width="1.6"/>'
        )

    step = max(1, len(centers) // 8)
    bar_width = INNER_W / len(centers)
    for index in range(0, len(centers), step):
        x = PAD["left"] + index * bar_width + bar_width / 2
        parts.append(_x_label(x, f"{centers[index]:,.0f}"))

    parts.append(_legend([("conteo", "var(--c-inv)"), ("log10", "var(--c-ai)")]))
    parts.append(
        f'<text x="{WIDTH / 2}" y="{HEIGHT - 6}" text-anchor="middle" {AXIS_FONT}>'
        f"{escape(str(data.get('x_label', '')))}</text>"
    )
    return _svg("".join(parts), "Histograma con escala logarítmica")


def bar_horizontal(data: dict[str, Any]) -> SafeString:
    """Signed horizontal bars — feature correlations, strongest first."""
    rows = data.get("data", [])
    if not rows:
        return _svg("", "Sin datos")

    keys = list(rows[0])
    label_key, value_key = keys[0], keys[1]
    values = [float(row[value_key]) for row in rows]
    extent = max(abs(v) for v in values) or 1.0

    height = max(HEIGHT, 40 + len(rows) * 22)
    label_width = 190
    plot_left = label_width + 10
    plot_width = WIDTH - plot_left - PAD["right"]
    zero_x = plot_left + plot_width / 2
    row_height = (height - PAD["top"] - 20) / len(rows)

    parts = [
        f'<line x1="{_num(zero_x)}" x2="{_num(zero_x)}" y1="{PAD["top"] - 6}" '
        f'y2="{height - 20}" stroke="rgba(148,163,184,0.25)"/>'
    ]
    for index, row in enumerate(rows):
        value = float(row[value_key])
        y = PAD["top"] + index * row_height
        bar_length = (abs(value) / extent) * (plot_width / 2 - 6)
        x = zero_x if value >= 0 else zero_x - bar_length
        color = "var(--c-conf)" if value >= 0 else "var(--c-drift)"
        parts.append(
            f'<rect x="{_num(x)}" y="{_num(y + 3)}" width="{_num(bar_length)}" '
            f'height="{_num(max(4.0, row_height - 8))}" fill="{color}" '
            f'fill-opacity="0.7" rx="2"/>'
            f'<text x="{label_width}" y="{_num(y + row_height / 2 + 3)}" text-anchor="end" '
            f"{AXIS_FONT}>{escape(str(row[label_key]))}</text>"
            f'<text x="{_num(zero_x + (bar_length + 6) * (1 if value >= 0 else -1))}" '
            f'y="{_num(y + row_height / 2 + 3)}" '
            f'text-anchor="{"start" if value >= 0 else "end"}" {AXIS_FONT}>'
            f"{value:+.3f}</text>"
        )
    return _svg("".join(parts), "Correlaciones por feature", height)


def bar_group(data: dict[str, Any]) -> SafeString:
    """Grouped bars over a categorical axis, one group per numeric column."""
    rows = data.get("data", [])
    if not rows:
        return _svg("", "Sin datos")

    x_key = data.get("x_key") or next(iter(rows[0]))
    value_keys = [k for k in rows[0] if k != x_key]
    if not value_keys:
        return _svg("", "Sin datos")

    # Columns can differ by orders of magnitude (mean demand vs observation
    # counts), so each series is normalised to its own maximum and the real
    # value is printed above the bar.
    maxima = {k: max(float(r[k]) for r in rows) or 1.0 for k in value_keys}

    group_width = INNER_W / len(rows)
    bar_width = (group_width - 10) / len(value_keys)
    baseline = HEIGHT - PAD["bottom"]
    parts: list[str] = []

    for row_index, row in enumerate(rows):
        group_x = PAD["left"] + row_index * group_width + 5
        for series_index, key in enumerate(value_keys):
            value = float(row[key])
            bar_height = (value / maxima[key]) * INNER_H
            x = group_x + series_index * bar_width
            color = SERIES_COLORS[series_index % len(SERIES_COLORS)]
            parts.append(
                f'<rect x="{_num(x)}" y="{_num(baseline - bar_height)}" '
                f'width="{_num(max(0.0, bar_width - 3))}" height="{_num(bar_height)}" '
                f'fill="{color}" fill-opacity="0.7" rx="2"/>'
                f'<text x="{_num(x + bar_width / 2 - 1.5)}" '
                f'y="{_num(baseline - bar_height - 5)}" text-anchor="middle" {AXIS_FONT}>'
                f"{value:,.0f}</text>"
            )
        parts.append(_x_label(group_x + (group_width - 10) / 2, str(row[x_key])))

    parts.append(
        _legend(
            [
                (key.replace("_", " "), SERIES_COLORS[i % len(SERIES_COLORS)])
                for i, key in enumerate(value_keys)
            ]
        )
    )
    return _svg("".join(parts), "Demanda por banda de stockout")


def scatter(data: dict[str, Any]) -> SafeString:
    """Point cloud of two numeric columns."""
    rows = data.get("data", [])
    if not rows:
        return _svg("", "Sin datos")

    xs = [float(r["x"]) for r in rows]
    ys = [float(r["y"]) for r in rows]
    x_low, x_high = min(xs), max(xs)
    x_span = (x_high - x_low) or 1.0

    grid, scale_y = _y_axis(min(ys), max(ys))
    parts = [grid]
    for x_value, y_value in zip(xs, ys, strict=True):
        x = PAD["left"] + ((x_value - x_low) / x_span) * INNER_W
        parts.append(
            f'<circle cx="{_num(x)}" cy="{_num(scale_y(y_value))}" r="2.6" '
            f'fill="var(--c-ai)" fill-opacity="0.45"/>'
        )

    for fraction in (0, 0.25, 0.5, 0.75, 1.0):
        parts.append(
            _x_label(PAD["left"] + fraction * INNER_W, f"{x_low + fraction * x_span:,.0f}")
        )
    return _svg("".join(parts), "Horas de stockout frente a demanda observada")


def boxplot(data: dict[str, Any]) -> SafeString:
    """Box-and-whisker sketch per series (±1σ box, ±2σ whiskers)."""
    boxes = data.get("data", [])
    if not boxes:
        return _svg("", "Sin datos")

    values = [v for box in boxes for v in (box["min"], box["max"])]
    grid, scale_y = _y_axis(min(values), max(values))

    slot = INNER_W / len(boxes)
    box_width = min(26.0, slot * 0.55)
    parts = [grid]

    for index, box in enumerate(boxes):
        center = PAD["left"] + index * slot + slot / 2
        left = center - box_width / 2
        y_q3 = scale_y(box["q3"])
        y_q1 = scale_y(box["q1"])
        parts.append(
            f'<line x1="{_num(center)}" x2="{_num(center)}" y1="{_num(scale_y(box["max"]))}" '
            f'y2="{_num(scale_y(box["min"]))}" stroke="rgba(148,163,184,0.45)"/>'
            f'<rect x="{_num(left)}" y="{_num(y_q3)}" width="{_num(box_width)}" '
            f'height="{_num(max(1.0, y_q1 - y_q3))}" fill="var(--c-inv)" fill-opacity="0.35" '
            f'stroke="var(--c-inv)" rx="2"/>'
            f'<line x1="{_num(left)}" x2="{_num(left + box_width)}" '
            f'y1="{_num(scale_y(box["median"]))}" y2="{_num(scale_y(box["median"]))}" '
            f'stroke="var(--c-conf)" stroke-width="2"/>'
        )
        parts.append(_x_label(center, str(box["id"]), rotate=True))

    return _svg("".join(parts), "Dispersión de demanda por serie")


def series_grid(data: dict[str, Any]) -> SafeString:
    """Small-multiples summary table of the most representative series."""
    rows = data.get("series", [])
    if not rows:
        return _svg("", "Sin datos")

    columns = [k for k in rows[0] if k != "series_id"]
    row_height = 26
    height = 44 + len(rows) * row_height
    label_width = 150
    column_width = (WIDTH - label_width - PAD["right"]) / max(1, len(columns))

    parts = [
        f'<text x="{PAD["left"] - 6}" y="24" {AXIS_FONT}>serie</text>',
    ]
    for index, column in enumerate(columns):
        parts.append(
            f'<text x="{_num(label_width + index * column_width + column_width / 2)}" y="24" '
            f'text-anchor="middle" {AXIS_FONT}>{escape(column.replace("_", " "))}</text>'
        )

    for row_index, row in enumerate(rows):
        y = 44 + row_index * row_height
        if row_index % 2 == 0:
            parts.append(
                f'<rect x="{PAD["left"] - 10}" y="{_num(y - 14)}" '
                f'width="{WIDTH - PAD["left"] - PAD["right"] + 10}" height="{row_height - 4}" '
                f'fill="rgba(255,255,255,0.02)" rx="4"/>'
            )
        parts.append(
            f'<text x="{PAD["left"] - 6}" y="{_num(y)}" font-family="var(--font-mono)" '
            f'font-size="11" fill="var(--text-1)">{escape(str(row["series_id"]))}</text>'
        )
        for index, column in enumerate(columns):
            value = row[column]
            text = "—" if value is None else f"{float(value):,.2f}".rstrip("0").rstrip(".")
            parts.append(
                f'<text x="{_num(label_width + index * column_width + column_width / 2)}" '
                f'y="{_num(y)}" text-anchor="middle" font-family="var(--font-mono)" '
                f'font-size="11" fill="var(--text-2)">{escape(text)}</text>'
            )

    return _svg("".join(parts), "Series representativas", height)


RENDERERS = {
    "line_dual": line_dual,
    "histogram": histogram,
    "histogram_dual": histogram_dual,
    "bar_horizontal": bar_horizontal,
    "bar_group": bar_group,
    "scatter": scatter,
    "boxplot": boxplot,
    "series_grid": series_grid,
}


def render(data: dict[str, Any]) -> SafeString:
    """Dispatch to the renderer for ``data["type"]`` (empty svg if unknown)."""
    renderer = RENDERERS.get(data.get("type", ""))
    return renderer(data) if renderer else _svg("", "Gráfico no disponible")


# ── Research-plane charts ─────────────────────────────────────────────────────


def latent_compare(data: dict[str, Any], strategy_colors: dict[str, str]) -> SafeString:
    """Observed (censored) sale against each strategy's reconstruction.

    Days with stockout hours are shaded, because those are exactly the days
    where observed sale is a censored view of demand and the strategies diverge.
    """
    dates = data.get("dates", [])
    observed = data.get("observed", [])
    strategies = data.get("strategies", {})
    if not dates:
        return _svg("", "Sin datos")

    height = 380
    inner_h = height - PAD["top"] - PAD["bottom"]
    values = [v for v in observed if v is not None]
    for series in strategies.values():
        values.extend(v for v in series if v is not None)
    if not values:
        return _svg("", "Sin datos")

    grid, scale_y = _y_axis(min(values), max(values), height)
    divisor = max(1, len(dates) - 1)
    xs = [PAD["left"] + (i / divisor) * INNER_W for i in range(len(dates))]

    parts: list[str] = []

    # Stockout shading sits under the grid so the lines stay legible.
    stockout = data.get("stockout_hours", [])
    band_width = INNER_W / max(1, len(dates))
    for index, hours in enumerate(stockout):
        if hours:
            parts.append(
                f'<rect x="{_num(xs[index] - band_width / 2)}" y="{PAD["top"]}" '
                f'width="{_num(band_width)}" height="{_num(inner_h)}" '
                f'fill="rgba(245,158,11,0.10)"/>'
            )

    parts.append(grid)

    def polyline(series: Sequence[float | None], color: str, dashed: bool = False) -> str:
        points = " ".join(
            f"{_num(xs[i])},{_num(scale_y(v))}" for i, v in enumerate(series) if v is not None
        )
        if not points:
            return ""
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        return (
            f'<polyline points="{points}" fill="none" stroke="{escape(color)}" '
            f'stroke-width="2" stroke-linejoin="round"{dash}/>'
        )

    for name, series in strategies.items():
        parts.append(polyline(series, strategy_colors.get(name, "#94a3b8")))
    parts.append(polyline(observed, "var(--text-2)", dashed=True))

    step = max(1, len(dates) // 8)
    for index in range(0, len(dates), step):
        parts.append(_x_label(xs[index], str(dates[index])[5:], height))

    legend = [(strategy_meta_label(n), strategy_colors.get(n, "#94a3b8")) for n in strategies]
    legend.append(("observado", "var(--text-2)"))
    parts.append(_legend(legend))

    return _svg("".join(parts), "Reconstrucción de demanda latente por estrategia", height)


def strategy_meta_label(name: str) -> str:
    """Short label for a strategy legend entry."""
    from retail_forecasting.api.services.experiments import strategy_meta

    return strategy_meta(name)["short"]


def pareto_chart(rows: Sequence[dict[str, Any]]) -> SafeString:
    """Pinball vs Winkler scatter with the non-dominated front joined up."""
    points = [
        (float(r["pinball_loss"]), float(r["winkler_score"]), r)
        for r in rows
        if r.get("pinball_loss") is not None and r.get("winkler_score") is not None
    ]
    if not points:
        return _svg("", "Sin frente de Pareto")

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_low, x_high = min(xs), max(xs)
    x_span = (x_high - x_low) or 1.0

    grid, scale_y = _y_axis(min(ys), max(ys))

    def scale_x(value: float) -> float:
        return PAD["left"] + ((value - x_low) / x_span) * INNER_W

    # Non-dominated set: no other point is better on both objectives at once.
    front = sorted(
        (
            p
            for p in points
            if not any(o[0] <= p[0] and o[1] <= p[1] and o is not p for o in points)
        ),
        key=lambda p: p[0],
    )

    parts = [grid]
    if len(front) > 1:
        line = " ".join(f"{_num(scale_x(p[0]))},{_num(scale_y(p[1]))}" for p in front)
        parts.append(
            f'<polyline points="{line}" fill="none" stroke="var(--c-conf)" '
            f'stroke-width="1.6" stroke-dasharray="4 3"/>'
        )

    front_ids = {id(p) for p in front}
    for point in points:
        on_front = id(point) in front_ids
        parts.append(
            f'<circle cx="{_num(scale_x(point[0]))}" cy="{_num(scale_y(point[1]))}" '
            f'r="{5 if on_front else 3.5}" '
            f'fill="{"var(--c-conf)" if on_front else "var(--c-ai)"}" '
            f'fill-opacity="{0.9 if on_front else 0.45}"/>'
        )

    for fraction in (0, 0.5, 1.0):
        parts.append(_x_label(PAD["left"] + fraction * INNER_W, f"{x_low + fraction * x_span:.3f}"))

    parts.append(_legend([("frente", "var(--c-conf)"), ("dominados", "var(--c-ai)")]))
    parts.append(
        f'<text x="{WIDTH / 2}" y="{HEIGHT - 6}" text-anchor="middle" {AXIS_FONT}>'
        f"pinball loss → · winkler score ↑</text>"
    )
    return _svg("".join(parts), "Frente de Pareto de tuning multiobjetivo")


def sensitivity_chart(rows: Sequence[dict[str, Any]]) -> SafeString:
    """Cost (and service level) as the shortage/holding cost ratio varies."""
    if not rows:
        return _svg("", "Sin datos de sensibilidad")

    x_key = next((k for k in ("cs_co_ratio", "ratio", "c_under") if k in rows[0]), None)
    if x_key is None:
        return _svg("", "Sin datos de sensibilidad")

    value_keys = [
        k
        for k in rows[0]
        if k != x_key and isinstance(rows[0][k], int | float) and rows[0][k] is not None
    ][:4]
    if not value_keys:
        return _svg("", "Sin datos de sensibilidad")

    xs_raw = [float(r[x_key]) for r in rows]
    x_low, x_high = min(xs_raw), max(xs_raw)
    x_span = (x_high - x_low) or 1.0

    parts: list[str] = []
    legend: list[tuple[str, str]] = []

    # Each series is normalised to its own range: the columns are different
    # units (cost, service level) and would otherwise be unreadable together.
    for index, key in enumerate(value_keys):
        values = [float(r[key]) for r in rows]
        low, high = min(values), max(values)
        span = (high - low) or 1.0
        color = SERIES_COLORS[index % len(SERIES_COLORS)]
        line = " ".join(
            f"{_num(PAD['left'] + ((x - x_low) / x_span) * INNER_W)},"
            f"{_num(PAD['top'] + (1 - (v - low) / span) * INNER_H)}"
            for x, v in zip(xs_raw, values, strict=True)
        )
        parts.append(f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2"/>')
        legend.append((key.replace("_", " "), color))

    for fraction in (0, 0.25, 0.5, 0.75, 1.0):
        parts.append(_x_label(PAD["left"] + fraction * INNER_W, f"{x_low + fraction * x_span:.2f}"))

    parts.append(_legend(legend))
    parts.append(
        f'<text x="{WIDTH / 2}" y="{HEIGHT - 6}" text-anchor="middle" {AXIS_FONT}>'
        f"{escape(x_key.replace('_', ' '))} · series normalizadas a su propio rango</text>"
    )
    return _svg("".join(parts), "Sensibilidad al ratio de costes")
