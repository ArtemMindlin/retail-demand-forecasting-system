"""Server-rendered SVG chart primitives.

The dashboard's charts were always hand-written SVG rather than a JavaScript
charting library, which is what makes rendering them on the server practical:
the same geometry that React computed in the browser is computed here in Python
and shipped as markup. No client-side chart code, no chart library.

Each function returns a :class:`~django.utils.safestring.SafeString` holding a
complete ``<svg>`` element. Colours are CSS custom properties so the existing
stylesheet keeps full control of the palette.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from django.utils.html import escape
from django.utils.safestring import SafeString, mark_safe

Point = tuple[float, float]


def _num(value: float) -> str:
    """Format a coordinate compactly, dropping trailing zeros."""
    return f"{value:.2f}".rstrip("0").rstrip(".")


def smooth_path(points: Sequence[Point], tension: float = 0.2) -> str:
    """Catmull–Rom spline expressed as cubic Bézier segments.

    Same smoothing the browser applied, so the rendered curve is identical.
    """
    if not points:
        return ""
    if len(points) == 1:
        return f"M {_num(points[0][0])} {_num(points[0][1])}"

    commands = [f"M {_num(points[0][0])} {_num(points[0][1])}"]
    for i in range(len(points) - 1):
        p0 = points[i - 1] if i > 0 else points[i]
        p1 = points[i]
        p2 = points[i + 1]
        p3 = points[i + 2] if i + 2 < len(points) else p2
        c1x = p1[0] + (p2[0] - p0[0]) * tension
        c1y = p1[1] + (p2[1] - p0[1]) * tension
        c2x = p2[0] - (p3[0] - p1[0]) * tension
        c2y = p2[1] - (p3[1] - p1[1]) * tension
        commands.append(
            f"C {_num(c1x)} {_num(c1y)}, {_num(c2x)} {_num(c2y)}, {_num(p2[0])} {_num(p2[1])}"
        )
    return " ".join(commands)


def nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    """Round axis ticks spanning ``[low, high]`` at a human-friendly step."""
    span = high - low
    if span <= 0:
        return [low]

    rough_step = span / count
    magnitude = 10 ** math.floor(math.log10(rough_step))
    normalized = rough_step / magnitude
    if normalized >= 5:
        step = 10 * magnitude
    elif normalized >= 2:
        step = 5 * magnitude
    elif normalized >= 1:
        step = 2 * magnitude
    else:
        step = magnitude

    start = math.floor(low / step) * step
    ticks = []
    value = start
    while value <= high + 1e-9:
        ticks.append(value)
        value += step
    return ticks


def hero_y_domain(values: Sequence[float], count: int = 5) -> tuple[float, float, list[float]]:
    """Padded y domain plus the round ticks that fall inside it.

    Snapping the domain out to the first and last tick wastes plot height: a 40-250
    range snaps down to 0, and the marks then live in the top half of the card with
    dead space below. Clipping the ticks to the domain instead keeps the labels round
    and lets the marks use the full height. Falls back to snapping when the range is
    so narrow that fewer than two ticks land inside it, which would leave no grid.
    """
    low, high = float(min(values)), float(max(values))
    span = (high - low) or 1.0
    low -= span * 0.08
    high += span * 0.08
    ticks = [tick for tick in nice_ticks(low, high, count) if low <= tick <= high]
    if len(ticks) < 2:
        ticks = nice_ticks(low, high, count)
        low, high = min(low, ticks[0]), max(high, ticks[-1])
    return low, high, ticks


def distribution_chart(
    pre: Sequence[float],
    post: Sequence[float],
    color: str,
    width: int = 200,
    height: int = 56,
) -> SafeString:
    """Reference-vs-current histogram overlay for one feature.

    ``pre`` (reference window) renders as muted grey bars, ``post`` (current
    window) as translucent bars in the severity colour on top, so the shift
    between the two distributions is the visible signal.
    """
    bins = len(pre)
    if bins == 0 or len(post) != bins:
        return mark_safe(f'<svg width="{width}" height="{height}" aria-hidden="true"></svg>')

    peak = max((*pre, *post), default=0.0)
    if peak <= 0:
        return mark_safe(f'<svg width="{width}" height="{height}" aria-hidden="true"></svg>')

    bar_width = width / bins
    usable = height - 4
    parts: list[str] = []

    for values, fill, opacity in (
        (pre, "var(--c-slate)", "0.25"),
        (post, color, "0.6"),
    ):
        for index, value in enumerate(values):
            bar_height = (value / peak) * usable
            parts.append(
                f'<rect x="{_num(index * bar_width + 1)}" y="{_num(height - bar_height)}" '
                f'width="{_num(max(0.0, bar_width - 2))}" height="{_num(bar_height)}" '
                f'fill="{escape(fill)}" fill-opacity="{opacity}" rx="1.5"/>'
            )

    return mark_safe(  # noqa: S308 - all interpolated values are numeric or escaped
        f'<svg width="{width}" height="{height}" style="display:block" '
        f'role="img" aria-label="Distribución de referencia frente a actual">'
        f"{''.join(parts)}</svg>"
    )


def sparkline(
    values: Sequence[float],
    color: str = "var(--c-inv)",
    width: int = 90,
    height: int = 24,
    stroke_width: float = 1.4,
    fill: bool = True,
) -> SafeString:
    """Compact trend line, optionally filled beneath the curve."""
    if len(values) < 2:
        return mark_safe(f'<svg width="{width}" height="{height}" aria-hidden="true"></svg>')

    low = min(values)
    high = max(values)
    span = (high - low) or 1.0
    step = width / (len(values) - 1)

    points = [
        (index * step, height - ((value - low) / span) * (height - 2) - 1)
        for index, value in enumerate(values)
    ]
    line = " ".join(f"{_num(x)},{_num(y)}" for x, y in points)

    area = ""
    if fill:
        area = (
            f'<polygon points="0,{height} {line} {_num(width)},{height}" '
            f'fill="{escape(color)}" fill-opacity="0.12"/>'
        )

    return mark_safe(  # noqa: S308 - numeric coordinates, escaped colour
        f'<svg width="{width}" height="{height}" style="display:block" aria-hidden="true">'
        f'{area}<polyline points="{line}" fill="none" stroke="{escape(color)}" '
        f'stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"/></svg>'
    )


# ── Hero chart geometry ───────────────────────────────────────────────────────
# User-space dimensions for the viewBox; the <svg> scales uniformly to fill its
# container width (see .forecast-svg in components.css), which is what replaces the
# browser's ResizeObserver. No layout JavaScript required. Scaling must stay uniform:
# stretching the viewBox non-uniformly (independent x/y factors) distorts every native
# <circle> and <text> into ellipses and squeezed glyphs as the panel is resized — the
# straight lines survive that because they carry vector-effect="non-scaling-stroke",
# but markers and axis labels have no equivalent escape hatch.
#
# One ratio for every full-width chart card. Because these SVGs are width-driven, the
# ratio IS the rendered height: at the 1200px `.chart-card` cap this one lands at ~307px,
# so /dashboard/, /ops/ and /latent/ all show a chart of the same height instead of each
# picking its own and reading as a different component. See HERO_HEIGHT in eda_charts.py
# for the charts drawn on the 760-wide grid.
HERO_WIDTH = 900
HERO_HEIGHT = 230
HERO_RATIO = HERO_WIDTH / HERO_HEIGHT
_PAD = {"top": 18, "right": 18, "bottom": 30, "left": 48}

FORECAST_WIDTH = HERO_WIDTH
FORECAST_HEIGHT = HERO_HEIGHT


def forecast_chart(series: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Actual vs predicted demand with the conformal band, as one SVG.

    Returns the markup plus the per-point geometry, so the template can render a
    hover tooltip without recomputing any of the scaling in the browser.
    """
    if not series:
        return {"svg": mark_safe(""), "points": [], "width": FORECAST_WIDTH}

    inner_w = FORECAST_WIDTH - _PAD["left"] - _PAD["right"]
    inner_h = FORECAST_HEIGHT - _PAD["top"] - _PAD["bottom"]
    divisor = max(1, len(series) - 1)
    xs = [_PAD["left"] + (i / divisor) * inner_w for i in range(len(series))]

    lows: list[float] = []
    highs: list[float] = []
    for row in series:
        actual = row.get("actual")
        lows.append(min(row["lower"], row["predicted"], actual if actual is not None else math.inf))
        highs.append(
            max(row["upper"], row["predicted"], actual if actual is not None else -math.inf)
        )
    low, high, ticks = hero_y_domain([min(lows), max(highs)])

    def scale_y(value: float) -> float:
        return _PAD["top"] + (1 - (value - low) / (high - low)) * inner_h

    actual_points = [
        (xs[i], scale_y(row["actual"]))
        for i, row in enumerate(series)
        if row.get("actual") is not None
    ]
    predicted_points = [(xs[i], scale_y(row["predicted"])) for i, row in enumerate(series)]
    upper_points = [(xs[i], scale_y(row["upper"])) for i, row in enumerate(series)]
    lower_points = [(xs[i], scale_y(row["lower"])) for i, row in enumerate(series)]

    band = (
        "M "
        + " L ".join(f"{_num(x)} {_num(y)}" for x, y in upper_points)
        + " L "
        + " L ".join(f"{_num(x)} {_num(y)}" for x, y in reversed(lower_points))
        + " Z"
    )

    parts: list[str] = [
        "<defs>"
        '<linearGradient id="gradInterval" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0%" stop-color="var(--c-slate)" stop-opacity="0.26"/>'
        '<stop offset="60%" stop-color="var(--c-inv)" stop-opacity="0.10"/>'
        '<stop offset="100%" stop-color="var(--c-inv)" stop-opacity="0.04"/>'
        "</linearGradient>"
        '<linearGradient id="gradPred" x1="0" y1="0" x2="1" y2="0">'
        '<stop offset="0%" stop-color="var(--c-ai)" stop-opacity="0.5"/>'
        '<stop offset="60%" stop-color="var(--c-ai)" stop-opacity="1"/>'
        '<stop offset="100%" stop-color="var(--c-ai-bright)" stop-opacity="1"/>'
        "</linearGradient>"
        "</defs>"
    ]

    # Y grid and labels.
    for tick in ticks:
        y = scale_y(tick)
        parts.append(
            f'<line x1="{_PAD["left"]}" x2="{FORECAST_WIDTH - _PAD["right"]}" '
            f'y1="{_num(y)}" y2="{_num(y)}" stroke="var(--border)" stroke-width="1"/>'
            f'<text x="{_PAD["left"] - 8}" y="{_num(y + 3)}" text-anchor="end" '
            f'fill="var(--text-3)" font-family="var(--font-mono)" font-size="10.5">'
            f"{round(tick)}</text>"
        )

    # X labels every fourth point, plus the last one.
    for i, row in enumerate(series):
        if i % 4 == 0 or i == len(series) - 1:
            parts.append(
                f'<text x="{_num(xs[i])}" y="{FORECAST_HEIGHT - _PAD["bottom"] + 16}" '
                f'text-anchor="middle" fill="var(--text-3)" '
                f'font-family="var(--font-mono)" font-size="10.5">'
                f"{escape(str(row['label']))}</text>"
            )

    parts.append(f'<path d="{band}" fill="url(#gradInterval)"/>')
    for band_points in (upper_points, lower_points):
        parts.append(
            f'<path d="{smooth_path(band_points)}" fill="none" stroke="var(--c-slate)" stroke-opacity="0.35" '
            f'stroke-width="1" stroke-dasharray="2 3" vector-effect="non-scaling-stroke"/>'
        )
    parts.append(
        f'<path d="{smooth_path(predicted_points)}" fill="none" stroke="url(#gradPred)" '
        f'stroke-width="2" stroke-dasharray="6 4" stroke-linecap="round" '
        f'vector-effect="non-scaling-stroke"/>'
    )
    parts.append(
        f'<path d="{smooth_path(actual_points)}" fill="none" stroke="var(--c-inv)" '
        f'stroke-width="2" stroke-linecap="round" vector-effect="non-scaling-stroke"/>'
    )

    # Hover furniture, positioned by the template's small pointer handler.
    parts.append(
        '<g class="chart-hover" style="display:none">'
        f'<line class="chart-hover-line" y1="{_PAD["top"]}" y2="{_PAD["top"] + inner_h}" '
        'stroke="var(--c-slate)" stroke-opacity="0.35" stroke-dasharray="3 3" stroke-width="1"/>'
        '<circle class="chart-hover-actual" r="4" fill="var(--c-inv)" stroke="var(--text-1)" '
        'stroke-width="1.5"/>'
        '<circle class="chart-hover-pred" r="4" fill="var(--c-ai)" stroke="var(--text-1)" '
        'stroke-width="1.5"/>'
        "</g>"
    )

    svg = (
        f'<svg class="forecast-svg" viewBox="0 0 {FORECAST_WIDTH} {FORECAST_HEIGHT}" '
        f'role="img" aria-label="Demanda real frente a predicha con intervalo conformal">'
        f"{''.join(parts)}</svg>"
    )

    points = [
        {
            "x": round(xs[i], 2),
            "yPred": round(scale_y(row["predicted"]), 2),
            "yActual": round(scale_y(row["actual"]), 2) if row.get("actual") is not None else None,
            "label": row["label"],
            "actual": row.get("actual"),
            "predicted": row["predicted"],
            "lower": row["lower"],
            "upper": row["upper"],
            # Preformatted: the tooltip handler assigns fields verbatim rather than
            # knowing how any one chart wants its numbers composed.
            "interval": f"{row['lower']}–{row['upper']}",
        }
        for i, row in enumerate(series)
    ]

    return {  # noqa: S308
        "svg": mark_safe(svg),
        "points": points,
        "width": FORECAST_WIDTH,
        "height": FORECAST_HEIGHT,
    }


# ── OPS backtest trajectory ───────────────────────────────────────────────────
# Same geometry and same visual language as the forecast chart: violet dashed line for
# what the model predicted, grey-to-blue wash for the conformal band, and the realized
# value as a dot. A reader moving from /dashboard/ to /ops/ should not have to relearn
# which colour means what.
OPS_WIDTH = HERO_WIDTH
OPS_HEIGHT = HERO_HEIGHT
_OPS_PAD = dict(_PAD)


def ops_trajectory_chart(
    points: Sequence[dict[str, Any]],
    selected_week: int | None = None,
) -> dict[str, Any]:
    """Weekly forecast, conformal band and revealed actuals for one series.

    Actuals are drawn green inside the band and red outside it, so a glance at
    the dots is a read on realized coverage.

    Returns the markup plus the per-origin geometry, so the shared pointer handler
    can position the crosshair and tooltip without recomputing any scaling.
    """
    if not points:
        return {"svg": mark_safe(""), "points": [], "width": OPS_WIDTH, "height": OPS_HEIGHT}

    values = [
        value
        for point in points
        for value in (point["lower"], point["upper"], point["y_pred"], point.get("y_true"))
        if value is not None
    ]
    low, high, ticks = hero_y_domain(values, count=4)

    inner_w = OPS_WIDTH - _OPS_PAD["left"] - _OPS_PAD["right"]
    inner_h = OPS_HEIGHT - _OPS_PAD["top"] - _OPS_PAD["bottom"]
    divisor = max(1, len(points) - 1)

    def scale_x(index: int) -> float:
        fraction = 0.5 if len(points) == 1 else index / divisor
        return _OPS_PAD["left"] + fraction * inner_w

    def scale_y(value: float) -> float:
        return _OPS_PAD["top"] + (1 - (value - low) / (high - low)) * inner_h

    parts: list[str] = [
        "<defs>"
        '<linearGradient id="opsGradInterval" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0%" stop-color="var(--c-slate)" stop-opacity="0.26"/>'
        '<stop offset="60%" stop-color="var(--c-inv)" stop-opacity="0.10"/>'
        '<stop offset="100%" stop-color="var(--c-inv)" stop-opacity="0.04"/>'
        "</linearGradient>"
        '<linearGradient id="opsGradPred" x1="0" y1="0" x2="1" y2="0">'
        '<stop offset="0%" stop-color="var(--c-ai)" stop-opacity="0.5"/>'
        '<stop offset="60%" stop-color="var(--c-ai)" stop-opacity="1"/>'
        '<stop offset="100%" stop-color="var(--c-ai-bright)" stop-opacity="1"/>'
        "</linearGradient>"
        "</defs>"
    ]

    # Round ticks, solid hairline grid: the axis used to label the padded data extremes
    # (241 / 141 / 42), which are arbitrary numbers the reader cannot anchor to.
    for tick in ticks:
        y = scale_y(tick)
        parts.append(
            f'<line x1="{_OPS_PAD["left"]}" x2="{OPS_WIDTH - _OPS_PAD["right"]}" '
            f'y1="{_num(y)}" y2="{_num(y)}" stroke="var(--border)" stroke-width="1"/>'
            f'<text x="{_OPS_PAD["left"] - 8}" y="{_num(y + 3)}" text-anchor="end" '
            f'fill="var(--text-3)" font-family="var(--font-mono)" font-size="10.5">'
            f"{round(tick)}</text>"
        )

    marker = None
    if selected_week is not None:
        marker = next((i for i, p in enumerate(points) if p["week_index"] == selected_week), None)
    if marker is not None:
        x = scale_x(marker)
        parts.append(
            f'<line x1="{_num(x)}" x2="{_num(x)}" y1="{_OPS_PAD["top"]}" '
            f'y2="{_num(_OPS_PAD["top"] + inner_h)}" stroke="var(--c-ai)" stroke-width="1" '
            f'opacity="0.35" vector-effect="non-scaling-stroke"/>'
        )

    upper_points = [(scale_x(i), scale_y(p["upper"])) for i, p in enumerate(points)]
    lower_points = [(scale_x(i), scale_y(p["lower"])) for i, p in enumerate(points)]
    predicted_points = [(scale_x(i), scale_y(p["y_pred"])) for i, p in enumerate(points)]

    band = (
        "M "
        + " L ".join(f"{_num(x)} {_num(y)}" for x, y in upper_points)
        + " L "
        + " L ".join(f"{_num(x)} {_num(y)}" for x, y in reversed(lower_points))
        + " Z"
    )
    parts.append(f'<path d="{band}" fill="url(#opsGradInterval)"/>')
    for band_points in (upper_points, lower_points):
        parts.append(
            f'<path d="{smooth_path(band_points)}" fill="none" stroke="var(--c-slate)" stroke-opacity="0.35" '
            f'stroke-width="1" stroke-dasharray="2 3" vector-effect="non-scaling-stroke"/>'
        )
    parts.append(
        f'<path d="{smooth_path(predicted_points)}" fill="none" stroke="url(#opsGradPred)" '
        f'stroke-width="2" stroke-dasharray="6 4" stroke-linecap="round" '
        f'vector-effect="non-scaling-stroke"/>'
    )

    for index, point in enumerate(points):
        if point.get("y_true") is None:
            continue
        fill = "var(--c-conf)" if point["covered"] else "var(--c-crit)"
        cx, cy = scale_x(index), scale_y(point["y_true"])
        if index == marker:
            parts.append(
                f'<circle cx="{_num(cx)}" cy="{_num(cy)}" r="8" fill="none" '
                f'stroke="{fill}" stroke-width="1" opacity="0.4"/>'
            )
        # 2px ring in the surface colour, so a dot stays legible over the band edge.
        parts.append(
            f'<circle cx="{_num(cx)}" cy="{_num(cy)}" r="4" fill="{fill}" '
            f'stroke="var(--bg-1)" stroke-width="2"/>'
        )

    for index, point in enumerate(points):
        weight = ' font-weight="600"' if index == marker else ""
        colour = "var(--text-2)" if index == marker else "var(--text-3)"
        parts.append(
            f'<text x="{_num(scale_x(index))}" y="{OPS_HEIGHT - _OPS_PAD["bottom"] + 16}" '
            f'text-anchor="middle" font-family="var(--font-mono)" font-size="10.5" '
            f'fill="{colour}"{weight}>{escape(point["origin_date"][5:])}</text>'
        )

    # Hover furniture, positioned by the shared pointer handler (static/js/chart.js).
    parts.append(
        '<g class="chart-hover" style="display:none">'
        f'<line class="chart-hover-line" y1="{_OPS_PAD["top"]}" '
        f'y2="{_num(_OPS_PAD["top"] + inner_h)}" stroke="var(--c-slate)" stroke-opacity="0.35" '
        'stroke-dasharray="3 3" stroke-width="1"/>'
        '<circle class="chart-hover-pred" r="4" fill="var(--c-ai)" stroke="var(--text-1)" '
        'stroke-width="1.5"/>'
        "</g>"
    )

    svg = (
        f'<svg class="forecast-svg" viewBox="0 0 {OPS_WIDTH} {OPS_HEIGHT}" '
        f'role="img" aria-label="Trayectoria semanal de la serie">{"".join(parts)}</svg>'
    )

    hover_points = [
        {
            "x": round(scale_x(index), 2),
            "yPred": round(scale_y(point["y_pred"]), 2),
            "yActual": (
                None if point.get("y_true") is None else round(scale_y(point["y_true"]), 2)
            ),
            "label": point["origin_date"],
            "predicted": round(float(point["y_pred"])),
            "interval": f"{round(float(point['lower']))}–{round(float(point['upper']))}",
            "actual": None if point.get("y_true") is None else round(float(point["y_true"])),
            "coverage": (
                "—" if point.get("y_true") is None else ("dentro" if point["covered"] else "fuera")
            ),
            "order": round(float(point["order_quantity"])),
            "cost": f"{float(point['total_cost']):,.0f}",
        }
        for index, point in enumerate(points)
    ]

    return {  # noqa: S308 - numeric coordinates and escaped labels only
        "svg": mark_safe(svg),
        "points": hover_points,
        "width": OPS_WIDTH,
        "height": OPS_HEIGHT,
    }
