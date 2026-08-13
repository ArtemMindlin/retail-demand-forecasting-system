/* Hover interaction for the server-rendered charts (forecast and OPS backtest).
 *
 * The chart geometry is computed in Python and baked into the SVG. All this
 * does is map the pointer to the nearest data point and move the crosshair and
 * tooltip that the server already emitted. Event delegation on document means
 * it keeps working after htmx swaps the chart in.
 *
 * Chart-agnostic by contract: every [data-field="x"] in the tooltip is filled from
 * the point's "x" property, already formatted server-side, and the crosshair dots
 * are optional. A chart opts in by wrapping its SVG in .chart-canvas with
 * data-chart-points / data-chart-width / data-chart-height.
 */
(function () {
  "use strict";

  function nearestIndex(points, userX) {
    var best = 0;
    var bestDistance = Infinity;
    for (var i = 0; i < points.length; i++) {
      var distance = Math.abs(points[i].x - userX);
      if (distance < bestDistance) {
        bestDistance = distance;
        best = i;
      }
    }
    return best;
  }

  function readPoints(canvas) {
    if (canvas._chartPoints) return canvas._chartPoints;
    try {
      canvas._chartPoints = JSON.parse(canvas.dataset.chartPoints || "[]");
    } catch (err) {
      canvas._chartPoints = [];
    }
    return canvas._chartPoints;
  }

  /* A chart may omit either dot (the OPS chart has no separate "actual" marker: its
     realized value is already a permanent dot), and a point may have no actual yet. */
  function moveDot(dot, x, y) {
    if (!dot) return;
    if (y === null || y === undefined) {
      dot.style.display = "none";
      return;
    }
    dot.style.display = "";
    dot.setAttribute("cx", x);
    dot.setAttribute("cy", y);
  }

  function show(canvas, event) {
    var points = readPoints(canvas);
    if (!points.length) return;

    var svg = canvas.querySelector(".forecast-svg");
    var hover = canvas.querySelector(".chart-hover");
    var tooltip = canvas.querySelector(".chart-tooltip");
    if (!svg || !hover || !tooltip) return;

    var rect = svg.getBoundingClientRect();
    if (!rect.width) return;

    var viewWidth = Number(canvas.dataset.chartWidth) || 900;
    var viewHeight = Number(canvas.dataset.chartHeight) || 230;
    var offsetX = event.clientX - rect.left;
    var point = points[nearestIndex(points, (offsetX / rect.width) * viewWidth)];

    // Crosshair lives in SVG user space.
    hover.style.display = "";
    hover.querySelector(".chart-hover-line").setAttribute("x1", point.x);
    hover.querySelector(".chart-hover-line").setAttribute("x2", point.x);

    moveDot(hover.querySelector(".chart-hover-pred"), point.x, point.yPred);
    moveDot(hover.querySelector(".chart-hover-actual"), point.x, point.yActual);

    // Tooltip lives in CSS pixel space.
    var pixelX = (point.x / viewWidth) * rect.width;
    var pixelY = (point.yPred / viewHeight) * rect.height;
    tooltip.hidden = false;
    tooltip.querySelector(".chart-tooltip-label").textContent = point.label;
    var fields = tooltip.querySelectorAll("[data-field]");
    for (var f = 0; f < fields.length; f++) {
      var value = point[fields[f].dataset.field];
      fields[f].textContent = value === null || value === undefined ? "—" : value;
    }

    var flip = pixelX > rect.width - 200;
    tooltip.style.left = (flip ? pixelX - 190 : pixelX + 14) + "px";
    tooltip.style.top = Math.max(8, pixelY - 60) + "px";
  }

  function hide(canvas) {
    var hover = canvas.querySelector(".chart-hover");
    var tooltip = canvas.querySelector(".chart-tooltip");
    if (hover) hover.style.display = "none";
    if (tooltip) tooltip.hidden = true;
  }

  document.addEventListener("pointermove", function (event) {
    var canvas = event.target.closest(".chart-canvas");
    if (canvas) show(canvas, event);
  });

  document.addEventListener("pointerleave", function (event) {
    var canvas = event.target.closest && event.target.closest(".chart-canvas");
    if (canvas) hide(canvas);
  }, true);
})();
