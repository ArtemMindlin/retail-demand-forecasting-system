/* Hover interaction for the server-rendered forecast chart.
 *
 * The chart geometry is computed in Python and baked into the SVG. All this
 * does is map the pointer to the nearest data point and move the crosshair and
 * tooltip that the server already emitted. Event delegation on document means
 * it keeps working after htmx swaps the chart in.
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
    var offsetX = event.clientX - rect.left;
    var point = points[nearestIndex(points, (offsetX / rect.width) * viewWidth)];

    // Crosshair lives in SVG user space.
    hover.style.display = "";
    hover.querySelector(".chart-hover-line").setAttribute("x1", point.x);
    hover.querySelector(".chart-hover-line").setAttribute("x2", point.x);

    var predDot = hover.querySelector(".chart-hover-pred");
    predDot.setAttribute("cx", point.x);
    predDot.setAttribute("cy", point.yPred);

    var actualDot = hover.querySelector(".chart-hover-actual");
    if (point.yActual === null) {
      actualDot.style.display = "none";
    } else {
      actualDot.style.display = "";
      actualDot.setAttribute("cx", point.x);
      actualDot.setAttribute("cy", point.yActual);
    }

    // Tooltip lives in CSS pixel space.
    var pixelX = (point.x / viewWidth) * rect.width;
    var pixelY = (point.yPred / 320) * rect.height;
    tooltip.hidden = false;
    tooltip.querySelector(".chart-tooltip-label").textContent = point.label;
    tooltip.querySelector('[data-field="actual"]').textContent =
      point.actual === null || point.actual === undefined ? "—" : point.actual;
    tooltip.querySelector('[data-field="predicted"]').textContent = point.predicted;
    tooltip.querySelector('[data-field="interval"]').textContent =
      point.lower + "–" + point.upper;

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
