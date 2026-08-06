/* Render every [data-latex] element with KaTeX.
 *
 * The formulas are LaTeX source emitted by the server; KaTeX turns them into
 * markup in the browser. Runs on load and again after each htmx swap, so
 * formulas inside freshly-swapped modals get typeset too.
 */
(function () {
  "use strict";

  function renderAll(root) {
    if (typeof window.katex === "undefined") return;
    var nodes = (root || document).querySelectorAll("[data-latex]:not([data-latex-done])");
    for (var i = 0; i < nodes.length; i++) {
      var node = nodes[i];
      try {
        window.katex.render(node.dataset.latex, node, {
          displayMode: node.dataset.display === "block",
          throwOnError: false,
        });
      } catch (err) {
        node.textContent = node.dataset.latex;
      }
      node.setAttribute("data-latex-done", "");
    }
  }

  document.addEventListener("DOMContentLoaded", function () {
    renderAll(document);
  });

  document.body.addEventListener("htmx:afterSwap", function (event) {
    renderAll(event.target);
  });
})();
