"""Dump every dashboard view to HTML, as a safety net for markup and CSS refactors.

There is no test that looks at the rendered page as a whole, so a refactor of the
stylesheets or the templates has nothing to check itself against. This writes one
HTML file per view (plus a copy of ``static/``) under ``tmp/render/``, which gives
two things:

- a **CSS-only** change must leave every snapshot byte-identical. ``diff -r`` on two
  runs is then a proof that no markup moved.
- a **markup** change produces a readable diff of exactly what moved.

The output is also a servable site, so the same snapshots can be opened in a browser
for the visual half of the check:

    uv run python scripts/render_snapshots.py --out tmp/render/after
    python -m http.server 8123 -d tmp/render/after

Auth is done programmatically with a throwaway password set in this process only, so
running this never touches the real dashboard credentials.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path

SNAPSHOT_PASSWORD = "render-snapshot-only"  # noqa: S105 - throwaway, this process only
USERNAME = "snapshot"

# Every view, plus the fragments that are swapped in over them. Overlay fragments are
# included because they carry markup of their own that no other check renders.
PAGES: tuple[tuple[str, str], ...] = (
    ("/dashboard/", "dashboard.html"),
    ("/ops/", "ops.html"),
    ("/skus/", "skus.html"),
    ("/drift/", "drift.html"),
    ("/eda/", "eda.html"),
    ("/latent/", "latent.html"),
    ("/pareto/", "pareto.html"),
    ("/api/", "api_docs.html"),
    ("/alertas/", "_alerts_panel.html"),
    ("/configuracion/", "_config_modal.html"),
    ("/dashboard/modulo/conformal/", "_academic_modal.html"),
    ("/pipeline/status/", "_pipeline_console.html"),
)


def _normalise(html: str) -> str:
    """Blank out values that change on every run, so ``diff`` means something.

    The CSRF token is per-session and appears in the body tag and in every form, so
    without this every snapshot differs from every other one and the whole harness is
    noise.
    """
    html = re.sub(r'(name="csrfmiddlewaretoken" value=")[^"]*', r"\g<1>CSRF", html)
    return re.sub(r'("X-CSRFToken": ")[^"]*', r"\g<1>CSRF", html)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("tmp/render"))
    parser.add_argument(
        "--no-static",
        action="store_true",
        help="Skip copying static/, when only the HTML diff matters.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "retail_forecasting.api.settings")
    os.environ["AUTH_USERNAME"] = USERNAME
    os.environ["AUTH_PASSWORD"] = SNAPSHOT_PASSWORD
    os.environ["DJANGO_DEBUG"] = "true"
    # No artifact directory to point at: the views discover runs through the MLflow store,
    # so what they render is whatever `RETAIL_MLFLOW_*` resolves to for this process.

    import django

    django.setup()

    from django.test import Client

    client = Client()
    response = client.post("/login/", {"username": USERNAME, "password": SNAPSHOT_PASSWORD})
    if response.status_code != 302:
        raise SystemExit("Snapshot login failed; the auth view or its settings changed.")

    args.out.mkdir(parents=True, exist_ok=True)
    for url, name in PAGES:
        page = client.get(url)
        (args.out / name).write_text(_normalise(page.content.decode()), encoding="utf-8")
        print(f"  {page.status_code}  {url:34} -> {name}")

    if not args.no_static:
        static_root = Path("src/retail_forecasting/api/static")
        target = args.out / "static"
        shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(static_root, target)

    print(f"✅ {len(PAGES)} snapshots in {args.out}")


if __name__ == "__main__":
    main()
