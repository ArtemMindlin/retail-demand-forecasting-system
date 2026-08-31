"""Reject a math spacing macro immediately followed by `\\%` in the memoria's chapters.

Inside `$...$`, `\\,` `\\:` `\\;` `\\!` produce glue in `mu` units, while `\\%` under this
preamble (babel spanish + T1) contributes glue in ordinary units. TeX cannot add the two
and aborts the whole build with "Incompatible glue units", so `$80\\,\\%$` compiles nowhere
while the same percentage written in text mode, `80\\,\\%`, is fine.

The rule is narrow, and the narrowness is the point: `\\,` on its own in math is perfectly
legal (`$[L_{s,t},\\, U_{s,t}]$` builds), and so is `\\%` on its own (`$80\\%$` builds). Only
the juxtaposition fails, which is why a broader check produced four false positives on
chapters that compile. Measured against tectonic with this document's preamble:

    $80\\,\\%$    fails        $80\\%$              builds
    $80\\;\\%$    fails        $80\\,x$             builds
    $80\\!\\%$    fails        $80\\quad\\%$        builds  (em units, not mu)
                             $80\\,\\text{\\%}$    builds  (leaves math mode)

Worth a hook because nothing else looks at LaTeX: ruff and mypy never open a `.tex`, so a
commit can be green and still leave the thesis unbuildable until someone runs `make pdf`.

Usage: scripts/check_latex_math_spacing.py <ficheros .tex>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# The mu-unit spacing macros, immediately before `\%`. `\quad`/`\qquad` are em-based and
# combine fine, so they are deliberately absent.
_OFFENDER = re.compile(r"\\[,:;!]\s*\\%")


def offending_lines(text: str) -> list[tuple[int, str]]:
    """Line number and content for every line that hits the pattern inside math mode."""
    found: list[tuple[int, str]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        # Odd-indexed segments sit inside math mode. An escaped `\$` is not a delimiter, so
        # it is neutralised before splitting.
        segments = line.replace("\\$", "\x00").split("$")
        if any(_OFFENDER.search(segment) for segment in segments[1::2]):
            found.append((number, line.strip()))
    return found


def main(paths: list[str]) -> int:
    failures = 0
    for path in paths:
        file = Path(path)
        if file.suffix != ".tex" or not file.is_file():
            continue
        for number, line in offending_lines(file.read_text(encoding="utf-8")):
            failures += 1
            excerpt = line if len(line) <= 110 else f"{line[:107]}..."
            print(f"{file}:{number}: espacio fino antes de \\% en modo matemático\n    {excerpt}")
    if failures:
        print(
            f"\n{failures} caso(s). Rompen `make pdf` con 'Incompatible glue units'. "
            "Escribe el porcentaje fuera de $...$: `80\\,\\%`, no `$80\\,\\%$`."
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
