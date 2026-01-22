from __future__ import annotations

from pathlib import Path

from .spec import PlotSpec
from .codegen import generate_plot_code


def export_code_scaffold(spec: PlotSpec, out_dir: str) -> str:
    """
    Writes ONE file:
      out_dir/plot_code.txt

    The output file is standalone plotting code (no project imports).
    The only thing the user edits is CSV_PATHS.
    """
    spec = spec.normalised()
    target = Path(out_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)

    code = generate_plot_code(spec)
    out_path = target / "plot_code.py"
    out_path.write_text(code, encoding="utf-8")

    return str(out_path)
