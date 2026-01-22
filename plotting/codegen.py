from __future__ import annotations

from typing import Any, Dict, List

from .spec import PlotSpec


def _b(x: Any) -> bool:
    return bool(x)


def _i(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _f(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _s(x: Any) -> str:
    return "" if x is None else str(x)


def _plot_type_slug(spec_dict: Dict[str, Any]) -> str:
    return _s(spec_dict.get("plot_type", "line")).strip().lower()


def _default_colour(raw: str, idx: int) -> str:
    t = _s(raw).strip()
    return t if t else ("C" + str(idx))


def _default_marker(raw: str) -> str:
    return _s(raw).strip()


def generate_plot_code(spec: PlotSpec) -> str:
    """
    Generate standalone plotting code as a string.
    - No imports from your project
    - No runtime branching on plot type (baked in)
    - User edits only CSV_PATHS (data)
    - No f-strings used in generated file
    """
    spec = spec.normalised()
    d = spec.to_dict()

    plot_type = _plot_type_slug(d)
    style = d.get("style", {}) or {}

    series_list = d.get("series", []) or []
    series_count = _i(d.get("series_count", len(series_list) or 1), 1)
    while len(series_list) < series_count:
        series_list.append({"label": "Series " + str(len(series_list) + 1), "style": {}, "inline": {"x_text": "", "y_text": ""}})
    series_list = series_list[:series_count]

    # Global style baked as constants
    font_family = _s(style.get("font_family", "")).strip()
    title = _s(style.get("title", ""))
    title_bold = _b(style.get("title_bold", False))
    title_italic = _b(style.get("title_italic", False))
    title_underline = _b(style.get("title_underline", False))
    x_label = _s(style.get("x_label", ""))
    y_label = _s(style.get("y_label", ""))
    show_grid = _b(style.get("show_grid", True))
    show_legend = _b(style.get("show_legend", True))
    base_font_size = _i(style.get("base_font_size", 11), 11)

    labels: List[str] = []
    colours: List[str] = []
    markers: List[str] = []
    marker_sizes: List[float] = []
    line_widths: List[float] = []
    line_styles: List[str] = []
    use_x_std: List[bool] = []
    use_y_std: List[bool] = []

    for idx, s in enumerate(series_list):
        labels.append(_s(s.get("label", "Series " + str(idx + 1))))
        sstyle = s.get("style", {}) or {}
        colours.append(_default_colour(_s(sstyle.get("color", "")), idx))
        markers.append(_default_marker(_s(sstyle.get("marker", ""))))
        marker_sizes.append(_f(sstyle.get("marker_size", 6.0), 6.0))
        line_widths.append(_f(sstyle.get("line_width", 1.6), 1.6))
        line_styles.append(_s(sstyle.get("line_style", "solid")).strip() or "solid")

        use_x_std.append(bool(s.get("use_x_std", False)))
        use_y_std.append(bool(s.get("use_y_std", False)))

    y_only = plot_type in ("hist", "box", "violin")

    out: List[str] = []
    out.append("from __future__ import annotations")
    out.append("")
    out.append("import csv")
    out.append("from pathlib import Path")
    out.append("from typing import List, Tuple")
    out.append("")
    out.append("import numpy as np")
    out.append("import matplotlib.pyplot as plt")
    if title_underline:
        out.append("from matplotlib.lines import Line2D")
    out.append("")

    out.append("CSV_PATHS: List[str] = [")
    for i in range(series_count):
        out.append("    'series_" + str(i + 1) + ".csv',")
    out.append("]")
    out.append("")
    out.append("OUT_PATH: str = 'figure.pdf'")
    out.append("DPI: int = 300")
    out.append("")

    out.append("PLOT_TYPE: str = " + repr(plot_type))
    out.append("LABELS: List[str] = " + repr(labels))
    out.append("COLOURS: List[str] = " + repr(colours))
    out.append("MARKERS: List[str] = " + repr(markers))
    out.append("MARKER_SIZES: List[float] = " + repr(marker_sizes))
    out.append("LINE_WIDTHS: List[float] = " + repr(line_widths))
    out.append("LINE_STYLES: List[str] = " + repr(line_styles))
    out.append("USE_X_STD: List[bool] = " + repr(use_x_std))
    out.append("USE_Y_STD: List[bool] = " + repr(use_y_std))
    out.append("")
    out.append("FONT_FAMILY: str = " + repr(font_family))
    out.append("BASE_FONT_SIZE: int = " + str(base_font_size))
    out.append("TITLE: str = " + repr(title))
    out.append("TITLE_BOLD: bool = " + ("True" if title_bold else "False"))
    out.append("TITLE_ITALIC: bool = " + ("True" if title_italic else "False"))
    out.append("X_LABEL: str = " + repr(x_label))
    out.append("Y_LABEL: str = " + repr(y_label))
    out.append("SHOW_GRID: bool = " + ("True" if show_grid else "False"))
    out.append("SHOW_LEGEND: bool = " + ("True" if show_legend else "False"))
    out.append("")

    out.append("def _is_float(s: str) -> bool:")
    out.append("    try:")
    out.append("        float(s)")
    out.append("        return True")
    out.append("    except Exception:")
    out.append("        return False")
    out.append("")
    out.append("def _read_csv_rows(path: Path) -> List[List[str]]:")
    out.append("    with path.open('r', newline='', encoding='utf-8') as f:")
    out.append("        reader = csv.reader(f)")
    out.append("        return [row for row in reader if row and any(cell.strip() for cell in row)]")
    out.append("")
    out.append("def _detect_header(rows: List[List[str]]) -> bool:")
    out.append("    first = rows[0]")
    out.append("    numeric = sum(1 for c in first if _is_float(c.strip()))")
    out.append("    return numeric < max(1, len(first) // 2)")
    out.append("")
    out.append("def _lower_strip(xs: List[str]) -> List[str]:")
    out.append("    return [x.strip().lower() for x in xs]")
    out.append("")
    out.append("def _col_index(headers: List[str], preferred: List[str], fallback: int) -> int:")
    out.append("    lower = _lower_strip(headers)")
    out.append("    for name in preferred:")
    out.append("        n = name.lower()")
    out.append("        if n in lower:")
    out.append("            return lower.index(n)")
    out.append("    return min(fallback, len(headers) - 1)")
    out.append("")
    out.append("def _col_as_floats(rows: List[List[str]], idx: int, path: Path) -> np.ndarray:")
    out.append("    out: List[float] = []")
    out.append("    for r_i, row in enumerate(rows):")
    out.append("        if idx >= len(row):")
    out.append("            continue")
    out.append("        raw = row[idx].strip()")
    out.append("        if raw == '':")
    out.append("            continue")
    out.append("        if not _is_float(raw):")
    out.append("            raise ValueError('Non-numeric value in ' + str(path) + ' at row ' + str(r_i + 1) + ': ' + repr(raw))")
    out.append("        out.append(float(raw))")
    out.append("    if not out:")
    out.append("        raise ValueError('No numeric values found in ' + str(path) + ' column ' + str(idx + 1))")
    out.append("    return np.asarray(out, dtype=float)")
    out.append("")

    if y_only:
        out.append("def load_series_values(csv_paths: List[str]) -> List[np.ndarray]:")
        out.append("    datasets: List[np.ndarray] = []")
        out.append("    for p in csv_paths:")
        out.append("        path = Path(p).expanduser().resolve()")
        out.append("        rows = _read_csv_rows(path)")
        out.append("        if not rows:")
        out.append("            raise ValueError('CSV is empty: ' + str(path))")
        out.append("        has_header = _detect_header(rows)")
        out.append("        if has_header:")
        out.append("            headers = rows[0]")
        out.append("            data_rows = rows[1:]")
        out.append("            y_idx = _col_index(headers, ['values', 'value', 'y'], 0)")
        out.append("        else:")
        out.append("            data_rows = rows")
        out.append("            y_idx = 0")
        out.append("        datasets.append(_col_as_floats(data_rows, y_idx, path))")
        out.append("    return datasets")
        out.append("")
    else:
        out.append("def load_series_xy(csv_paths: List[str]) -> Tuple[List[np.ndarray], List[np.ndarray], List[object], List[object]]:")
        out.append("    xs: List[np.ndarray] = []")
        out.append("    ys: List[np.ndarray] = []")
        out.append("    xstds: List[object] = []  # np.ndarray or None")
        out.append("    ystds: List[object] = []  # np.ndarray or None")
        out.append("    for i, p in enumerate(csv_paths):")
        out.append("        path = Path(p).expanduser().resolve()")
        out.append("        rows = _read_csv_rows(path)")
        out.append("        if not rows:")
        out.append("            raise ValueError('CSV is empty: ' + str(path))")
        out.append("        has_header = _detect_header(rows)")
        out.append("        if has_header:")
        out.append("            headers = rows[0]")
        out.append("            data_rows = rows[1:]")
        out.append("            x_idx = _col_index(headers, ['x'], 0)")
        out.append("            y_idx = _col_index(headers, ['y'], 1 if len(headers) > 1 else 0)")
        out.append("            xstd_idx = _col_index(headers, ['x_std', 'xerr', 'x_error', 'x_sd', 'x_sigma'], x_idx) if USE_X_STD[i] else -1")
        out.append("            ystd_idx = _col_index(headers, ['y_std', 'yerr', 'y_error', 'y_sd', 'y_sigma'], y_idx) if USE_Y_STD[i] else -1")
        out.append("        else:")
        out.append("            data_rows = rows")
        out.append("            x_idx, y_idx = 0, 1")
        out.append("            xstd_idx = 2 if USE_X_STD[i] else -1")
        out.append("            ystd_idx = 3 if USE_Y_STD[i] else -1")
        out.append("        x = _col_as_floats(data_rows, x_idx, path)")
        out.append("        y = _col_as_floats(data_rows, y_idx, path)")
        out.append("        if x.size != y.size:")
        out.append("            raise ValueError('Length mismatch in ' + str(path) + ': x has ' + str(x.size) + ', y has ' + str(y.size))")
        out.append("        xstd = None")
        out.append("        ystd = None")
        out.append("        if xstd_idx >= 0:")
        out.append("            xstd = _col_as_floats(data_rows, xstd_idx, path)")
        out.append("            if xstd.size != x.size:")
        out.append("                raise ValueError('Length mismatch in ' + str(path) + ': x_std has ' + str(xstd.size) + ', x has ' + str(x.size))")
        out.append("        if ystd_idx >= 0:")
        out.append("            ystd = _col_as_floats(data_rows, ystd_idx, path)")
        out.append("            if ystd.size != y.size:")
        out.append("                raise ValueError('Length mismatch in ' + str(path) + ': y_std has ' + str(ystd.size) + ', y has ' + str(y.size))")
        out.append("        xs.append(x)")
        out.append("        ys.append(y)")
        out.append("        xstds.append(xstd)")
        out.append("        ystds.append(ystd)")
        out.append("    return xs, ys, xstds, ystds")
        out.append("")

    if title_underline:
        out.append("def underline_title(ax, title_text) -> None:")
        out.append("    fig = ax.figure")
        out.append("    canvas = fig.canvas")
        out.append("    canvas.draw()")
        out.append("    renderer = canvas.get_renderer()")
        out.append("    bbox = title_text.get_window_extent(renderer=renderer)")
        out.append("    inv = fig.transFigure.inverted()")
        out.append("    (x0, y0) = inv.transform((bbox.x0, bbox.y0))")
        out.append("    (x1, _) = inv.transform((bbox.x1, bbox.y0))")
        out.append("    y = y0 - (2.0 / fig.bbox.height)")
        out.append("    lw = max(0.8, float(title_text.get_fontsize()) / 14.0)")
        out.append("    line = Line2D([x0, x1], [y, y], transform=fig.transFigure, color=title_text.get_color(), linewidth=lw)")
        out.append("    fig.add_artist(line)")
        out.append("")

    out.append("def apply_style(ax, legend=None) -> None:")
    out.append("    ax.tick_params(labelsize=BASE_FONT_SIZE)")
    out.append("    ax.set_xlabel(X_LABEL, fontfamily=(FONT_FAMILY or None), fontsize=BASE_FONT_SIZE)")
    out.append("    ax.set_ylabel(Y_LABEL, fontfamily=(FONT_FAMILY or None), fontsize=BASE_FONT_SIZE)")
    out.append("    ax.grid(SHOW_GRID)")
    out.append("    if FONT_FAMILY:")
    out.append("        for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):")
    out.append("            t.set_fontfamily(FONT_FAMILY)")
    out.append("    title_text = ax.set_title(TITLE)")
    out.append("    title_text.set_fontsize(BASE_FONT_SIZE + 2)")
    out.append("    if FONT_FAMILY:")
    out.append("        title_text.set_fontfamily(FONT_FAMILY)")
    out.append("    title_text.set_fontweight('bold' if TITLE_BOLD else 'normal')")
    out.append("    title_text.set_fontstyle('italic' if TITLE_ITALIC else 'normal')")
    if title_underline:
        out.append("    underline_title(ax, title_text)")
    out.append("    if legend is not None and FONT_FAMILY:")
    out.append("        for t in legend.get_texts():")
    out.append("            t.set_fontfamily(FONT_FAMILY)")
    out.append("            t.set_fontsize(BASE_FONT_SIZE)")
    out.append("")

    out.append("def main() -> None:")
    out.append("    fig, ax = plt.subplots()")

    if y_only:
        out.append("    datasets = load_series_values(CSV_PATHS)")
        if plot_type == "hist":
            out.append("    ax.hist(datasets, label=LABELS, color=COLOURS)")
            out.append("    legend = ax.legend() if SHOW_LEGEND else None")
            out.append("    apply_style(ax, legend=legend)")
        elif plot_type == "box":
            out.append("    ax.boxplot(datasets, labels=LABELS)")
            out.append("    apply_style(ax, legend=None)")
        else:
            out.append("    pos = np.arange(1, len(datasets) + 1)")
            out.append("    ax.violinplot(datasets, positions=pos, showmedians=True)")
            out.append("    ax.set_xticks(pos)")
            out.append("    ax.set_xticklabels(LABELS)")
            out.append("    apply_style(ax, legend=None)")
    else:
        out.append("    xs, ys, xstds, ystds = load_series_xy(CSV_PATHS)")

        if plot_type == "bar":
            out.append("    n = max(1, len(xs))")
            out.append("    width = 0.8 / n")
            out.append("    for i in range(len(xs)):")
            out.append("        offset = (i - (n - 1) / 2.0) * width")
            out.append("        kwargs = dict(width=width, label=LABELS[i], color=COLOURS[i])")
            out.append("        if ystds[i] is not None:")
            out.append("            kwargs['yerr'] = ystds[i]; kwargs['capsize'] = 3")
            out.append("        ax.bar(xs[i] + offset, ys[i], **kwargs)")

        elif plot_type == "area":
            out.append("    for i in range(len(xs)):")
            out.append("        ax.fill_between(xs[i], ys[i], 0, label=LABELS[i], color=COLOURS[i])")

        elif plot_type == "scatter":
            out.append("    for i in range(len(xs)):")
            out.append("        m = MARKERS[i] if MARKERS[i] else None")
            out.append("        if xstds[i] is not None or ystds[i] is not None:")
            out.append("            ax.errorbar(xs[i], ys[i], xerr=xstds[i], yerr=ystds[i], label=LABELS[i], color=COLOURS[i], ecolor=COLOURS[i], linestyle='none', marker=(m or None), markersize=max(1.0, MARKER_SIZES[i]), capsize=3)")
            out.append("        else:")
            out.append("            ax.scatter(xs[i], ys[i], label=LABELS[i], c=COLOURS[i], marker=m, s=max(1.0, MARKER_SIZES[i]) ** 2)")

        elif plot_type == "step":
            out.append("    for i in range(len(xs)):")
            out.append("        m = MARKERS[i] if MARKERS[i] else None")
            out.append("        ax.step(xs[i], ys[i], label=LABELS[i], color=COLOURS[i], linestyle=LINE_STYLES[i], linewidth=max(0.1, LINE_WIDTHS[i]), marker=m, markersize=max(1.0, MARKER_SIZES[i]))")
            out.append("        if xstds[i] is not None or ystds[i] is not None:")
            out.append("            ax.errorbar(xs[i], ys[i], xerr=xstds[i], yerr=ystds[i], fmt='none', ecolor=COLOURS[i], capsize=3)")

        else:
            out.append("    for i in range(len(xs)):")
            out.append("        m = MARKERS[i] if MARKERS[i] else None")
            out.append("        if xstds[i] is not None or ystds[i] is not None:")
            out.append("            ax.errorbar(xs[i], ys[i], xerr=xstds[i], yerr=ystds[i], label=LABELS[i], color=COLOURS[i], ecolor=COLOURS[i], linestyle=LINE_STYLES[i], linewidth=max(0.1, LINE_WIDTHS[i]), marker=(m or None), markersize=max(1.0, MARKER_SIZES[i]), capsize=3)")
            out.append("        else:")
            out.append("            ax.plot(xs[i], ys[i], label=LABELS[i], color=COLOURS[i], linestyle=LINE_STYLES[i], linewidth=max(0.1, LINE_WIDTHS[i]), marker=m, markersize=max(1.0, MARKER_SIZES[i]))")

        out.append("    legend = ax.legend() if SHOW_LEGEND else None")
        out.append("    apply_style(ax, legend=legend)")

    out.append("    fig.savefig(OUT_PATH, dpi=DPI, bbox_inches='tight')")
    out.append("    plt.show()")
    out.append("")
    out.append("if __name__ == '__main__':")
    out.append("    main()")

    return "\n".join(out)
