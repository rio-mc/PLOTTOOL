from __future__ import annotations

from dataclasses import dataclass, fields
from typing import List, Optional
import math

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator
import matplotlib.colors as mcolors

from .spec import PlotSpec, PlotType, SeriesStyleSpec, SeriesInputMode
from .parsing import (
    ParseIssue,
    parse_inline_vector,
    validate_xy_lengths,
    parse_table_with_header,
    table_to_columns,
    col_as_float,
)
from .sample_data import SeriesData, make_sample_series
from .plot_types import meta_for

def _compute_outlier_mask(y: np.ndarray, sigma: float, method: str = "std") -> np.ndarray:
    y = np.asarray(y, dtype=float)
    mask = np.zeros(y.shape, dtype=bool)

    finite = np.isfinite(y)
    if not finite.any():
        return mask

    yf = y[finite]
    k = float(sigma)
    method = (method or "std").strip().lower()

    if method == "mad":
        med = float(np.median(yf))
        mad = float(np.median(np.abs(yf - med)))
        if not np.isfinite(mad) or mad <= 0.0:
            return mask
        robust_sd = 1.4826 * mad
        mask[finite] = np.abs(yf - med) > (k * robust_sd)
        return mask

    mu = float(np.mean(yf))
    sd = float(np.std(yf))
    if not np.isfinite(sd) or sd <= 0.0:
        return mask
    mask[finite] = np.abs(yf - mu) > (k * sd)
    return mask

def _inject_single_outlier(y: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    y = np.asarray(y, dtype=float).copy()
    finite = np.isfinite(y)
    if not finite.any():
        return y
    yf = y[finite]
    mu = float(np.mean(yf))
    sd = float(np.std(yf))
    if not np.isfinite(sd) or sd <= 0:
        # fallback: just bump the largest finite value
        idx = np.where(finite)[0][-1]
        y[idx] = y[idx] + 10.0
        return y

    # pick a stable index to spike (last finite)
    idx = np.where(finite)[0][-1]
    y[idx] = mu + (sigma * 4.0) * sd  # comfortably beyond threshold
    return y

@dataclass(frozen=True)
class BuildResult:
    series: List[SeriesData]
    series_styles: List[SeriesStyleSpec]
    issues: List[ParseIssue]


_SERIESDATA_FIELDS = {f.name for f in fields(SeriesData)}
_SERIESSTYLE_FIELDS = {f.name for f in fields(SeriesStyleSpec)}


def _series_data(**kwargs) -> SeriesData:
    filtered = {k: v for k, v in kwargs.items() if k in _SERIESDATA_FIELDS}
    return SeriesData(**filtered)


def _series_style(**kwargs) -> SeriesStyleSpec:
    filtered = {k: v for k, v in kwargs.items() if k in _SERIESSTYLE_FIELDS}
    return SeriesStyleSpec(**filtered)

def build_series_data(spec: PlotSpec) -> BuildResult:
    spec = spec.normalised()
    issues: List[ParseIssue] = []
    out_series: List[SeriesData] = []
    out_styles: List[SeriesStyleSpec] = []

    m = meta_for(spec.plot_type)
    sigma = float(getattr(spec.style, "outlier_sigma", 3.0))
    outlier_method = str(getattr(spec.style, "outlier_method", "std") or "std")

    requires_z = bool(getattr(m, "requires_z", False))

    def series_highlight(sty: SeriesStyleSpec) -> bool:
        return bool(getattr(sty, "highlight_outliers", False))

    # --- If user has no data: sample behaviour (optionally inject a demo outlier) ---
    if not spec.you_got_data:
        sample = make_sample_series(spec.series_count, is_3d=requires_z)
        for i, sdata in enumerate(sample):
            label = spec.series[i].label if i < len(spec.series) else sdata.label
            st = spec.series[i].style if i < len(spec.series) else SeriesStyleSpec()

            yvals = sdata.y
            if series_highlight(st):
                yvals = _inject_single_outlier(yvals, sigma)

            out_mask = _compute_outlier_mask(yvals, sigma, outlier_method) if series_highlight(st) else None

            out_series.append(
                _series_data(
                    x=sdata.x,
                    y=yvals,
                    z=sdata.z,
                    label=label,
                    x_std=None,
                    y_std=None,
                    outlier_mask=out_mask,
                )
            )
            out_styles.append(st)

        return BuildResult(series=out_series, series_styles=out_styles, issues=[])

    # --- User has data ---
    color_cycle = [f"C{i}" for i in range(10)]
    cycle_idx = 0

    for idx, s in enumerate(spec.series):
        # TABLE MODE (supports grouping)
        if s.input_mode == SeriesInputMode.TABLE:
            header, rows, err = parse_table_with_header(s.inline.table_text)
            if err:
                issues.append(ParseIssue(message=err, series_index=idx, axis="table"))
                fb = make_sample_series(1, seed=400 + idx, is_3d=requires_z)[0]

                yvals = fb.y
                if series_highlight(s.style):
                    yvals = _inject_single_outlier(yvals, sigma)

                out_mask = _compute_outlier_mask(yvals, sigma, outlier_method) if series_highlight(s.style) else None

                out_series.append(
                    _series_data(
                        x=fb.x,
                        y=yvals,
                        z=fb.z,
                        label=s.label or f"Series {idx + 1}",
                        x_std=None,
                        y_std=None,
                        outlier_mask=out_mask,
                    )
                )
                out_styles.append(s.style)
                continue

            cols = table_to_columns(header, rows)

            def get_col(name: str) -> Optional[List[str]]:
                if name in cols:
                    return cols[name]
                low = {k.lower(): k for k in cols.keys()}
                if name.lower() in low:
                    return cols[low[name.lower()]]
                return None

            x_col = get_col(s.inline.x_col)
            y_col = get_col(s.inline.y_col)
            z_col = get_col(s.inline.z_col) if requires_z else None
            gx_col = get_col(s.inline.group_col) if s.split_by_group else None

            if x_col is None and m.requires_x:
                issues.append(ParseIssue(message=f"Missing x column '{s.inline.x_col}'.", series_index=idx, axis="table"))
                continue
            if y_col is None:
                issues.append(ParseIssue(message=f"Missing y column '{s.inline.y_col}'.", series_index=idx, axis="table"))
                continue
            if requires_z and z_col is None:
                issues.append(ParseIssue(message=f"Missing z column '{s.inline.z_col}'.", series_index=idx, axis="table"))
                continue

            x, errx = col_as_float(x_col or [], s.inline.x_col) if m.requires_x else (np.asarray([], dtype=float), None)
            y, erry = col_as_float(y_col or [], s.inline.y_col)
            if errx:
                issues.append(ParseIssue(message=errx, series_index=idx, axis="x"))
                continue
            if erry:
                issues.append(ParseIssue(message=erry, series_index=idx, axis="y"))
                continue

            z = None
            if requires_z:
                z_arr, errz = col_as_float(z_col or [], s.inline.z_col)
                if errz:
                    issues.append(ParseIssue(message=errz, series_index=idx, axis="z"))
                    continue
                z = z_arr

            # std columns (optional)
            x_std = None
            y_std = None
            if bool(getattr(m, "supports_x_std", False)) and s.use_x_std:
                xstd_col = get_col(s.inline.x_std_col)
                if xstd_col is None:
                    issues.append(ParseIssue(message=f"Missing x std column '{s.inline.x_std_col}'.", series_index=idx, axis="x std"))
                else:
                    arr, e = col_as_float(xstd_col, s.inline.x_std_col)
                    if e:
                        issues.append(ParseIssue(message=e, series_index=idx, axis="x std"))
                    else:
                        x_std = arr

            if bool(getattr(m, "supports_y_std", False)) and s.use_y_std:
                ystd_col = get_col(s.inline.y_std_col)
                if ystd_col is None:
                    issues.append(ParseIssue(message=f"Missing y std column '{s.inline.y_std_col}'.", series_index=idx, axis="y std"))
                else:
                    arr, e = col_as_float(ystd_col, s.inline.y_std_col)
                    if e:
                        issues.append(ParseIssue(message=e, series_index=idx, axis="y std"))
                    else:
                        y_std = arr

            # Basic length checks
            if m.requires_x and x.size != y.size:
                issues.append(ParseIssue(message=f"x and y must have the same length ({x.size} vs {y.size}).", series_index=idx, axis="xy"))
                continue
            if z is not None and z.size != y.size:
                issues.append(ParseIssue(message=f"z and y must have the same length ({z.size} vs {y.size}).", series_index=idx, axis="zy"))
                continue
            if x_std is not None and m.requires_x and x_std.size != x.size:
                issues.append(ParseIssue(message=f"x std must match x length ({x_std.size} vs {x.size}).", series_index=idx, axis="x std"))
                x_std = None
            if y_std is not None and y_std.size != y.size:
                issues.append(ParseIssue(message=f"y std must match y length ({y_std.size} vs {y.size}).", series_index=idx, axis="y std"))
                y_std = None

            # --- Group split ---
            if s.split_by_group:
                if gx_col is None:
                    issues.append(ParseIssue(message=f"Missing group column '{s.inline.group_col}'.", series_index=idx, axis="group"))
                    continue

                groups = [str(g).strip() if g is not None else "" for g in gx_col]
                if len(groups) != y.size:
                    issues.append(ParseIssue(message=f"group column length must match y length ({len(groups)} vs {y.size}).", series_index=idx, axis="group"))
                    continue

                explicit = _parse_group_order(s.inline.group_order_text)
                unique: List[str] = []
                seen: set[str] = set()
                for g in groups:
                    if g not in seen:
                        seen.add(g)
                        unique.append(g)
                ordered = [g for g in explicit if g in seen] + [g for g in unique if g not in explicit]

                for g in ordered:
                    mask = np.asarray([gg == g for gg in groups], dtype=bool)
                    if not mask.any():
                        continue

                    label_prefix = (s.group_label_prefix.strip() + ": ") if s.group_label_prefix.strip() else ""
                    group_label = f"{label_prefix}{g}"

                    st = s.style
                    color = st.color.strip()
                    if not color and s.group_color_by_cycle:
                        color = color_cycle[cycle_idx % len(color_cycle)]
                        cycle_idx += 1

                    out_styles.append(
                        _series_style(
                            color=color,
                            marker=st.marker,
                            marker_size=st.marker_size,
                            line_width=st.line_width,
                            line_style=st.line_style,
                            highlight_outliers=bool(getattr(st, "highlight_outliers", False)),
                        )
                    )

                    yg = y[mask]
                    out_mask = _compute_outlier_mask(yg, sigma, outlier_method) if series_highlight(st) else None

                    out_series.append(
                        _series_data(
                            x=x[mask] if m.requires_x else np.arange(mask.sum(), dtype=float),
                            y=yg,
                            z=(z[mask] if z is not None else None),
                            label=group_label,
                            x_std=(x_std[mask] if (x_std is not None and m.requires_x) else None),
                            y_std=(y_std[mask] if y_std is not None else None),
                            outlier_mask=out_mask,
                        )
                    )
                continue

            # Not split: single series (user data, do not inject)
            out_mask = _compute_outlier_mask(y, sigma, outlier_method) if series_highlight(s.style) else None
            out_series.append(
                _series_data(
                    x=x if m.requires_x else np.arange(y.size, dtype=float),
                    y=y,
                    z=z,
                    label=s.label or f"Series {idx + 1}",
                    x_std=x_std,
                    y_std=y_std,
                    outlier_mask=out_mask,
                )
            )
            out_styles.append(s.style)
            continue

        # INLINE MODE
        y, err_y = parse_inline_vector(s.inline.y_text)
        if err_y:
            issues.append(ParseIssue(message=err_y, series_index=idx, axis="y"))

        x_std = None
        y_std = None
        z = None

        if m.requires_x:
            x, err_x = parse_inline_vector(s.inline.x_text)
            if err_x:
                issues.append(ParseIssue(message=err_x, series_index=idx, axis="x"))

            len_err = validate_xy_lengths(x, y) if m.requires_xy else None
            if len_err:
                issues.append(ParseIssue(message=len_err, series_index=idx, axis="xy"))

            # z inline
            err_z = None
            if requires_z:
                z_arr, err_z = parse_inline_vector(getattr(s.inline, "z_text", ""))
                if err_z:
                    issues.append(ParseIssue(message=err_z, series_index=idx, axis="z"))
                else:
                    if z_arr.size != x.size:
                        issues.append(ParseIssue(message=f"z must match x length ({z_arr.size} vs {x.size}).", series_index=idx, axis="z"))
                        err_z = "z length mismatch"
                    else:
                        z = z_arr

            std_err = False
            if bool(getattr(m, "supports_x_std", False)) and s.use_x_std:
                xstd_arr, err_xstd = parse_inline_vector(getattr(s.inline, "x_std_text", ""))
                if err_xstd:
                    issues.append(ParseIssue(message=err_xstd, series_index=idx, axis="x std"))
                    std_err = True
                elif xstd_arr.size != x.size:
                    issues.append(ParseIssue(message=f"x std must match x length ({xstd_arr.size} vs {x.size}).", series_index=idx, axis="x std"))
                    std_err = True
                else:
                    x_std = xstd_arr

            if bool(getattr(m, "supports_y_std", False)) and s.use_y_std:
                ystd_arr, err_ystd = parse_inline_vector(getattr(s.inline, "y_std_text", ""))
                if err_ystd:
                    issues.append(ParseIssue(message=err_ystd, series_index=idx, axis="y std"))
                    std_err = True
                elif ystd_arr.size != y.size:
                    issues.append(ParseIssue(message=f"y std must match y length ({ystd_arr.size} vs {y.size}).", series_index=idx, axis="y std"))
                    std_err = True
                else:
                    y_std = ystd_arr

            if err_x or err_y or len_err or std_err or err_z:
                fb = make_sample_series(1, seed=100 + idx, is_3d=requires_z)[0]

                yvals = fb.y
                if series_highlight(s.style):
                    yvals = _inject_single_outlier(yvals, sigma)

                out_mask = _compute_outlier_mask(yvals, sigma, outlier_method) if series_highlight(s.style) else None

                out_series.append(
                    _series_data(
                        x=fb.x,
                        y=yvals,
                        z=fb.z,
                        label=s.label or f"Series {idx + 1}",
                        x_std=None,
                        y_std=None,
                        outlier_mask=out_mask,
                    )
                )
                out_styles.append(s.style)
            else:
                out_mask = _compute_outlier_mask(y, sigma, outlier_method) if series_highlight(s.style) else None
                out_series.append(
                    _series_data(
                        x=x,
                        y=y,
                        z=z,
                        label=s.label or f"Series {idx + 1}",
                        x_std=x_std,
                        y_std=y_std,
                        outlier_mask=out_mask,
                    )
                )
                out_styles.append(s.style)

        else:
            # y-only plots
            if y.size == 0:
                issues.append(ParseIssue(message="Values must contain at least one number.", series_index=idx, axis="y"))
                fb = make_sample_series(1, seed=200 + idx)[0]

                yvals = fb.y
                if series_highlight(s.style):
                    yvals = _inject_single_outlier(yvals, sigma)

                out_mask = _compute_outlier_mask(yvals, sigma, outlier_method) if series_highlight(s.style) else None

                out_series.append(
                    _series_data(
                        x=fb.x,
                        y=yvals,
                        z=None,
                        label=s.label or f"Series {idx + 1}",
                        x_std=None,
                        y_std=None,
                        outlier_mask=out_mask,
                    )
                )
                out_styles.append(s.style)
            else:
                out_mask = _compute_outlier_mask(y, sigma, outlier_method) if series_highlight(s.style) else None
                out_series.append(
                    _series_data(
                        x=np.arange(y.size, dtype=float),
                        y=y,
                        z=None,
                        label=s.label or f"Series {idx + 1}",
                        x_std=None,
                        y_std=None,
                        outlier_mask=out_mask,
                    )
                )
                out_styles.append(s.style)

    return BuildResult(series=out_series, series_styles=out_styles, issues=issues)

def _apply_font_family_to_ticks(ax, font_family: str) -> None:
    if not font_family:
        return
    for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        t.set_fontfamily(font_family)


def _apply_font_family_to_legend(legend, font_family: str, base_size: int) -> None:
    if not legend or not font_family:
        return
    for t in legend.get_texts():
        t.set_fontfamily(font_family)
        t.set_fontsize(base_size)


def _set_tick_label_props(tick, base_size: int, font_family: str) -> None:
    """
    Best-effort tick label styling across Matplotlib versions.

    Matplotlib versions differ:
    - older: tick.label
    - newer: tick.label1 / tick.label2
    """
    for attr in ("label1", "label2", "label", "_label"):
        t = getattr(tick, attr, None)
        if t is None:
            continue
        try:
            t.set_fontsize(base_size)
        except Exception:
            pass
        if font_family:
            try:
                t.set_fontfamily(font_family)
            except Exception:
                pass


def _configure_minor_ticks(ax, enable: bool) -> None:
    if enable:
        try:
            ax.minorticks_on()
            return
        except Exception:
            pass
        try:
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
        except Exception:
            pass
    else:
        try:
            ax.minorticks_off()
            return
        except Exception:
            pass
        try:
            ax.xaxis.set_minor_locator(None)
            ax.yaxis.set_minor_locator(None)
        except Exception:
            pass

def apply_style(ax, spec: PlotSpec, legend=None) -> None:
    style = spec.style
    font_family = (style.font_family or "").strip()

    base = int(getattr(style, "base_font_size", 11))
    title_size = int(getattr(style, "title_font_size", base + 2))

    # ---- title ----
    title_str = style.title if style.title else ("Figure title" if not spec.you_got_data else "")
    title_text = ax.set_title(title_str)

    title_text.set_fontsize(title_size)
    if font_family:
        title_text.set_fontfamily(font_family)

    title_text.set_fontweight("bold" if style.title_bold else "normal")
    title_text.set_fontstyle("italic" if style.title_italic else "normal")

    if style.title_offset is not None:
        title_text.set_y(1.0 + style.title_offset)

    if style.title_underline and title_text.get_text():
        _manual_underline_title(ax, title_text)
    else:
        fig = ax.figure
        old = getattr(fig, "_plotting_title_underline", None)
        if old is not None:
            try:
                old.remove()
            except Exception:
                pass
            fig._plotting_title_underline = None

    try:
        title_text.set_underline(bool(style.title_underline))
    except Exception:
        pass

    # ---- axis labels ----
    if style.x_label:
        ax.set_xlabel(style.x_label, fontsize=base, fontfamily=(font_family or None))
    if style.y_label:
        ax.set_ylabel(style.y_label, fontsize=base, fontfamily=(font_family or None))

    # ---- 3D z label / ticks ----
    if hasattr(ax, "zaxis"):
        if style.z_label:
            ax.set_zlabel(style.z_label, fontsize=base, fontfamily=(font_family or None))
        try:
            ax.zaxis.label.set_size(base)
        except Exception:
            pass
        if font_family:
            try:
                ax.zaxis.label.set_family(font_family)
            except Exception:
                pass

        for tick in ax.zaxis.get_major_ticks():
            _set_tick_label_props(tick, base, font_family)

    # ---- ticks ----
    ax.tick_params(labelsize=base)
    _apply_font_family_to_ticks(ax, font_family)

    # ---- minor ticks + grid ----
    is_3d = hasattr(ax, "zaxis")

    if is_3d:
        # mplot3d doesn't reliably distinguish major/minor grid; treat minor toggles as no-ops
        try:
            ax.grid(bool(style.show_grid))
        except Exception:
            pass
    else:
        enable_minor = bool(getattr(style, "show_minor_ticks", False) or getattr(style, "show_minor_grid", False))
        _configure_minor_ticks(ax, enable_minor)

        try:
            ax.grid(bool(style.show_grid), which="major")
        except Exception:
            ax.grid(bool(style.show_grid))

        try:
            ax.grid(bool(getattr(style, "show_minor_grid", False)), which="minor")
        except Exception:
            pass

    # ---- legend ----
    _apply_font_family_to_legend(legend, font_family, base)

def draw(ax, spec: PlotSpec, series: List[SeriesData], series_styles: Optional[List[SeriesStyleSpec]] = None) -> None:
    spec = spec.normalised()
    plot_type = spec.plot_type

    # ---- helpers ----
    def _norm_marker(m: str) -> str:
        m = (m or "").strip()
        return m if m else "o"

    def _outlier_colour(series_colour: str, series_index: int) -> object:
        """
        Per-series outlier colour.

        Priority:
        1) spec.style.outlier_color (explicit override) -> same for all series
        2) derived from series colour -> contrasting but "belongs" to that series
        3) fallback: deterministic Tableau colour by index
        """
        # 1) explicit global override
        c = str(getattr(spec.style, "outlier_color", "") or "").strip()
        if c:
            return c

        # 2) derive from series colour (invert + clamp)
        sc = (series_colour or "").strip()
        if sc:
            try:
                r, g, b = mcolors.to_rgb(sc)  # robust parsing of named/hex/etc
                inv = (0.15 + 0.85 * (1.0 - r), 0.15 + 0.85 * (1.0 - g), 0.15 + 0.85 * (1.0 - b))
                return inv
            except Exception:
                pass

        # 3) fallback: per-series tableau colour (shifted so it differs from common series defaults)
        tableau = list(mcolors.TABLEAU_COLORS.values())
        return tableau[(series_index + 3) % len(tableau)]

    def _overlay_outliers_1d(
        ax_,
        xvals: np.ndarray,
        y_level: float,
        series_colour: str,
        series_index: int,
        size: float,
        marker: str,
        zorder: float = 10.0,
    ) -> None:
        xvals = np.asarray(xvals, dtype=float)
        if xvals.size == 0:
            return
        m = _norm_marker(marker)
        oc = _outlier_colour(series_colour, series_index)
        s = (max(1.0, float(size)) * 1.15) ** 2  # slightly larger than base

        ax_.scatter(
            xvals,
            np.full(xvals.shape, y_level, dtype=float),
            label="_nolegend_",
            marker=m,
            s=s,
            c=oc,
            edgecolors=oc,
            linewidths=1.2,
            zorder=zorder,
        )

    def _overlay_outliers_xy(
        ax_,
        xvals: np.ndarray,
        yvals: np.ndarray,
        series_colour: str,
        series_index: int,
        size: float,
        marker: str,
        zorder: float = 10.0,
    ) -> None:
        xvals = np.asarray(xvals, dtype=float)
        yvals = np.asarray(yvals, dtype=float)
        if xvals.size == 0 or yvals.size == 0:
            return
        m = _norm_marker(marker)
        oc = _outlier_colour(series_colour, series_index)
        s = (max(1.0, float(size)) * 1.15) ** 2

        ax_.scatter(
            xvals,
            yvals,
            label="_nolegend_",
            marker=m,
            s=s,
            c=oc,
            edgecolors=oc,
            linewidths=1.2,
            zorder=zorder,
        )

    # ---- outlier legend helpers ----
    outlier_legend_handles: List[Line2D] = []
    outlier_legend_labels: List[str] = []
    _outlier_legend_added: set[int] = set()

    def _add_outlier_legend(i: int, label: str, series_colour: str, marker_size: float, line_width: float, marker: str) -> None:
        if not spec.style.show_legend:
            return
        if i in _outlier_legend_added:
            return
        _outlier_legend_added.add(i)

        oc = _outlier_colour(series_colour, i)
        proxy = Line2D(
            [0], [0],
            linestyle="none",
            marker=_norm_marker(marker),
            markersize=max(4.0, float(marker_size) * 1.05),
            markeredgewidth=max(0.8, float(line_width) / 2.0),
            color=oc,
            label=f"{label} (outliers)",
        )
        outlier_legend_handles.append(proxy)
        outlier_legend_labels.append(proxy.get_label())

    def _legend_with_outliers():
        if not spec.style.show_legend:
            return None
        handles, labels = ax.get_legend_handles_labels()
        if outlier_legend_handles:
            handles = list(handles) + list(outlier_legend_handles)
            labels = list(labels) + list(outlier_legend_labels)
        return ax.legend(handles, labels)

    # ---- series style lookup ----
    def series_style(i: int):
        if series_styles is not None and i < len(series_styles):
            st = series_styles[i]
            color = (st.color or "").strip()
            marker = (st.marker or "").strip()
            marker_size = float(st.marker_size)
            line_width = float(st.line_width)
            line_style = (st.line_style or "solid").strip()
            label = series[i].label
            return color, marker, marker_size, line_width, line_style, label

        sspec = spec.series[i] if i < len(spec.series) else None
        sstyle = sspec.style if sspec is not None else None
        color = (sstyle.color.strip() if (sstyle and sstyle.color) else "")
        marker = (sstyle.marker.strip() if (sstyle and sstyle.marker) else "")
        marker_size = float(sstyle.marker_size) if sstyle else 6.0
        line_width = float(sstyle.line_width) if sstyle else 1.6
        line_style = (sstyle.line_style.strip() if (sstyle and sstyle.line_style) else "solid")
        label = sspec.label if sspec else series[i].label
        return color, marker, marker_size, line_width, line_style, label

    def series_highlight(i: int) -> bool:
        if series_styles is not None and i < len(series_styles):
            return bool(getattr(series_styles[i], "highlight_outliers", False))
        sspec = spec.series[i] if i < len(spec.series) else None
        if sspec is None:
            return False
        return bool(getattr(sspec.style, "highlight_outliers", False))

    def get_outlier_mask(i: int, sdata: SeriesData) -> Optional[np.ndarray]:
        if not series_highlight(i):
            return None
        mask = getattr(sdata, "outlier_mask", None)
        if mask is None:
            sigma = float(getattr(spec.style, "outlier_sigma", 3.0))
            method = str(getattr(spec.style, "outlier_method", "std") or "std")
            mask = _compute_outlier_mask(sdata.y, sigma, method)
        return mask

    # ---- y-only plots ----
    if plot_type == PlotType.HIST:
        datasets = [s.y for s in series]
        labels = [series_style(i)[5] for i in range(len(series))]
        colors = [series_style(i)[0] or None for i in range(len(series))]
        color_arg = colors if any(c is not None for c in colors) else None

        ax.hist(datasets, label=labels, color=color_arg)

        ymin, ymax = ax.get_ylim()
        span = max(1e-9, (ymax - ymin))
        base = ymin + 0.03 * span
        step = 0.02 * span

        for i, sdata in enumerate(series):
            mask = get_outlier_mask(i, sdata)
            if mask is None or not np.any(mask):
                continue
            color, marker, msz, lw, _, lab = series_style(i)
            _overlay_outliers_1d(ax, np.asarray(sdata.y)[mask], base + i * step, color, i, msz, marker)
            _add_outlier_legend(i, lab, color, msz, lw, marker)

        legend = _legend_with_outliers()
        apply_style(ax, spec, legend=legend)
        return

    if plot_type == PlotType.BOX:
        datasets = [np.asarray(s.y, dtype=float) for s in series]
        labels = [series_style(i)[5] for i in range(len(series))]
        positions = np.arange(1, len(datasets) + 1, dtype=float)

        ax.boxplot(datasets, labels=labels, positions=positions, showfliers=False)

        for i, sdata in enumerate(series):
            mask = get_outlier_mask(i, sdata)
            if mask is None or not np.any(mask):
                continue
            color, marker, msz, lw, _, lab = series_style(i)
            x = np.full(np.sum(mask), positions[i], dtype=float)
            y = np.asarray(sdata.y, dtype=float)[mask]
            jitter = (np.random.RandomState(1000 + i).randn(x.size) * 0.02)
            _overlay_outliers_xy(ax, x + jitter, y, color, i, msz, marker)
            _add_outlier_legend(i, lab, color, msz, lw, marker)

        legend = _legend_with_outliers()
        apply_style(ax, spec, legend=legend)
        return

    if plot_type == PlotType.VIOLIN:
        datasets = [np.asarray(s.y, dtype=float) for s in series]
        labels = [series_style(i)[5] for i in range(len(series))]
        positions = np.arange(1, len(datasets) + 1, dtype=float)

        ax.violinplot(datasets, positions=positions, showmedians=True)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)

        for i, sdata in enumerate(series):
            mask = get_outlier_mask(i, sdata)
            if mask is None or not np.any(mask):
                continue
            color, marker, msz, lw, _, lab = series_style(i)
            x = np.full(np.sum(mask), positions[i], dtype=float)
            y = np.asarray(sdata.y, dtype=float)[mask]
            jitter = (np.random.RandomState(2000 + i).randn(x.size) * 0.02)
            _overlay_outliers_xy(ax, x + jitter, y, color, i, msz, marker)
            _add_outlier_legend(i, lab, color, msz, lw, marker)

        legend = _legend_with_outliers()
        apply_style(ax, spec, legend=legend)
        return

    if plot_type == PlotType.KDE:
        for i, sdata in enumerate(series):
            color, marker, msz, lw, ls, lab = series_style(i)
            xg, dens = _simple_kde(sdata.y)
            kwargs = {"label": lab, "linewidth": max(0.1, lw), "linestyle": ls or "solid"}
            if color:
                kwargs["color"] = color
            ax.plot(xg, dens, **kwargs)

            mask = get_outlier_mask(i, sdata)
            if mask is not None and np.any(mask):
                out_x = np.asarray(sdata.y, dtype=float)[mask]
                out_y = np.interp(out_x, np.asarray(xg, dtype=float), np.asarray(dens, dtype=float))
                good = np.isfinite(out_y)
                _overlay_outliers_xy(ax, out_x[good], out_y[good], color, i, msz, marker)
                _add_outlier_legend(i, lab, color, msz, lw, marker)

        legend = _legend_with_outliers()
        apply_style(ax, spec, legend=legend)
        return

    if plot_type == PlotType.ECDF:
        for i, sdata in enumerate(series):
            color, marker, msz, lw, ls, lab = series_style(i)
            y = np.asarray(sdata.y, dtype=float)
            y = y[np.isfinite(y)]
            if y.size == 0:
                continue
            y_sorted = np.sort(y)
            p = np.arange(1, y_sorted.size + 1, dtype=float) / y_sorted.size
            kwargs = {"label": lab, "linewidth": max(0.1, lw), "linestyle": ls or "solid"}
            if color:
                kwargs["color"] = color
            ax.step(y_sorted, p, where="post", **kwargs)

            sigma = float(getattr(spec.style, "outlier_sigma", 3.0))
            method = str(getattr(spec.style, "outlier_method", "std") or "std")
            mask_sorted = _compute_outlier_mask(y_sorted, sigma, method) if series_highlight(i) else None
            if mask_sorted is not None and np.any(mask_sorted):
                _overlay_outliers_xy(ax, y_sorted[mask_sorted], p[mask_sorted], color, i, msz, marker)
                _add_outlier_legend(i, lab, color, msz, lw, marker)

        legend = _legend_with_outliers()
        apply_style(ax, spec, legend=legend)
        return

    if plot_type == PlotType.QQNORM:
        for i, sdata in enumerate(series):
            color, marker, msz, lw, _, lab = series_style(i)
            y = np.asarray(sdata.y, dtype=float)
            y = y[np.isfinite(y)]
            if y.size < 2:
                continue
            y_sorted = np.sort(y)
            n = y_sorted.size
            p = (np.arange(1, n + 1) - 0.5) / n
            x_theory = np.array([_norm_ppf(float(pi)) for pi in p], dtype=float)

            kwargs = {"label": lab, "marker": _norm_marker(marker), "s": max(1.0, msz) ** 2}
            if color:
                kwargs["c"] = color
            ax.scatter(x_theory, y_sorted, **kwargs)

            sigma = float(getattr(spec.style, "outlier_sigma", 3.0))
            method = str(getattr(spec.style, "outlier_method", "std") or "std")
            mask_sorted = _compute_outlier_mask(y_sorted, sigma, method) if series_highlight(i) else None
            if mask_sorted is not None and np.any(mask_sorted):
                _overlay_outliers_xy(ax, x_theory[mask_sorted], y_sorted[mask_sorted], color, i, msz, marker)
                _add_outlier_legend(i, lab, color, msz, lw, marker)

        legend = _legend_with_outliers()
        apply_style(ax, spec, legend=legend)
        return

    # ---- bar plot ----
    if plot_type == PlotType.BAR:
        n = len(series)
        width = 0.8 / max(1, n)
        for i, sdata in enumerate(series):
            color, marker, msz, lw, _, lab = series_style(i)
            offset = (i - (n - 1) / 2.0) * width
            x = np.asarray(sdata.x, dtype=float) + offset

            kwargs = {"label": lab, "width": width}
            if color:
                kwargs["color"] = color
            if sdata.y_std is not None:
                kwargs["yerr"] = sdata.y_std
                kwargs["capsize"] = 3
            ax.bar(x, sdata.y, **kwargs)

            mask = get_outlier_mask(i, sdata)
            if mask is not None and np.any(mask):
                _overlay_outliers_xy(ax, x[mask], np.asarray(sdata.y, dtype=float)[mask], color, i, msz, marker)
                _add_outlier_legend(i, lab, color, msz, lw, marker)

        legend = _legend_with_outliers()
        apply_style(ax, spec, legend=legend)
        return

    if plot_type == PlotType.HEXBIN:
        if len(series) > 0:
            s0 = series[0]
            ax.hexbin(s0.x, s0.y, gridsize=30, cmap="viridis")

            mask = get_outlier_mask(0, s0)
            if mask is not None and np.any(mask):
                color, marker, msz, lw, _, lab = series_style(0)
                _overlay_outliers_xy(ax, np.asarray(s0.x, dtype=float)[mask], np.asarray(s0.y, dtype=float)[mask], color, 0, msz, marker)
                _add_outlier_legend(0, lab, color, msz, lw, marker)

        legend = _legend_with_outliers()
        apply_style(ax, spec, legend=legend)
        return

    if plot_type == PlotType.HIST2D:
        if len(series) > 0:
            s0 = series[0]
            ax.hist2d(s0.x, s0.y, bins=40, cmap="viridis")

            mask = get_outlier_mask(0, s0)
            if mask is not None and np.any(mask):
                color, marker, msz, lw, _, lab = series_style(0)
                _overlay_outliers_xy(ax, np.asarray(s0.x, dtype=float)[mask], np.asarray(s0.y, dtype=float)[mask], color, 0, msz, marker)
                _add_outlier_legend(0, lab, color, msz, lw, marker)

        legend = _legend_with_outliers()
        apply_style(ax, spec, legend=legend)
        return

    # ---- unified per-series plotting for 2D + 3D ----
    is_3d = plot_type in (PlotType.LINE3D, PlotType.SCATTER3D)
    is_scatter = plot_type in (PlotType.SCATTER, PlotType.SCATTER3D)
    is_step = plot_type == PlotType.STEP
    is_area = plot_type == PlotType.AREA

    for i, sdata in enumerate(series):
        color, marker, msz, lw, ls, lab = series_style(i)

        z = getattr(sdata, "z", None)
        if is_3d and z is None:
            continue

        mask = get_outlier_mask(i, sdata)
        has_out = mask is not None and bool(np.any(mask))

        has_err_2d = (not is_3d) and ((sdata.x_std is not None) or (sdata.y_std is not None))

        if is_scatter:
            if has_err_2d:
                kwargs = {"label": lab, "linestyle": "none", "capsize": 3}
                if color:
                    kwargs["color"] = color
                    kwargs["ecolor"] = color
                if marker:
                    kwargs["marker"] = marker
                kwargs["markersize"] = max(1.0, msz)
                ax.errorbar(sdata.x, sdata.y, xerr=sdata.x_std, yerr=sdata.y_std, **kwargs)
            else:
                if is_3d:
                    kwargs = {"label": lab, "marker": _norm_marker(marker), "s": max(1.0, msz) ** 2}
                    if color:
                        kwargs["c"] = color
                    ax.scatter(sdata.x, sdata.y, zs=z, **kwargs)
                else:
                    kwargs = {"label": lab, "marker": _norm_marker(marker), "s": max(1.0, msz) ** 2}
                    if color:
                        kwargs["c"] = color
                    ax.scatter(sdata.x, sdata.y, **kwargs)

        elif is_step:
            kwargs = {"label": lab, "linewidth": max(0.1, lw), "linestyle": ls or "solid"}
            if color:
                kwargs["color"] = color
            if marker:
                kwargs["marker"] = marker
                kwargs["markersize"] = max(1.0, msz)
            ax.step(sdata.x, sdata.y, **kwargs)

            if has_err_2d:
                ekwargs = {"fmt": "none", "capsize": 3}
                if color:
                    ekwargs["ecolor"] = color
                ax.errorbar(sdata.x, sdata.y, xerr=sdata.x_std, yerr=sdata.y_std, **ekwargs)

        elif is_area:
            kwargs = {"label": lab}
            if color:
                kwargs["color"] = color
            ax.fill_between(sdata.x, sdata.y, 0, **kwargs)

        else:
            if has_err_2d:
                kwargs = {"label": lab, "linewidth": max(0.1, lw), "linestyle": ls or "solid", "capsize": 3}
                if color:
                    kwargs["color"] = color
                    kwargs["ecolor"] = color
                if marker:
                    kwargs["marker"] = marker
                    kwargs["markersize"] = max(1.0, msz)
                ax.errorbar(sdata.x, sdata.y, xerr=sdata.x_std, yerr=sdata.y_std, **kwargs)
            else:
                kwargs = {"label": lab, "linewidth": max(0.1, lw), "linestyle": ls or "solid"}
                if color:
                    kwargs["color"] = color
                if marker:
                    kwargs["marker"] = marker
                    kwargs["markersize"] = max(1.0, msz)

                if is_3d:
                    ax.plot(sdata.x, sdata.y, z, **kwargs)
                else:
                    ax.plot(sdata.x, sdata.y, **kwargs)

        # outliers: same marker, per-series colour, drawn on top
        if has_out:
            out_c = _outlier_colour(color, i)
            m = _norm_marker(marker)
            s = (max(1.0, float(msz)) * 1.15) ** 2

            if is_3d:
                zz = getattr(sdata, "z", None)
                if zz is not None:
                    ax.scatter(
                        np.asarray(sdata.x)[mask],
                        np.asarray(sdata.y)[mask],
                        zs=np.asarray(zz)[mask],
                        label="_nolegend_",
                        marker=m,
                        s=s,
                        c=out_c,
                        edgecolors=out_c,
                        linewidths=1.2,
                        zorder=10.0,
                    )
                    _add_outlier_legend(i, lab, color, msz, lw, marker)
            else:
                ax.scatter(
                    np.asarray(sdata.x)[mask],
                    np.asarray(sdata.y)[mask],
                    label="_nolegend_",
                    marker=m,
                    s=s,
                    c=out_c,
                    edgecolors=out_c,
                    linewidths=1.2,
                    zorder=10.0,
                )
                _add_outlier_legend(i, lab, color, msz, lw, marker)

    legend = _legend_with_outliers()
    apply_style(ax, spec, legend=legend)


def _parse_group_order(text: str) -> List[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


def _manual_underline_title(ax, title_text) -> None:
    """
    Draw a line under the title text using its rendered bounding box.
    Best-effort underline for Matplotlib versions without underline support.
    """
    fig = ax.figure
    canvas = fig.canvas
    try:
        canvas.draw()
        renderer = canvas.get_renderer()
    except Exception:
        return

    bbox = title_text.get_window_extent(renderer=renderer)
    inv = fig.transFigure.inverted()
    (x0, y0) = inv.transform((bbox.x0, bbox.y0))
    (x1, _) = inv.transform((bbox.x1, bbox.y0))

    y = y0 - (2.0 / fig.bbox.height)

    old = getattr(fig, "_plotting_title_underline", None)
    if old is not None:
        try:
            old.remove()
        except Exception:
            pass

    lw = max(0.8, title_text.get_fontsize() / 14.0)
    line = Line2D([x0, x1], [y, y], transform=fig.transFigure, color=title_text.get_color(), linewidth=lw)

    fig.add_artist(line)
    fig._plotting_title_underline = line

    
def _norm_ppf(p: float) -> float:
    a = [-3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
          3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p <= 0.0:
        return -float("inf")
    if p >= 1.0:
        return float("inf")
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                 ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    q = p - 0.5
    r = q*q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)


def _simple_kde(y: np.ndarray, n_grid: int = 200) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 2:
        x = np.linspace(0, 1, n_grid)
        return x, np.zeros_like(x)

    std = float(np.std(y, ddof=1)) if y.size > 1 else 1.0
    iqr = float(np.subtract(*np.percentile(y, [75, 25])))
    sigma = min(std, iqr / 1.349) if (std > 0 and iqr > 0) else max(std, 1e-6)
    h = 0.9 * sigma * (y.size ** (-1/5))
    h = max(h, 1e-6)

    lo, hi = float(np.min(y)), float(np.max(y))
    pad = 0.1 * (hi - lo) if hi > lo else 1.0
    x = np.linspace(lo - pad, hi + pad, n_grid)

    u = (x[:, None] - y[None, :]) / h
    dens = np.exp(-0.5 * u*u).sum(axis=1) / (y.size * h * math.sqrt(2 * math.pi))
    return x, dens
