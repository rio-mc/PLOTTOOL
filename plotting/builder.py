from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from matplotlib.lines import Line2D

from .spec import PlotSpec, PlotType
from .parsing import ParseIssue, parse_inline_vector, validate_xy_lengths
from .sample_data import SeriesData, make_sample_series
from .plot_types import meta_for


@dataclass(frozen=True)
class BuildResult:
    series: List[SeriesData]
    issues: List[ParseIssue]


def build_series_data(spec: PlotSpec) -> BuildResult:
    spec = spec.normalised()
    issues: List[ParseIssue] = []

    m = meta_for(spec.plot_type)

    # No user data: use sample
    if not spec.you_got_data:
        is_3d = getattr(meta_for(spec.plot_type), "requires_z", False)
        sample = make_sample_series(
            spec.series_count,
            is_3d=is_3d,
        )
        relabelled: List[SeriesData] = []
        for i, s in enumerate(sample):
            label = spec.series[i].label if i < len(spec.series) else s.label
            relabelled.append(
                SeriesData(
                    x=s.x,
                    y=s.y,
                    label=label,
                    z=None,
                    x_std=None,
                    y_std=None,
                )
            )
        return BuildResult(series=relabelled, issues=[])

    series_out: List[SeriesData] = []
    for idx, s in enumerate(spec.series):
        # Always parse y text (even for y-only plots)
        y, err_y = parse_inline_vector(s.inline.y_text)
        if err_y:
            issues.append(ParseIssue(message=err_y, series_index=idx, axis="y"))

        x_std: Optional[np.ndarray] = None
        y_std: Optional[np.ndarray] = None
        z: Optional[np.ndarray] = None

        if m.requires_x:
            x, err_x = parse_inline_vector(s.inline.x_text)
            if err_x:
                issues.append(ParseIssue(message=err_x, series_index=idx, axis="x"))

            len_err = validate_xy_lengths(x, y) if m.requires_xy else None
            if len_err:
                issues.append(ParseIssue(message=len_err, series_index=idx, axis="xy"))

            # Parse z (3D only)
            err_z: Optional[str] = None
            if getattr(m, "requires_z", False):
                z_arr, err_z = parse_inline_vector(getattr(s.inline, "z_text", ""))
                if err_z:
                    issues.append(ParseIssue(message=err_z, series_index=idx, axis="z"))
                else:
                    if z_arr.size != x.size:
                        issues.append(
                            ParseIssue(
                                message=f"z must have the same length as x/y (got {z_arr.size} vs {x.size}).",
                                series_index=idx,
                                axis="z",
                            )
                        )
                        err_z = "z length mismatch"
                    else:
                        z = z_arr

            # std parsing (only if supported + enabled)
            std_err = False

            if getattr(m, "supports_x_std", False) and bool(getattr(s, "use_x_std", False)):
                xstd_arr, err_xstd = parse_inline_vector(getattr(s.inline, "x_std_text", ""))
                if err_xstd:
                    issues.append(ParseIssue(message=err_xstd, series_index=idx, axis="x std"))
                    std_err = True
                else:
                    if xstd_arr.size != x.size:
                        issues.append(
                            ParseIssue(
                                message=f"x std must have the same length as x (got {xstd_arr.size} vs {x.size}).",
                                series_index=idx,
                                axis="x std",
                            )
                        )
                        std_err = True
                    else:
                        x_std = xstd_arr

            if getattr(m, "supports_y_std", False) and bool(getattr(s, "use_y_std", False)):
                ystd_arr, err_ystd = parse_inline_vector(getattr(s.inline, "y_std_text", ""))
                if err_ystd:
                    issues.append(ParseIssue(message=err_ystd, series_index=idx, axis="y std"))
                    std_err = True
                else:
                    if ystd_arr.size != y.size:
                        issues.append(
                            ParseIssue(
                                message=f"y std must have the same length as y (got {ystd_arr.size} vs {y.size}).",
                                series_index=idx,
                                axis="y std",
                            )
                        )
                        std_err = True
                    else:
                        y_std = ystd_arr

            # Fallback if any parsing/length issues
            if err_x or err_y or len_err or std_err or err_z:
                fallback = make_sample_series(1, seed=100 + idx)[0]
                series_out.append(
                    SeriesData(
                        x=fallback.x,
                        y=fallback.y,
                        label=s.label or f"Series {idx + 1}",
                        z=None,
                        x_std=None,
                        y_std=None,
                    )
                )
            else:
                series_out.append(
                    SeriesData(
                        x=x,
                        y=y,
                        label=s.label or f"Series {idx + 1}",
                        z=z,
                        x_std=x_std,
                        y_std=y_std,
                    )
                )

        else:
            # y-only plots: histogram/box/violin
            if y.size == 0:
                issues.append(ParseIssue(message="Values must contain at least one number.", series_index=idx, axis="y"))
                fallback = make_sample_series(1, seed=200 + idx)[0]
                series_out.append(
                    SeriesData(
                        x=fallback.x,
                        y=fallback.y,
                        label=s.label or f"Series {idx + 1}",
                        z=None,
                        x_std=None,
                        y_std=None,
                    )
                )
            else:
                # x is unused for these plots, but keep SeriesData shape unchanged
                fallback_x = make_sample_series(1, seed=300 + idx)[0].x
                series_out.append(
                    SeriesData(
                        x=fallback_x,
                        y=y,
                        label=s.label or f"Series {idx + 1}",
                        z=None,
                        x_std=None,
                        y_std=None,
                    )
                )

    return BuildResult(series=series_out, issues=issues)


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


def apply_style(ax, spec: PlotSpec, legend=None) -> None:
    style = spec.style
    font_family = (style.font_family or "").strip()
    base = int(style.base_font_size)

    # Title text object with styling
    if style.title:
        title_str = style.title
    else:
        title_str = "Figure title" if not spec.you_got_data else ""
    title_text = ax.set_title(title_str)

    title_text.set_fontsize(base + 2)
    if font_family:
        title_text.set_fontfamily(font_family)

    title_text.set_fontweight("bold" if style.title_bold else "normal")
    title_text.set_fontstyle("italic" if style.title_italic else "normal")

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

    ax.set_xlabel(style.x_label if style.x_label else "", fontfamily=font_family or None, fontsize=base)
    ax.set_ylabel(style.y_label if style.y_label else "", fontfamily=font_family or None, fontsize=base)

    ax.grid(bool(style.show_grid))

    ax.tick_params(labelsize=base)
    _apply_font_family_to_ticks(ax, font_family)

    _apply_font_family_to_legend(legend, font_family, base)


def draw(ax, spec: PlotSpec, series: List[SeriesData]) -> None:
    spec = spec.normalised()
    plot_type = spec.plot_type

    def series_style(i: int):
        sspec = spec.series[i] if i < len(spec.series) else None
        sstyle = sspec.style if sspec is not None else None
        color = (sstyle.color.strip() if (sstyle and sstyle.color) else "")
        marker = (sstyle.marker.strip() if (sstyle and sstyle.marker) else "")
        marker_size = float(sstyle.marker_size) if sstyle else 6.0
        line_width = float(sstyle.line_width) if sstyle else 1.6
        line_style = (sstyle.line_style.strip() if (sstyle and sstyle.line_style) else "solid")
        label = sspec.label if sspec else series[i].label
        return color, marker, marker_size, line_width, line_style, label

    # ---- y-only plots ----
    if plot_type == PlotType.HIST:
        datasets = [s.y for s in series]
        labels = [series_style(i)[5] for i in range(len(series))]
        colors = [series_style(i)[0] or None for i in range(len(series))]
        color_arg = colors if any(c is not None for c in colors) else None
        ax.hist(datasets, label=labels, color=color_arg)
        legend = ax.legend() if spec.style.show_legend else None
        apply_style(ax, spec, legend=legend)
        return

    if plot_type == PlotType.BOX:
        datasets = [s.y for s in series]
        labels = [series_style(i)[5] for i in range(len(series))]
        ax.boxplot(datasets, labels=labels)
        apply_style(ax, spec, legend=None)
        return

    if plot_type == PlotType.VIOLIN:
        datasets = [s.y for s in series]
        labels = [series_style(i)[5] for i in range(len(series))]
        positions = np.arange(1, len(datasets) + 1)
        ax.violinplot(datasets, positions=positions, showmedians=True)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        apply_style(ax, spec, legend=None)
        return

    # ---- bar plot ----
    if plot_type == PlotType.BAR:
        n = len(series)
        width = 0.8 / max(1, n)
        for i, sdata in enumerate(series):
            color, marker, msize, lw, ls, label = series_style(i)
            offset = (i - (n - 1) / 2.0) * width
            x = np.asarray(sdata.x, dtype=float) + offset

            kwargs = {"label": label, "width": width}
            if color:
                kwargs["color"] = color
            if sdata.y_std is not None:
                kwargs["yerr"] = sdata.y_std
                kwargs["capsize"] = 3
            ax.bar(x, sdata.y, **kwargs)

        legend = ax.legend() if spec.style.show_legend else None
        apply_style(ax, spec, legend=legend)
        return

    # ---- unified per-series plotting for 2D + 3D ----
    is_3d = plot_type in (PlotType.LINE3D, PlotType.SCATTER3D)
    is_scatter = plot_type in (PlotType.SCATTER, PlotType.SCATTER3D)
    is_step = plot_type == PlotType.STEP
    is_area = plot_type == PlotType.AREA
    is_line = plot_type in (PlotType.LINE, PlotType.LINE3D)

    for i, sdata in enumerate(series):
        color, marker, marker_size, line_width, line_style, label = series_style(i)

        # If 3D plot type, require z to draw this series
        z = getattr(sdata, "z", None)
        if is_3d and z is None:
            continue

        # NOTE: std/errorbars are implemented for 2D only in this architecture.
        # If you later want 3D errorbars, we can implement a 3D "caps" renderer.
        has_err_2d = (not is_3d) and ((sdata.x_std is not None) or (sdata.y_std is not None))

        if is_scatter:
            if has_err_2d:
                # 2D scatter with errorbars via errorbar(fmt='none' equivalent behaviour)
                kwargs = {"label": label, "linestyle": "none", "capsize": 3}
                if color:
                    kwargs["color"] = color
                    kwargs["ecolor"] = color
                if marker:
                    kwargs["marker"] = marker
                kwargs["markersize"] = max(1.0, marker_size)
                ax.errorbar(sdata.x, sdata.y, xerr=sdata.x_std, yerr=sdata.y_std, **kwargs)
            else:
                # Scatter 2D or 3D
                if is_3d:
                    kwargs = {"label": label}
                    if color:
                        kwargs["c"] = color
                    if marker:
                        kwargs["marker"] = marker
                    kwargs["s"] = max(1.0, marker_size) ** 2
                    ax.scatter(sdata.x, sdata.y, z, **kwargs)
                else:
                    kwargs = {"label": label}
                    if color:
                        kwargs["c"] = color
                    if marker:
                        kwargs["marker"] = marker
                    kwargs["s"] = max(1.0, marker_size) ** 2
                    ax.scatter(sdata.x, sdata.y, **kwargs)

        elif is_step:
            # Step is only 2D in our plot types
            kwargs = {"label": label, "linewidth": max(0.1, line_width), "linestyle": line_style or "solid"}
            if color:
                kwargs["color"] = color
            if marker:
                kwargs["marker"] = marker
                kwargs["markersize"] = max(1.0, marker_size)
            ax.step(sdata.x, sdata.y, **kwargs)

            if has_err_2d:
                ekwargs = {"fmt": "none", "capsize": 3}
                if color:
                    ekwargs["ecolor"] = color
                ax.errorbar(sdata.x, sdata.y, xerr=sdata.x_std, yerr=sdata.y_std, **ekwargs)

        elif is_area:
            # Area is 2D only
            kwargs = {"label": label}
            if color:
                kwargs["color"] = color
            ax.fill_between(sdata.x, sdata.y, 0, **kwargs)

        else:
            # Line (2D) or Line3D
            if has_err_2d:
                kwargs = {
                    "label": label,
                    "linewidth": max(0.1, line_width),
                    "linestyle": line_style or "solid",
                    "capsize": 3,
                }
                if color:
                    kwargs["color"] = color
                    kwargs["ecolor"] = color
                if marker:
                    kwargs["marker"] = marker
                    kwargs["markersize"] = max(1.0, marker_size)
                ax.errorbar(sdata.x, sdata.y, xerr=sdata.x_std, yerr=sdata.y_std, **kwargs)
            else:
                kwargs = {"label": label, "linewidth": max(0.1, line_width), "linestyle": line_style or "solid"}
                if color:
                    kwargs["color"] = color
                if marker:
                    kwargs["marker"] = marker
                    kwargs["markersize"] = max(1.0, marker_size)

                if is_3d:
                    ax.plot(sdata.x, sdata.y, z, **kwargs)
                else:
                    ax.plot(sdata.x, sdata.y, **kwargs)

    legend = ax.legend() if spec.style.show_legend else None
    apply_style(ax, spec, legend=legend)



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
