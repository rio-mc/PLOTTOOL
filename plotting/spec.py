from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional

@dataclass(frozen=True)
class PlotSettingsSpec:
    # per-plot options (bins, gridsize, cmap, bandwidth, etc.)
    options: Dict[str, Any] = field(default_factory=dict)

class PlotFamily(str, Enum):
    BASIC = "basic"
    DISTRIBUTION = "distribution"
    RELATIONSHIPS = "relationships"
    MULTIVARIATE = "multivariate"
    TIMESERIES = "timeseries"
    GEOSPATIAL = "geospatial"
    NETWORK = "network"
    SURVIVAL = "survival"
    GENOMICS = "genomics"
    ML_EVAL = "ml_eval"


class PlotType(str, Enum):
    # BASIC
    LINE = "line"
    SCATTER = "scatter"
    STEP = "step"
    AREA = "area"
    BAR = "bar"

    # DISTRIBUTION
    HIST = "hist"
    BOX = "box"
    VIOLIN = "violin"

    # 3D
    LINE3D = "line3d"
    SCATTER3D = "scatter3d"

    # TIME SERIES
    TIMESERIES = "timeseries"

    # DISTRIBUTION (y-only)
    KDE = "kde"
    ECDF = "ecdf"
    QQNORM = "qqnorm"

    # NEW MULTIVARIATE (x/y density)
    HEXBIN = "hexbin"
    HIST2D = "hist2d"


class SeriesInputMode(str, Enum):
    INLINE = "inline"
    TABLE = "table"


@dataclass(frozen=True)
class SeriesInlineData:
    # --- Inline vectors ---
    x_text: str = ""
    y_text: str = ""
    z_text: str = ""

    # Optional std vectors for inline mode
    x_std_text: str = ""
    y_std_text: str = ""

    # --- Table mode ---
    table_text: str = ""

    # Column names (for table mode)
    x_col: str = "x"
    y_col: str = "y"
    z_col: str = "z"

    # Std columns (optional)
    x_std_col: str = "x_std"
    y_std_col: str = "y_std"

    # Grouping (optional)
    group_col: str = "group"
    group_order_text: str = ""  # comma-separated explicit order


@dataclass(frozen=True)
class SeriesStyleSpec:
    color: str = ""
    marker: str = "o"
    marker_size: float = 6.0
    line_width: float = 1.6
    line_style: str = "solid"  # solid|dashed|dotted|dashdot
    highlight_outliers: bool = False

@dataclass(frozen=True)
class SeriesSpec:
    label: str = "Series"

    # Mode selection (INLINE vs TABLE)
    input_mode: SeriesInputMode = SeriesInputMode.INLINE

    # Optional std usage toggles
    use_x_std: bool = False
    use_y_std: bool = False

    x_tick_label_angle: int = 0
    y_tick_label_angle: int = 0

    # Group split controls (table mode)
    split_by_group: bool = False
    group_label_prefix: str = ""
    group_color_by_cycle: bool = True

    inline: SeriesInlineData = field(default_factory=SeriesInlineData)
    style: SeriesStyleSpec = field(default_factory=SeriesStyleSpec)


@dataclass(frozen=True)
class StyleSpec:
    font_family: str = ""

    title: str = ""
    title_bold: bool = False
    title_italic: bool = False
    title_underline: bool = False
    title_offset: Optional[float] = None

    x_label: str = ""
    y_label: str = ""
    z_label: Optional[str] = None

    # NEW: axis tick label angles (degrees)
    x_tick_label_angle: int = 0
    y_tick_label_angle: int = 0

    show_grid: bool = True
    show_minor_ticks: bool = False
    show_minor_grid: bool = False
    show_legend: bool = True

    base_font_size: int = 11
    title_font_size: int = 13

    outlier_sigma: float = 3.0              # k in ±kσ or ±k*MAD
    outlier_method: str = "std"             # "std" or "mad"


@dataclass(frozen=True)
class PlotSpec:
    plot_family: PlotFamily = PlotFamily.BASIC

    plot_type: PlotType = PlotType.LINE
    settings: PlotSettingsSpec = field(default_factory=PlotSettingsSpec)
    you_got_data: bool = False
    series_count: int = 1
    series: List[SeriesSpec] = field(default_factory=list)
    style: StyleSpec = field(default_factory=StyleSpec)

    def normalised(self) -> PlotSpec:
        # Import here to avoid circular imports at module load
        from .plot_types import family_for_type, default_type_for_family, family_is_available

        # clamp series count
        count = int(max(1, min(10, self.series_count)))

        # ensure series list length
        sers = list(self.series)
        if len(sers) < count:
            for i in range(len(sers), count):
                sers.append(SeriesSpec(label=f"Series {i + 1}"))
        elif len(sers) > count:
            sers = sers[:count]

        pf = self.plot_family
        pt = self.plot_type


        # If someone loads a config with a family that currently has no registered plots,
        # snap to BASIC (prevents “Line appears in every family” confusion).
        if not family_is_available(pf):
            pf = PlotFamily.BASIC

        # Ensure type belongs to selected family
        try:
            if family_for_type(pt) != pf:
                pt = default_type_for_family(pf)
        except Exception:
            pf = PlotFamily.BASIC
            pt = PlotType.LINE

        # Ensure labels are not blank
        fixed = []
        for i, s in enumerate(sers):
            lbl = (s.label or "").strip() or f"Series {i + 1}"
            fixed.append(
                SeriesSpec(
                    label=lbl,
                    input_mode=s.input_mode,
                    use_x_std=bool(getattr(s, "use_x_std", False)),
                    use_y_std=bool(getattr(s, "use_y_std", False)),
                    split_by_group=bool(getattr(s, "split_by_group", False)),
                    group_label_prefix=str(getattr(s, "group_label_prefix", "")),
                    group_color_by_cycle=bool(getattr(s, "group_color_by_cycle", True)),
                    inline=s.inline,
                    style=s.style,
                )
            )

        return PlotSpec(
            plot_family=pf,
            plot_type=pt,
            you_got_data=bool(self.you_got_data),
            series_count=count,
            series=fixed,
            style=self.style,
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["plot_family"] = self.plot_family.value
        d["plot_type"] = self.plot_type.value
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> PlotSpec:
        pf = PlotFamily(d.get("plot_family", PlotFamily.BASIC.value))
        pt = PlotType(d.get("plot_type", PlotType.LINE.value))

        style_d = d.get("style", {}) or {}
        series_d = d.get("series", []) or []

        series: List[SeriesSpec] = []
        for i, sd in enumerate(series_d):
            inline_d = (sd.get("inline", {}) or {})
            style_sd = (sd.get("style", {}) or {})

            series.append(
                SeriesSpec(
                    label=str(sd.get("label", f"Series {i+1}")),
                    input_mode=SeriesInputMode(sd.get("input_mode", SeriesInputMode.INLINE.value)),
                    use_x_std=bool(sd.get("use_x_std", False)),
                    use_y_std=bool(sd.get("use_y_std", False)),
                    split_by_group=bool(sd.get("split_by_group", False)),
                    group_label_prefix=str(sd.get("group_label_prefix", "")),
                    group_color_by_cycle=bool(sd.get("group_color_by_cycle", True)),
                    inline=SeriesInlineData(
                        x_text=str(inline_d.get("x_text", "")),
                        y_text=str(inline_d.get("y_text", "")),
                        z_text=str(inline_d.get("z_text", "")),
                        x_std_text=str(inline_d.get("x_std_text", "")),
                        y_std_text=str(inline_d.get("y_std_text", "")),
                        table_text=str(inline_d.get("table_text", "")),
                        x_col=str(inline_d.get("x_col", "x")),
                        y_col=str(inline_d.get("y_col", "y")),
                        z_col=str(inline_d.get("z_col", "z")),
                        x_std_col=str(inline_d.get("x_std_col", "x_std")),
                        y_std_col=str(inline_d.get("y_std_col", "y_std")),
                        group_col=str(inline_d.get("group_col", "group")),
                        group_order_text=str(inline_d.get("group_order_text", "")),
                    ),
                    style=SeriesStyleSpec(
                        color=str(style_sd.get("color", "")),
                        marker=str(style_sd.get("marker", "o")),
                        marker_size=float(style_sd.get("marker_size", 6.0)),
                        line_width=float(style_sd.get("line_width", 1.6)),
                        line_style=str(style_sd.get("line_style", "solid")),
                        highlight_outliers=bool(style_sd.get("highlight_outliers", False)),
                    ),
                )
            )

        return PlotSpec(
            plot_family=pf,
            plot_type=pt,
            you_got_data=bool(d.get("you_got_data", False)),
            series_count=int(d.get("series_count", max(1, len(series) or 1))),
            series=series,
            style=StyleSpec(
                font_family=str(style_d.get("font_family", "")),

                title=str(style_d.get("title", "")),
                title_bold=bool(style_d.get("title_bold", False)),
                title_italic=bool(style_d.get("title_italic", False)),
                title_underline=bool(style_d.get("title_underline", False)),
                title_offset=(
                    None if style_d.get("title_offset", None) in (None, "")
                    else float(style_d.get("title_offset"))
                ),

                x_label=str(style_d.get("x_label", "")),
                y_label=str(style_d.get("y_label", "")),
                z_label=(None if not str(style_d.get("z_label", "")).strip() else str(style_d.get("z_label")).strip()),

                x_tick_label_angle=int(style_d.get("x_tick_label_angle", 0)),
                y_tick_label_angle=int(style_d.get("y_tick_label_angle", 0)),

                show_grid=bool(style_d.get("show_grid", True)),
                show_minor_ticks=bool(style_d.get("show_minor_ticks", False)),
                show_minor_grid=bool(style_d.get("show_minor_grid", False)),
                show_legend=bool(style_d.get("show_legend", True)),

                base_font_size=int(style_d.get("base_font_size", 11)),
                title_font_size=int(style_d.get("title_font_size", int(style_d.get("base_font_size", 11)) + 2)),
                outlier_sigma=float(style_d.get("outlier_sigma", 3.0)),
                outlier_method=str(style_d.get("outlier_method", "std")).strip().lower() or "std",
            ),
        ).normalised()
