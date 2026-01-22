from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List


class PlotType(str, Enum):
    LINE = "line"
    SCATTER = "scatter"
    STEP = "step"
    AREA = "area"
    BAR = "bar"
    HIST = "hist"
    BOX = "box"
    VIOLIN = "violin"

    LINE3D = "line3d"
    SCATTER3D = "scatter3d"


@dataclass(frozen=True)
class SeriesInlineData:
    x_text: str = ""
    y_text: str = ""
    z_text: str = ""
    x_std_text: str = ""
    y_std_text: str = ""


@dataclass(frozen=True)
class SeriesStyleSpec:
    color: str = ""
    marker: str = "o"
    marker_size: float = 6.0
    line_width: float = 1.6
    line_style: str = "solid"  # "solid" | "dashed" | "dotted" | "dashdot"


@dataclass(frozen=True)
class SeriesSpec:
    label: str = "Series"

    # Std toggles (per series)
    use_x_std: bool = False
    use_y_std: bool = False

    inline: SeriesInlineData = field(default_factory=SeriesInlineData)
    style: SeriesStyleSpec = field(default_factory=SeriesStyleSpec)


@dataclass(frozen=True)
class StyleSpec:
    # Applies to whole graph (labels, ticks, legend, title font family)
    font_family: str = ""  # empty means Matplotlib default

    # Title only
    title: str = ""
    title_bold: bool = False
    title_italic: bool = False
    title_underline: bool = False

    # Axes
    x_label: str = ""
    y_label: str = ""
    show_grid: bool = True
    show_legend: bool = True

    # Base sizing
    base_font_size: int = 11


@dataclass(frozen=True)
class PlotSpec:
    plot_type: PlotType = PlotType.LINE
    you_got_data: bool = False
    series_count: int = 1
    series: List[SeriesSpec] = field(default_factory=list)
    style: StyleSpec = field(default_factory=StyleSpec)

    def normalised(self) -> PlotSpec:
        count = int(max(1, min(10, self.series_count)))
        current = list(self.series)

        if len(current) < count:
            for i in range(len(current), count):
                current.append(SeriesSpec(label=f"Series {i + 1}"))
        elif len(current) > count:
            current = current[:count]

        return PlotSpec(
            plot_type=self.plot_type,
            you_got_data=bool(self.you_got_data),
            series_count=count,
            series=current,
            style=self.style,
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["plot_type"] = self.plot_type.value
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> PlotSpec:
        plot_type = PlotType(d.get("plot_type", PlotType.LINE.value))
        style_d = d.get("style", {}) or {}

        series_list: List[SeriesSpec] = []
        for s in d.get("series", []) or []:
            inline_d = (s.get("inline", {}) or {})
            sstyle_d = (s.get("style", {}) or {})
            series_list.append(
                SeriesSpec(
                    label=str(s.get("label", "Series")),
                    use_x_std=bool(s.get("use_x_std", False)),
                    use_y_std=bool(s.get("use_y_std", False)),
                    inline=SeriesInlineData(
                        x_text=str(inline_d.get("x_text", "")),
                        y_text=str(inline_d.get("y_text", "")),
                        z_text=str(inline_d.get("z_text", "")),
                        x_std_text=str(inline_d.get("x_std_text", "")),
                        y_std_text=str(inline_d.get("y_std_text", "")),
                    ),
                    style=SeriesStyleSpec(
                        color=str(sstyle_d.get("color", "")),
                        marker=str(sstyle_d.get("marker", "o")),
                        marker_size=float(sstyle_d.get("marker_size", 6.0)),
                        line_width=float(sstyle_d.get("line_width", 1.6)),
                        line_style=str(sstyle_d.get("line_style", "solid")),
                    ),
                )
            )

        return PlotSpec(
            plot_type=plot_type,
            you_got_data=bool(d.get("you_got_data", False)),
            series_count=int(d.get("series_count", max(1, len(series_list) or 1))),
            series=series_list,
            style=StyleSpec(
                font_family=str(style_d.get("font_family", "")),
                title=str(style_d.get("title", "")),
                title_bold=bool(style_d.get("title_bold", False)),
                title_italic=bool(style_d.get("title_italic", False)),
                title_underline=bool(style_d.get("title_underline", False)),
                x_label=str(style_d.get("x_label", "")),
                y_label=str(style_d.get("y_label", "")),
                show_grid=bool(style_d.get("show_grid", True)),
                show_legend=bool(style_d.get("show_legend", True)),
                base_font_size=int(style_d.get("base_font_size", 11)),
            ),
        ).normalised()
