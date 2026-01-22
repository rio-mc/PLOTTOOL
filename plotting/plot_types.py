from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .spec import PlotType


@dataclass(frozen=True)
class PlotMeta:
    label: str

    # Data expectations
    requires_x: bool
    requires_y: bool
    requires_z: bool
    y_is_values_only: bool  # hist/box/violin: y is a single vector of values

    # Per-series style controls that make sense
    supports_markers: bool
    supports_marker_size: bool
    supports_lines: bool  # linewidth + linestyle controls

    # Std / errorbar support (per series)
    supports_x_std: bool
    supports_y_std: bool

    @property
    def requires_xy(self) -> bool:
        # "xy length must match" is only relevant when we truly plot paired x,y points.
        return bool(self.requires_x and self.requires_y and not self.y_is_values_only)


PLOT_META: Dict[PlotType, PlotMeta] = {
    # XY plots (support errorbars)
    PlotType.LINE: PlotMeta(
        "Line",
        requires_x=True,
        requires_y=True,
        requires_z=False,
        y_is_values_only=False,
        supports_markers=True,
        supports_marker_size=True,
        supports_lines=True,
        supports_x_std=True,
        supports_y_std=True,
    ),
    PlotType.SCATTER: PlotMeta(
        "Scatter",
        requires_x=True,
        requires_y=True,
        requires_z=False,
        y_is_values_only=False,
        supports_markers=True,
        supports_marker_size=True,
        supports_lines=False,
        supports_x_std=True,
        supports_y_std=True,
    ),
    PlotType.STEP: PlotMeta(
        "Step",
        requires_x=True,
        requires_y=True,
        requires_z=False,
        y_is_values_only=False,
        supports_markers=True,
        supports_marker_size=True,
        supports_lines=True,
        supports_x_std=True,
        supports_y_std=True,
    ),

    # These could support errorbands, but std errorbars don't map cleanly; disable for now.
    PlotType.AREA: PlotMeta(
        "Area",
        requires_x=True,
        requires_y=True,
        requires_z=False,
        y_is_values_only=False,
        supports_markers=False,
        supports_marker_size=False,
        supports_lines=False,
        supports_x_std=False,
        supports_y_std=False,
    ),

    # Bar: y-errorbars are common; x std is not typical.
    PlotType.BAR: PlotMeta(
        "Bar",
        requires_x=True,
        requires_y=True,
        requires_z=False,
        y_is_values_only=False,
        supports_markers=False,
        supports_marker_size=False,
        supports_lines=False,
        supports_x_std=False,
        supports_y_std=True,
    ),

    # y-only plots (std not supported here in this implementation)
    PlotType.HIST: PlotMeta(
        "Histogram",
        requires_x=False,
        requires_y=True,
        requires_z=False,
        y_is_values_only=True,
        supports_markers=False,
        supports_marker_size=False,
        supports_lines=False,
        supports_x_std=False,
        supports_y_std=False,
    ),
    PlotType.BOX: PlotMeta(
        "Box",
        requires_x=False,
        requires_y=True,
        requires_z=False,
        y_is_values_only=True,
        supports_markers=False,
        supports_marker_size=False,
        supports_lines=False,
        supports_x_std=False,
        supports_y_std=False,
    ),
    PlotType.VIOLIN: PlotMeta(
        "Violin",
        requires_x=False,
        requires_y=True,
        requires_z=False,
        y_is_values_only=True,
        supports_markers=False,
        supports_marker_size=False,
        supports_lines=False,
        supports_x_std=False,
        supports_y_std=False,
    ),
    PlotType.LINE3D: PlotMeta(
    "Line 3D",
    requires_x=True,
    requires_y=True,
    requires_z=True,
    y_is_values_only=False,
    supports_markers=True,
    supports_marker_size=True,
    supports_lines=True,
    supports_x_std=False,
    supports_y_std=False,
    ),
    PlotType.SCATTER3D: PlotMeta(
        "Scatter 3D",
        requires_x=True,
        requires_y=True,
        requires_z=True,
        y_is_values_only=False,
        supports_markers=True,
        supports_marker_size=True,
        supports_lines=False,
        supports_x_std=False,
        supports_y_std=False,
    ),

}


def meta_for(plot_type: PlotType) -> PlotMeta:
    return PLOT_META[plot_type]
