from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Literal, Optional
from .spec import PlotFamily, PlotType

@dataclass(frozen=True)
class ControlDef:
    key: str                 # stored in spec.settings.options[key]
    label: str
    kind: str                # "bool" | "int" | "float" | "choice" | "text"
    default: Any
    min: float | None = None
    max: float | None = None
    choices: List[str] | None = None
    tooltip: str = ""

@dataclass(frozen=True)
class PlotMeta:
    # UI
    family: PlotFamily
    label: str

    # Data expectations
    requires_x: bool
    requires_y: bool
    y_is_values_only: bool  # hist/box/violin: y is a single vector of values
    requires_z: bool = False

    # Per-series style controls that make sense
    supports_markers: bool = False
    supports_marker_size: bool = False
    supports_lines: bool = False  # linewidth + linestyle controls

    # “Featurability” flags (extend over time)
    supports_grouping: bool = True
    supports_errorbars: bool = False
    supports_log_x: bool = True
    supports_log_y: bool = True

    requires_xy: bool = True
    supports_x_std: bool = False
    supports_y_std: bool = False

    x_is_datetime: bool = False

    # Feature controls shown in UI for this plot type
    controls: List[ControlDef] = ()

FAMILY_LABELS: Dict[PlotFamily, str] = {
    PlotFamily.BASIC: "Basic",
    PlotFamily.DISTRIBUTION: "Distribution",
    PlotFamily.RELATIONSHIPS: "Relationships",
    PlotFamily.MULTIVARIATE: "Multivariate",
    PlotFamily.TIMESERIES: "Time Series",
    PlotFamily.GEOSPATIAL: "Geospatial",
    PlotFamily.NETWORK: "Network",
    PlotFamily.SURVIVAL: "Survival",
    PlotFamily.GENOMICS: "Genomics",
    PlotFamily.ML_EVAL: "ML / Evaluation",
}


PLOT_META: Dict[PlotType, PlotMeta] = {
    # ----------------
    # BASIC (2D XY)
    # ----------------
    PlotType.LINE: PlotMeta(
        family=PlotFamily.BASIC,
        label="Line",
        requires_x=True,
        requires_y=True,
        requires_z=False,
        y_is_values_only=False,
        x_is_datetime=False,
        supports_markers=True,
        supports_marker_size=True,
        supports_lines=True,
        supports_grouping=True,
        supports_errorbars=True,
        supports_x_std=True,
        supports_y_std=True,
        requires_xy=True,
    ),
    PlotType.SCATTER: PlotMeta(
        family=PlotFamily.BASIC,
        label="Scatter",
        requires_x=True,
        requires_y=True,
        requires_z=False,
        y_is_values_only=False,
        x_is_datetime=False,
        supports_markers=True,
        supports_marker_size=True,
        supports_lines=False,
        supports_grouping=True,
        supports_errorbars=True,
        supports_x_std=True,
        supports_y_std=True,
        requires_xy=True,
    ),
    PlotType.STEP: PlotMeta(
        family=PlotFamily.BASIC,
        label="Step",
        requires_x=True,
        requires_y=True,
        requires_z=False,
        y_is_values_only=False,
        x_is_datetime=False,
        supports_markers=True,
        supports_marker_size=True,
        supports_lines=True,
        supports_grouping=True,
        supports_errorbars=True,
        supports_x_std=True,
        supports_y_std=True,
        requires_xy=True,
    ),
    PlotType.AREA: PlotMeta(
        family=PlotFamily.BASIC,
        label="Area",
        requires_x=True,
        requires_y=True,
        requires_z=False,
        y_is_values_only=False,
        x_is_datetime=False,
        supports_markers=False,
        supports_marker_size=False,
        supports_lines=False,
        supports_grouping=True,
        supports_errorbars=False,
        supports_x_std=False,
        supports_y_std=False,
        requires_xy=True,
    ),
    PlotType.BAR: PlotMeta(
        family=PlotFamily.BASIC,
        label="Bar",
        requires_x=True,
        requires_y=True,
        requires_z=False,
        y_is_values_only=False,
        x_is_datetime=False,
        supports_markers=False,
        supports_marker_size=False,
        supports_lines=False,
        supports_grouping=True,
        supports_errorbars=True,
        supports_x_std=False,
        supports_y_std=True,
        requires_xy=True,
    ),

    # ----------------
    # DISTRIBUTION (y-only)
    # ----------------
    PlotType.HIST: PlotMeta(
        family=PlotFamily.DISTRIBUTION,
        label="Histogram",
        requires_x=False,
        requires_y=True,
        y_is_values_only=True,
        x_is_datetime=False,
        supports_markers=False,
        supports_marker_size=False,
        supports_lines=False,
        supports_grouping=True,
        supports_log_x=False,
        supports_log_y=True,
        supports_errorbars=False,
        requires_xy=False,
    ),
    PlotType.BOX: PlotMeta(
        family=PlotFamily.DISTRIBUTION,
        label="Box",
        requires_x=False,
        requires_y=True,
        y_is_values_only=True,
        x_is_datetime=False,
        supports_markers=False,
        supports_marker_size=False,
        supports_lines=False,
        supports_grouping=True,
        supports_log_x=False,
        supports_log_y=True,
        supports_errorbars=False,
        requires_xy=False,
    ),
    PlotType.VIOLIN: PlotMeta(
        family=PlotFamily.DISTRIBUTION,
        label="Violin",
        requires_x=False,
        requires_y=True,
        y_is_values_only=True,
        x_is_datetime=False,
        supports_markers=False,
        supports_marker_size=False,
        supports_lines=False,
        supports_grouping=True,
        supports_log_x=False,
        supports_log_y=True,
        supports_errorbars=False,
        requires_xy=False,
    ),

    # ----------------
    # MULTIVARIATE (3D)
    # ----------------
    PlotType.LINE3D: PlotMeta(
        family=PlotFamily.BASIC,
        label="Line (3D)",
        requires_x=True,
        requires_y=True,
        requires_z=True,
        y_is_values_only=False,
        x_is_datetime=False,
        supports_markers=True,
        supports_marker_size=True,
        supports_lines=True,
        supports_grouping=True,
        supports_errorbars=False,
        supports_x_std=False,
        supports_y_std=False,
        requires_xy=True,
    ),
    PlotType.SCATTER3D: PlotMeta(
        family=PlotFamily.BASIC,
        label="Scatter (3D)",
        requires_x=True,
        requires_y=True,
        requires_z=True,
        y_is_values_only=False,
        x_is_datetime=False,
        supports_markers=True,
        supports_marker_size=True,
        supports_lines=False,
        supports_grouping=True,
        supports_errorbars=False,
        supports_x_std=False,
        supports_y_std=False,
        requires_xy=True,
    ),

    # ----------------
    # DISTRIBUTION (y-only)
    # ----------------
    PlotType.KDE: PlotMeta(
        family=PlotFamily.DISTRIBUTION,
        label="KDE (Density)",
        requires_x=False,
        requires_y=True,
        y_is_values_only=True,
        x_is_datetime=False,
        supports_markers=False,
        supports_marker_size=False,
        supports_lines=True,
        supports_grouping=True,
        supports_log_x=False,
        supports_log_y=True,
        supports_errorbars=False,
        requires_xy=False,
        supports_x_std=False,
        supports_y_std=False,
    ),
    PlotType.ECDF: PlotMeta(
        family=PlotFamily.DISTRIBUTION,
        label="ECDF",
        requires_x=False,
        requires_y=True,
        y_is_values_only=True,
        x_is_datetime=False,
        supports_markers=False,
        supports_marker_size=False,
        supports_lines=True,
        supports_grouping=True,
        supports_log_x=False,
        supports_log_y=True,
        supports_errorbars=False,
        requires_xy=False,
        supports_x_std=False,
        supports_y_std=False,
    ),
    PlotType.QQNORM: PlotMeta(
        family=PlotFamily.DISTRIBUTION,
        label="Q-Q (Normal)",
        requires_x=False,
        requires_y=True,
        y_is_values_only=True,
        x_is_datetime=False,
        supports_markers=True,
        supports_marker_size=True,
        supports_lines=False,
        supports_grouping=True,
        supports_log_x=False,
        supports_log_y=False,
        supports_errorbars=False,
        requires_xy=False,
        supports_x_std=False,
        supports_y_std=False,
    ),

    # ----------------
    # MULTIVARIATE (2D density)
    # ----------------
    PlotType.HEXBIN: PlotMeta(
        family=PlotFamily.MULTIVARIATE,
        label="Hexbin (Density)",
        requires_x=True,
        requires_y=True,
        requires_z=False,
        y_is_values_only=False,
        x_is_datetime=False,
        supports_markers=False,
        supports_marker_size=False,
        supports_lines=False,
        supports_grouping=False,   # typically one density layer
        supports_log_x=True,
        supports_log_y=True,
        supports_errorbars=False,
        requires_xy=True,
        supports_x_std=False,
        supports_y_std=False,
    ),
    PlotType.HIST2D: PlotMeta(
        family=PlotFamily.MULTIVARIATE,
        label="2D Histogram",
        requires_x=True,
        requires_y=True,
        requires_z=False,
        y_is_values_only=False,
        x_is_datetime=False,
        supports_markers=False,
        supports_marker_size=False,
        supports_lines=False,
        supports_grouping=False,
        supports_log_x=True,
        supports_log_y=True,
        supports_errorbars=False,
        requires_xy=True,
        supports_x_std=False,
        supports_y_std=False,
    ),
    PlotType.TIMESERIES: PlotMeta(
        family=PlotFamily.BASIC,
        label="Time series",
        requires_x=True,
        requires_y=True,
        requires_z=False,
        y_is_values_only=False,
        x_is_datetime=True,
        supports_markers=True,
        supports_marker_size=True,
        supports_lines=True,
        supports_grouping=True,
        supports_errorbars=True,
        supports_x_std=True,
        supports_y_std=True,
        requires_xy=True,
    ),
}


def meta_for(plot_type: PlotType) -> PlotMeta:
    return PLOT_META[plot_type]


def family_for_type(plot_type: PlotType) -> PlotFamily:
    return meta_for(plot_type).family


def label_for_family(family: PlotFamily) -> str:
    return FAMILY_LABELS.get(family, family.value)


def types_for_family(family: PlotFamily) -> List[PlotType]:
    # Stable ordering (dict insertion order)
    return [pt for pt, m in PLOT_META.items() if m.family == family]


def family_is_available(family: PlotFamily) -> bool:
    return len(types_for_family(family)) > 0


def available_families() -> List[PlotFamily]:
    return [fam for fam in PlotFamily if family_is_available(fam)]


def default_type_for_family(family: PlotFamily) -> PlotType:
    t = types_for_family(family)
    return t[0] if t else PlotType.LINE
