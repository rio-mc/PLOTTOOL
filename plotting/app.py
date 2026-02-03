from __future__ import annotations

from dataclasses import replace
from typing import List, Optional

from PySide6 import QtCore, QtWidgets

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import font_manager

from .spec import PlotSpec, PlotFamily, PlotType, StyleSpec, SeriesSpec, SeriesInlineData, SeriesStyleSpec
from .builder import build_series_data, draw
from .export_code import export_code_scaffold
from .plot_types import (
    meta_for,
    types_for_family,
    default_type_for_family,
    label_for_family,
    available_families,
)

try:
    import shiboken6
except Exception:
    shiboken6 = None

class MatplotlibPreview(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._fig = Figure()
        self._canvas = FigureCanvas(self._fig)
        self._ax = self._fig.add_subplot(111)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)
        self.setLayout(layout)

    def render(self, spec: PlotSpec) -> List[str]:
        self._fig.clear()

        pt = spec.plot_type
        is_3d = bool(getattr(meta_for(pt), "requires_z", False))
        self._ax = self._fig.add_subplot(111, projection="3d" if is_3d else None)

        result = build_series_data(spec)
        draw(self._ax, spec, result.series)

        self._canvas.draw_idle()

        messages: List[str] = []
        for iss in result.issues:
            if iss.series_index is None:
                messages.append(iss.message)
            else:
                s = iss.series_index + 1
                axis = iss.axis or ""
                messages.append(f"Series {s} {axis}: {iss.message}".strip())
        return messages
    
    def save_figure(self, path: str, dpi: int = 300) -> None:
        self._fig.savefig(path, dpi=dpi, bbox_inches="tight")

    def apply_view(self, elev: float, azim: float, roll: float = 0.0) -> None:
        # Only relevant if this is a 3D axes
        if getattr(self._ax, "name", "") != "3d":
            return

        # Matplotlib supports elev/azim widely; roll is version-dependent.
        try:
            self._ax.view_init(elev=elev, azim=azim, roll=roll)
        except TypeError:
            # Older Matplotlib: no roll parameter
            self._ax.view_init(elev=elev, azim=azim)

        self._canvas.draw_idle()


class SeriesEditor(QtWidgets.QGroupBox):
    changed = QtCore.Signal()

    MARKERS = [
        ("Circle (o)", "o"),
        ("Square (s)", "s"),
        ("Triangle up (^)", "^"),
        ("Triangle down (v)", "v"),
        ("Diamond (D)", "D"),
        ("Plus (+)", "+"),
        ("Cross (x)", "x"),
        ("Point (.)", "."),
        ("None", ""),
    ]

    LINESTYLES = [
        ("Solid", "solid"),
        ("Dashed", "dashed"),
        ("Dotted", "dotted"),
        ("Dash-dot", "dashdot"),
    ]

    _PH_X = "x values, e.g.\n0, 1, 2, 3\nor\n0 1 2 3"
    _PH_Y = "y values, e.g.\n0.1, 0.4, 0.2, 0.9"
    _PH_VALUES = "values, e.g.\n0.1, 0.4, 0.2, 0.9"
    _PH_Z = "z values, e.g.\n0.2, 0.7, 1.1, 1.4"

    def __init__(self, index: int, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._index = index
        self.setTitle(f"Series {index + 1}")

        self.label_edit = QtWidgets.QLineEdit()
        self.label_edit.setPlaceholderText(f"Series {index + 1}")

        # Data boxes
        self.x_edit = QtWidgets.QPlainTextEdit()
        self.y_edit = QtWidgets.QPlainTextEdit()
        self.z_edit = QtWidgets.QPlainTextEdit()

        self.x_edit.setPlaceholderText(self._PH_X)
        self.y_edit.setPlaceholderText(self._PH_Y)
        self.z_edit.setPlaceholderText(self._PH_Z)

        # Styling controls (per series)
        self.color_edit = QtWidgets.QLineEdit()
        self.color_edit.setPlaceholderText("Colour e.g. tab:blue or #1f77b4 (blank = default)")
        self.color_btn = QtWidgets.QPushButton("Pick…")

        self.marker_combo = QtWidgets.QComboBox()
        for text, code in self.MARKERS:
            self.marker_combo.addItem(text, code)

        self.marker_size = QtWidgets.QDoubleSpinBox()
        self.marker_size.setRange(1.0, 40.0)
        self.marker_size.setDecimals(1)
        self.marker_size.setSingleStep(0.5)
        self.marker_size.setValue(6.0)

        self.line_width = QtWidgets.QDoubleSpinBox()
        self.line_width.setRange(0.1, 12.0)
        self.line_width.setDecimals(1)
        self.line_width.setSingleStep(0.1)
        self.line_width.setValue(1.6)

        self.line_style = QtWidgets.QComboBox()
        for text, code in self.LINESTYLES:
            self.line_style.addItem(text, code)

        self.outliers_check = QtWidgets.QCheckBox("Highlight outliers (±kσ)")
        self.outliers_check.setChecked(False)

        self.outliers_check.toggled.connect(lambda *_: self.changed.emit())

        # Layout
        form = QtWidgets.QFormLayout()
        form.addRow("Label", self.label_edit)

        color_row = QtWidgets.QHBoxLayout()
        color_row.setContentsMargins(0, 0, 0, 0)
        color_row.addWidget(self.color_edit, 1)
        color_row.addWidget(self.color_btn)
        color_row_widget = QtWidgets.QWidget()
        color_row_widget.setLayout(color_row)
        form.addRow("Colour", color_row_widget)

        self.x_label = QtWidgets.QLabel("x")
        self.y_label = QtWidgets.QLabel("y")
        self.z_label = QtWidgets.QLabel("z")

        form.addRow(self.x_label, self.x_edit)
        form.addRow(self.y_label, self.y_edit)
        form.addRow(self.z_label, self.z_edit)

        self.marker_label = QtWidgets.QLabel("Marker")
        self.marker_size_label = QtWidgets.QLabel("Marker size")
        self.line_width_label = QtWidgets.QLabel("Line width")
        self.line_style_label = QtWidgets.QLabel("Line style")

        form.addRow(self.marker_label, self.marker_combo)
        form.addRow(self.marker_size_label, self.marker_size)
        form.addRow(self.line_width_label, self.line_width)
        form.addRow(self.line_style_label, self.line_style)
        form.addRow("", self.outliers_check)

        self.setLayout(form)

        # Signals
        self.label_edit.textChanged.connect(lambda *_: self.changed.emit())
        self.x_edit.textChanged.connect(lambda *_: self.changed.emit())
        self.y_edit.textChanged.connect(lambda *_: self.changed.emit())
        self.z_edit.textChanged.connect(lambda *_: self.changed.emit())

        self.color_edit.textChanged.connect(lambda *_: self.changed.emit())
        self.marker_combo.currentIndexChanged.connect(lambda *_: self.changed.emit())
        self.marker_size.valueChanged.connect(lambda *_: self.changed.emit())
        self.line_width.valueChanged.connect(lambda *_: self.changed.emit())
        self.line_style.currentIndexChanged.connect(lambda *_: self.changed.emit())

        self.color_btn.clicked.connect(self._pick_colour)

        # Default: hide z until a 3D plot type enables it
        self.z_label.setVisible(False)
        self.z_edit.setVisible(False)

    def _pick_colour(self) -> None:
        col = QtWidgets.QColorDialog.getColor(parent=self)
        if not col.isValid():
            return
        self.color_edit.setText(col.name())  # #RRGGBB

    def flash(self, duration_ms: int = 650) -> None:
        original = self.styleSheet()
        self.setStyleSheet("QGroupBox { border: 2px solid palette(highlight); }")
        QtCore.QTimer.singleShot(duration_ms, lambda: self.setStyleSheet(original))

    def to_series_spec(self) -> SeriesSpec:
        label = self.label_edit.text().strip() or f"Series {self._index + 1}"
        return SeriesSpec(
            label=label,
            inline=SeriesInlineData(
                x_text=self.x_edit.toPlainText(),
                y_text=self.y_edit.toPlainText(),
                z_text=self.z_edit.toPlainText(),
            ),
            style=SeriesStyleSpec(
                color=self.color_edit.text().strip(),
                marker=str(self.marker_combo.currentData()),
                marker_size=float(self.marker_size.value()),
                line_width=float(self.line_width.value()),
                line_style=str(self.line_style.currentData()),
                highlight_outliers=self.outliers_check.isChecked(),
            ),
        )

    def from_series_spec(self, spec: SeriesSpec) -> None:
        self.label_edit.setText(spec.label)
        self.x_edit.setPlainText(spec.inline.x_text)
        self.y_edit.setPlainText(spec.inline.y_text)
        self.z_edit.setPlainText(getattr(spec.inline, "z_text", ""))

        self.color_edit.setText(spec.style.color)

        marker_code = spec.style.marker
        idx = self.marker_combo.findData(marker_code)
        self.marker_combo.setCurrentIndex(idx if idx >= 0 else 0)

        self.marker_size.setValue(float(spec.style.marker_size))
        self.line_width.setValue(float(spec.style.line_width))

        ls_code = spec.style.line_style
        idx = self.line_style.findData(ls_code)
        self.line_style.setCurrentIndex(idx if idx >= 0 else 0)
        self.outliers_check.setChecked(bool(getattr(spec.style, "highlight_outliers", False)))

    def apply_plot_type(self, plot_type: PlotType) -> None:
        m = meta_for(plot_type)

        show_x = bool(getattr(m, "requires_x", True))
        show_z = bool(getattr(m, "requires_z", False))
        values_only = bool(getattr(m, "y_is_values_only", False)) or not show_x

        # Visibility (never clear text; switching dims should preserve inputs)
        self.x_label.setVisible(show_x)
        self.x_edit.setVisible(show_x)
        self.z_label.setVisible(show_z)
        self.z_edit.setVisible(show_z)

        # Labels + placeholders (single convention for 1D/2D/3D)
        self.y_label.setText("values" if values_only else "y")

        self.x_edit.setPlaceholderText(self._PH_X if show_x else "")
        self.y_edit.setPlaceholderText(self._PH_VALUES if values_only else self._PH_Y)
        self.z_edit.setPlaceholderText(self._PH_Z if show_z else "")

        # Style control visibility
        self.marker_label.setVisible(m.supports_markers)
        self.marker_combo.setVisible(m.supports_markers)

        self.marker_size_label.setVisible(m.supports_marker_size)
        self.marker_size.setVisible(m.supports_marker_size)

        self.line_width_label.setVisible(m.supports_lines)
        self.line_width.setVisible(m.supports_lines)

        self.line_style_label.setVisible(m.supports_lines)
        self.line_style.setVisible(m.supports_lines)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Plot Studio (Prototype)")
        self.resize(1200, 750)

        self._spec = PlotSpec().normalised()
        self._debounce_timer = QtCore.QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self._rebuild_preview)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        main_layout = QtWidgets.QHBoxLayout()
        central.setLayout(main_layout)

        controls = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout()
        controls.setLayout(controls_layout)
        controls_layout.setContentsMargins(0, 0, 0, 0)

        # --- NEW: Plot family + plot type ---
        self.plot_family_combo = QtWidgets.QComboBox()
        self.plot_type_combo = QtWidgets.QComboBox()

        # Only show families that currently have unique registered plot types.
        for fam in available_families():
            self.plot_family_combo.addItem(label_for_family(fam), fam.value)

        # Populate types based on current family
        self._rebuild_plot_type_combo(self._spec.plot_family)

        # Series count
        self.series_spin = QtWidgets.QSpinBox()
        self.series_spin.setRange(1, 10)
        self.series_spin.setValue(1)

        top_form = QtWidgets.QFormLayout()
        top_form.addRow("Plot family", self.plot_family_combo)
        top_form.addRow("Plot type", self.plot_type_combo)
        controls_layout.addLayout(top_form)

        # Style controls
        style_box = QtWidgets.QGroupBox("Style")
        style_form = QtWidgets.QFormLayout()

        self.font_combo = QtWidgets.QComboBox()
        self.font_combo.setEditable(True)
        self.font_combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.font_combo.setMaxVisibleItems(20)

        fonts = sorted({f.name for f in font_manager.fontManager.ttflist})
        self.font_combo.addItems(fonts)

        completer = QtWidgets.QCompleter(fonts, self.font_combo)
        completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        completer.setFilterMode(QtCore.Qt.MatchContains)
        completer.setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
        self.font_combo.setCompleter(completer)

        self.title_edit = QtWidgets.QLineEdit()

        self.title_bold = QtWidgets.QToolButton()
        self.title_bold.setText("B")
        self.title_bold.setCheckable(True)

        self.title_italic = QtWidgets.QToolButton()
        self.title_italic.setText("I")
        self.title_italic.setCheckable(True)

        self.title_underline = QtWidgets.QToolButton()
        self.title_underline.setText("U")
        self.title_underline.setCheckable(True)

        self.title_offset_edit = QtWidgets.QLineEdit()

        title_row = QtWidgets.QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.addWidget(self.title_edit, 1)
        title_row.addWidget(self.title_bold)
        title_row.addWidget(self.title_italic)
        title_row.addWidget(self.title_underline)

        title_row_widget = QtWidgets.QWidget()
        title_row_widget.setLayout(title_row)

        self.xlab_edit = QtWidgets.QLineEdit()
        self.ylab_edit = QtWidgets.QLineEdit()

        self.grid_check = QtWidgets.QCheckBox("Grid (major)")
        self.grid_check.setChecked(True)

        self.minor_ticks_check = QtWidgets.QCheckBox("Minor ticks")
        self.minor_ticks_check.setChecked(False)

        self.minor_grid_check = QtWidgets.QCheckBox("Grid (minor)")
        self.minor_grid_check.setChecked(False)
        self.minor_grid_check.setEnabled(False)

        def _minor_ticks_toggled(on: bool) -> None:
            self.minor_grid_check.setEnabled(bool(on))
            if not on:
                self.minor_grid_check.setChecked(False)

        self.minor_ticks_check.toggled.connect(_minor_ticks_toggled)

        self.legend_check = QtWidgets.QCheckBox("Legend")
        self.legend_check.setChecked(True)

        self.base_font_spin = QtWidgets.QSpinBox()
        self.base_font_spin.setRange(7, 36)
        self.base_font_spin.setValue(11)

        self.title_font_spin = QtWidgets.QSpinBox()
        self.title_font_spin.setRange(7, 48)
        self.title_font_spin.setValue(13)

        self.outlier_sigma_spin = QtWidgets.QDoubleSpinBox()
        self.outlier_sigma_spin.setRange(0.1, 10.0)
        self.outlier_sigma_spin.setDecimals(2)
        self.outlier_sigma_spin.setSingleStep(0.25)
        self.outlier_sigma_spin.setValue(3.0)

        self.outlier_method_combo = QtWidgets.QComboBox()
        self.outlier_method_combo.addItem("Mean ± k·Std (σ)", "std")
        self.outlier_method_combo.addItem("Median ± k·MAD (robust)", "mad")


        # Tick label angles
        self.x_tick_angle_spin = QtWidgets.QSpinBox()
        self.x_tick_angle_spin.setRange(-180, 180)
        self.x_tick_angle_spin.setValue(0)

        self.y_tick_angle_spin = QtWidgets.QSpinBox()
        self.y_tick_angle_spin.setRange(-180, 180)
        self.y_tick_angle_spin.setValue(0)

        style_form.addRow("x tick label angle", self.x_tick_angle_spin)
        style_form.addRow("y tick label angle", self.y_tick_angle_spin)

        self.x_tick_angle_spin.valueChanged.connect(self._schedule_rebuild)
        self.y_tick_angle_spin.valueChanged.connect(self._schedule_rebuild)


        style_form.addRow("Font", self.font_combo)
        style_form.addRow("Title", title_row_widget)
        style_form.addRow("x label", self.xlab_edit)
        style_form.addRow("y label", self.ylab_edit)
        style_form.addRow("", self.grid_check)
        style_form.addRow("", self.minor_ticks_check)
        style_form.addRow("", self.minor_grid_check)
        style_form.addRow("", self.legend_check)
        style_form.addRow("Text font size", self.base_font_spin)
        style_form.addRow("Title font size", self.title_font_spin)
        style_form.addRow("Outlier k", self.outlier_sigma_spin)
        style_form.addRow("Outlier method", self.outlier_method_combo)

        style_box.setLayout(style_form)
        controls_layout.addWidget(style_box)

        # Series editors in a scroll
        self.series_container = QtWidgets.QWidget()
        self.series_layout = QtWidgets.QVBoxLayout()
        self.series_layout.setContentsMargins(0, 0, 0, 0)
        self.series_container.setLayout(self.series_layout)

        self.series_scroll = QtWidgets.QScrollArea()
        self.series_scroll.setWidgetResizable(True)
        self.series_scroll.setWidget(self.series_container)
        self.series_scroll.setMinimumWidth(380)

        # Export buttons
        btn_row = QtWidgets.QHBoxLayout()
        self.export_fig_btn = QtWidgets.QPushButton("Export figure")
        self.export_code_btn = QtWidgets.QPushButton("Export code")
        btn_row.addWidget(self.export_fig_btn)
        btn_row.addWidget(self.export_code_btn)


        # Centre: preview
        self.preview = MatplotlibPreview()

        # Right: series + view panel
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right.setLayout(right_layout)

        right_controls = QtWidgets.QGroupBox("Series and view")
        right_controls_layout = QtWidgets.QFormLayout()
        right_controls.setLayout(right_controls_layout)

        right_controls_layout.addRow("Series", self.series_spin)

        # 3D view controls
        self.view_elev = QtWidgets.QDoubleSpinBox()
        self.view_elev.setRange(-180.0, 180.0)
        self.view_elev.setDecimals(1)
        self.view_elev.setSingleStep(1.0)
        self.view_elev.setValue(30.0)

        self.view_azim = QtWidgets.QDoubleSpinBox()
        self.view_azim.setRange(-180.0, 180.0)
        self.view_azim.setDecimals(1)
        self.view_azim.setSingleStep(1.0)
        self.view_azim.setValue(-60.0)

        self.view_roll = QtWidgets.QDoubleSpinBox()
        self.view_roll.setRange(-180.0, 180.0)
        self.view_roll.setDecimals(1)
        self.view_roll.setSingleStep(1.0)
        self.view_roll.setValue(0.0)

        right_controls_layout.addRow("Elevation", self.view_elev)
        right_controls_layout.addRow("Azimuth", self.view_azim)
        right_controls_layout.addRow("Roll", self.view_roll)

        def _apply_view_from_ui() -> None:
            self.preview.apply_view(
                float(self.view_elev.value()),
                float(self.view_azim.value()),
                float(self.view_roll.value()),
            )

        self.view_elev.valueChanged.connect(lambda *_: _apply_view_from_ui())
        self.view_azim.valueChanged.connect(lambda *_: _apply_view_from_ui())
        self.view_roll.valueChanged.connect(lambda *_: _apply_view_from_ui())

        preset_row = QtWidgets.QHBoxLayout()
        self.view_xy_btn = QtWidgets.QPushButton("XY")
        self.view_xz_btn = QtWidgets.QPushButton("XZ")
        self.view_yz_btn = QtWidgets.QPushButton("YZ")
        self.view_reset_btn = QtWidgets.QPushButton("Reset")
        preset_row.addWidget(self.view_xy_btn)
        preset_row.addWidget(self.view_xz_btn)
        preset_row.addWidget(self.view_yz_btn)
        preset_row.addWidget(self.view_reset_btn)
        preset_widget = QtWidgets.QWidget()
        preset_widget.setLayout(preset_row)
        right_controls_layout.addRow("Presets", preset_widget)

        right_layout.addWidget(right_controls, 0)

        # Move series editors scroll to the right
        right_layout.addWidget(self.series_scroll, 1)

        # Move export buttons to the right
        right_layout.addLayout(btn_row)

        # ---- Main 3-way splitter ----
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(controls)      # left
        splitter.addWidget(self.preview)  # centre
        splitter.addWidget(right)         # right

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 1)
        main_layout.addWidget(splitter)


        self.setStatusBar(QtWidgets.QStatusBar(self))

        # --- Wire events ---
        self.plot_family_combo.currentIndexChanged.connect(self._on_plot_family_changed)
        self.plot_type_combo.currentIndexChanged.connect(self._on_plot_type_changed)

        self.series_spin.valueChanged.connect(self._on_series_count_changed)

        for w in [self.title_edit, self.xlab_edit, self.ylab_edit]:
            w.textChanged.connect(self._schedule_rebuild)
        self.grid_check.toggled.connect(self._schedule_rebuild)
        self.minor_ticks_check.toggled.connect(self._schedule_rebuild)
        self.minor_grid_check.toggled.connect(self._schedule_rebuild)
        self.legend_check.toggled.connect(self._schedule_rebuild)
        self.base_font_spin.valueChanged.connect(self._schedule_rebuild)
        self.title_font_spin.valueChanged.connect(self._schedule_rebuild)

        self.font_combo.currentTextChanged.connect(self._schedule_rebuild)
        self.title_bold.toggled.connect(self._schedule_rebuild)
        self.title_italic.toggled.connect(self._schedule_rebuild)
        self.title_underline.toggled.connect(self._schedule_rebuild)

        self.outlier_sigma_spin.valueChanged.connect(self._schedule_rebuild)
        self.outlier_method_combo.currentIndexChanged.connect(self._schedule_rebuild)

        self.view_reset_btn.clicked.connect(self._reset_view)
        self.view_xy_btn.clicked.connect(self._view_xy)
        self.view_xz_btn.clicked.connect(self._view_xz)
        self.view_yz_btn.clicked.connect(self._view_yz)

        self.export_fig_btn.clicked.connect(self._export_figure)
        self.export_code_btn.clicked.connect(self._export_code)

        # Sync combos to current spec defaults
        self._set_view_controls_visible_for_plot_type(self._current_plot_type())

        self._rebuild_series_editors()
        self._rebuild_preview()

    # ---- NEW helpers for 2 dropdowns ----

    def _current_plot_family(self) -> PlotFamily:
        return PlotFamily(self.plot_family_combo.currentData())

    def _current_plot_type(self) -> PlotType:
        return PlotType(self.plot_type_combo.currentData())

    def _rebuild_plot_type_combo(self, family: PlotFamily) -> None:
        self.plot_type_combo.blockSignals(True)
        self.plot_type_combo.clear()

        types = types_for_family(family)
        for pt in types:
            self.plot_type_combo.addItem(meta_for(pt).label, pt.value)

        # If no types registered for this family, fall back to BASIC/LINE
        if not types:
            fallback = default_type_for_family(PlotFamily.BASIC)
            self.plot_type_combo.addItem(meta_for(fallback).label, fallback.value)

        self.plot_type_combo.blockSignals(False)

    def _sync_family_type_ui_from_spec(self, spec: PlotSpec) -> None:
        spec = spec.normalised()

        # If spec refers to a family that isn't currently in the dropdown, fall back to BASIC.
        fam_value = spec.plot_family.value
        idx = self.plot_family_combo.findData(fam_value)
        if idx < 0:
            fam_value = PlotFamily.BASIC.value
            idx = self.plot_family_combo.findData(fam_value)

        self.plot_family_combo.setCurrentIndex(idx if idx >= 0 else 0)

        fam = PlotFamily(self.plot_family_combo.currentData())
        self._rebuild_plot_type_combo(fam)

        type_idx = self.plot_type_combo.findData(spec.plot_type.value)
        if type_idx < 0:
            # Fall back to family default
            pt = default_type_for_family(fam)
            type_idx = self.plot_type_combo.findData(pt.value)

        self.plot_type_combo.setCurrentIndex(type_idx if type_idx >= 0 else 0)


    def _on_plot_family_changed(self) -> None:
        fam = self._current_plot_family()
        self._rebuild_plot_type_combo(fam)

        # Snap to the family's default type
        pt_default = default_type_for_family(fam)
        idx = self.plot_type_combo.findData(pt_default.value)
        if idx >= 0:
            self.plot_type_combo.setCurrentIndex(idx)

        # Now read the actual current type and update UI
        pt_now = self._current_plot_type()
        self._set_view_controls_visible_for_plot_type(pt_now)

        # Apply to series editors
        for ed in getattr(self, "_series_editors", []):
            ed.apply_plot_type(pt_now)

        self._schedule_rebuild()


    def _on_plot_type_changed(self) -> None:
        pt = self._current_plot_type()

        self._set_view_controls_visible_for_plot_type(pt)

        for ed in getattr(self, "_series_editors", []):
            ed.apply_plot_type(pt)

        self._schedule_rebuild()


    # ---- Existing behaviour ----

    def _schedule_rebuild(self, *_args) -> None:
        self._debounce_timer.start(120)

    def _on_series_count_changed(self, value: int) -> None:
        prev_spec = self._collect_spec_from_ui()
        old = prev_spec.series_count
        new = int(value)

        self._spec = replace(prev_spec, series_count=new).normalised()
        self._rebuild_series_editors()

        if new > old:
            self.statusBar().showMessage(f"Added Series {new}", 1600)
            QtCore.QTimer.singleShot(0, self._scroll_to_bottom)
            if self._series_editors:
                self._series_editors[-1].flash()
                self._series_editors[-1].label_edit.setFocus()
        else:
            self.statusBar().showMessage(f"Series count: {new}", 1600)

        self._schedule_rebuild()

    def _scroll_to_bottom(self) -> None:
        sb = self.series_scroll.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_have_data_toggled(self, checked: bool) -> None:
        self._spec = replace(self._spec, you_got_data=checked).normalised()
        self._schedule_rebuild()

    def _rebuild_series_editors(self) -> None:
        while self.series_layout.count():
            item = self.series_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        self._series_editors = []
        spec = self._spec.normalised()
        pt = spec.plot_type

        for i in range(spec.series_count):
            ed = SeriesEditor(i)
            if i < len(spec.series):
                ed.from_series_spec(spec.series[i])
            ed.apply_plot_type(pt)
            ed.changed.connect(self._schedule_rebuild)
            self.series_layout.addWidget(ed)
            self._series_editors.append(ed)

        self.series_layout.addStretch(1)


    def _collect_spec_from_ui(self) -> PlotSpec:
        plot_family = PlotFamily(self.plot_family_combo.currentData())
        plot_type = PlotType(self.plot_type_combo.currentData())
        series_count = int(self.series_spin.value())

        series_specs = [ed.to_series_spec() for ed in self._series_editors]

        # ---- parse title offset from text box ----
        raw_offset = self.title_offset_edit.text().strip()
        if raw_offset == "":
            title_offset = None
        else:
            try:
                title_offset = float(raw_offset)
            except ValueError:
                raise ValueError("Title offset must be a number")

        style = StyleSpec(
            font_family=self.font_combo.currentText().strip(),

            title=self.title_edit.text(),
            title_bold=self.title_bold.isChecked(),
            title_italic=self.title_italic.isChecked(),
            title_underline=self.title_underline.isChecked(),
            title_offset=title_offset,

            x_label=self.xlab_edit.text(),
            y_label=self.ylab_edit.text(),

            x_tick_label_angle=int(self.x_tick_angle_spin.value()),
            y_tick_label_angle=int(self.y_tick_angle_spin.value()),

            show_grid=self.grid_check.isChecked(),
            show_minor_ticks=self.minor_ticks_check.isChecked(),
            show_minor_grid=self.minor_grid_check.isChecked(),
            show_legend=self.legend_check.isChecked(),

            base_font_size=int(self.base_font_spin.value()),
            title_font_size=int(self.title_font_spin.value()),

            outlier_sigma=float(self.outlier_sigma_spin.value()),
            outlier_method=str(self.outlier_method_combo.currentData() or "std"),
        )

        return PlotSpec(
            plot_family=plot_family,
            plot_type=plot_type,
            you_got_data=True,
            series_count=series_count,
            series=series_specs,
            style=style,
        )


    def _rebuild_preview(self) -> None:
        self._spec = self._collect_spec_from_ui()

        # draw plot
        self.preview.render(self._spec)

        # re-apply camera if 3D (render recreates axes)
        self.preview.apply_view(
            float(self.view_elev.value()),
            float(self.view_azim.value()),
            float(self.view_roll.value()),
        )


    def _export_figure(self) -> None:
        self._spec = self._collect_spec_from_ui()
        self._rebuild_preview()

        filters = "PNG (*.png);PDF (*.pdf);SVG (*.svg)"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export figure", "figure.png", filters)
        if not path:
            return

        dpi = 300
        if path.lower().endswith(".png"):
            dpi, ok = QtWidgets.QInputDialog.getInt(self, "PNG DPI", "DPI", 300, 72, 1200, 1)
            if not ok:
                return

        self.preview.save_figure(path, dpi=int(dpi))
        self.statusBar().showMessage(f"Saved {path}", 1800)

    def _export_code(self) -> None:
        self._spec = self._collect_spec_from_ui()
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose export folder")
        if not out_dir:
            return

        export_path = export_code_scaffold(self._spec, out_dir)
        QtWidgets.QMessageBox.information(
            self,
            "Export complete",
            f"Wrote:\n{export_path}\n\nRun:\npython run_plot.py --show\nor\npython run_plot.py --out figure.pdf",
        )

    def _set_view(self, elev: float, azim: float, roll: float = 0.0) -> None:
        self.view_elev.blockSignals(True)
        self.view_azim.blockSignals(True)
        self.view_roll.blockSignals(True)
        self.view_elev.setValue(elev)
        self.view_azim.setValue(azim)
        self.view_roll.setValue(roll)
        self.view_roll.blockSignals(False)
        self.view_azim.blockSignals(False)
        self.view_elev.blockSignals(False)

        self.preview.apply_view(elev, azim, roll)

    # Suggested defaults (tweak as you like)
    def _reset_view(self) -> None:
        self._set_view(30.0, -60.0, 0.0)

    def _view_xy(self) -> None:
        # looking down z axis
        self._set_view(90.0, -90.0, 0.0)

    def _view_xz(self) -> None:
        # looking down y axis
        self._set_view(0.0, -90.0, 0.0)

    def _view_yz(self) -> None:
        # looking down x axis
        self._set_view(0.0, 0.0, 0.0)

    def _set_view_controls_visible_for_plot_type(self, pt: PlotType) -> None:
        is_3d = bool(getattr(meta_for(pt), "requires_z", False))
        for w in [
            self.view_elev, self.view_azim, self.view_roll,
            self.view_reset_btn, self.view_xy_btn, self.view_xz_btn, self.view_yz_btn,
        ]:
            w.setVisible(is_3d)



def main() -> None:
    app = QtWidgets.QApplication([])
    w = MainWindow()
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
