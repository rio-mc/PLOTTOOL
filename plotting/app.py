from __future__ import annotations

from dataclasses import replace
from typing import List, Optional

from PySide6 import QtCore, QtWidgets

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import font_manager

from .spec import PlotSpec, PlotType, StyleSpec, SeriesSpec, SeriesInlineData, SeriesStyleSpec
from .builder import build_series_data, draw
from .export_code import export_code_scaffold
from .plot_types import meta_for
from . import export_dialogs as ExportDialog

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
        if pt in (PlotType.LINE3D, PlotType.SCATTER3D):
            self._ax = self._fig.add_subplot(111, projection="3d")
        else:
            self._ax = self._fig.add_subplot(111)

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

    def __init__(self, index: int, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._index = index
        self.setTitle(f"Series {index + 1}")

        self.label_edit = QtWidgets.QLineEdit()
        self.label_edit.setPlaceholderText(f"Series {index + 1}")

        # Data boxes
        self.x_edit = QtWidgets.QPlainTextEdit()
        self.y_edit = QtWidgets.QPlainTextEdit()
        self.x_edit.setPlaceholderText("x values, e.g.\n0, 1, 2, 3\nor\n0 1 2 3")
        self.y_edit.setPlaceholderText("y values, e.g.\n0.1, 0.4, 0.2, 0.9")

        # Std controls (per series)
        self.x_std_check = QtWidgets.QCheckBox("Use x std")
        self.y_std_check = QtWidgets.QCheckBox("Use y std")

        self.x_std_edit = QtWidgets.QPlainTextEdit()
        self.y_std_edit = QtWidgets.QPlainTextEdit()
        self.x_std_edit.setPlaceholderText("x std (same length as x)")
        self.y_std_edit.setPlaceholderText("y std (same length as y)")
        self.x_std_edit.setEnabled(False)
        self.y_std_edit.setEnabled(False)

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

        # Layout
        form = QtWidgets.QFormLayout()
        form.addRow("Label", self.label_edit)

        self.x_label = QtWidgets.QLabel("x")
        self.y_label = QtWidgets.QLabel("y")
        form.addRow(self.x_label, self.x_edit)
        form.addRow(self.y_label, self.y_edit)
        self.z_edit = QtWidgets.QPlainTextEdit()
        self.z_edit.setPlaceholderText("z values, e.g.\n0, 1, 2, 3")
        self.z_label = QtWidgets.QLabel("z")

        # Std rows
        form.addRow(self.x_std_check, self.x_std_edit)
        form.addRow(self.y_std_check, self.y_std_edit)



        color_row = QtWidgets.QHBoxLayout()
        color_row.setContentsMargins(0, 0, 0, 0)
        color_row.addWidget(self.color_edit, 1)
        color_row.addWidget(self.color_btn)

        color_row_widget = QtWidgets.QWidget()
        color_row_widget.setLayout(color_row)

        self.color_label = QtWidgets.QLabel("Colour")
        form.addRow(self.color_label, color_row_widget)

        # Keep a reference so we can hide/show the whole row cleanly if needed
        self.color_row_widget = color_row_widget

        self.marker_label = QtWidgets.QLabel("Marker")
        self.marker_size_label = QtWidgets.QLabel("Marker size")
        self.line_width_label = QtWidgets.QLabel("Line width")
        self.line_style_label = QtWidgets.QLabel("Line style")

        form.addRow(self.z_label, self.z_edit)
        self.z_edit.textChanged.connect(lambda *_: self.changed.emit())

        form.addRow(self.marker_label, self.marker_combo)
        form.addRow(self.marker_size_label, self.marker_size)
        form.addRow(self.line_width_label, self.line_width)
        form.addRow(self.line_style_label, self.line_style)

        self.setLayout(form)

        # Signals
        self.label_edit.textChanged.connect(lambda *_: self.changed.emit())
        self.x_edit.textChanged.connect(lambda *_: self.changed.emit())
        self.y_edit.textChanged.connect(lambda *_: self.changed.emit())

        self.x_std_check.toggled.connect(self._on_x_std_toggled)
        self.y_std_check.toggled.connect(self._on_y_std_toggled)
        self.x_std_edit.textChanged.connect(lambda *_: self.changed.emit())
        self.y_std_edit.textChanged.connect(lambda *_: self.changed.emit())

        self.color_edit.textChanged.connect(lambda *_: self.changed.emit())
        self.marker_combo.currentIndexChanged.connect(lambda *_: self.changed.emit())
        self.marker_size.valueChanged.connect(lambda *_: self.changed.emit())
        self.line_width.valueChanged.connect(lambda *_: self.changed.emit())
        self.line_style.currentIndexChanged.connect(lambda *_: self.changed.emit())

        self.color_btn.clicked.connect(self._pick_colour)

    def _pick_colour(self) -> None:
        col = QtWidgets.QColorDialog.getColor(parent=self)
        if not col.isValid():
            return
        self.color_edit.setText(col.name())  # #RRGGBB

    def _on_x_std_toggled(self, checked: bool) -> None:
        self.x_std_edit.setEnabled(bool(checked) and self.x_std_edit.isVisible() and self.x_std_check.isEnabled())
        self.changed.emit()

    def _on_y_std_toggled(self, checked: bool) -> None:
        self.y_std_edit.setEnabled(bool(checked) and self.y_std_edit.isVisible() and self.y_std_check.isEnabled())
        self.changed.emit()

    def set_data_enabled(self, enabled: bool) -> None:
        # Keep styling editable even if the user has no data so the dummy preview remains designable.
        self.x_edit.setEnabled(enabled)
        self.y_edit.setEnabled(enabled)
        self.z_edit.setEnabled(enabled and self.z_edit.isVisible())

        self.x_std_check.setEnabled(enabled and self.x_std_check.isVisible())
        self.y_std_check.setEnabled(enabled and self.y_std_check.isVisible())

        self.x_std_edit.setEnabled(enabled and self.x_std_check.isChecked() and self.x_std_edit.isVisible())
        self.y_std_edit.setEnabled(enabled and self.y_std_check.isChecked() and self.y_std_edit.isVisible())

    def flash(self, duration_ms: int = 650) -> None:
        original = self.styleSheet()
        self.setStyleSheet("QGroupBox { border: 2px solid palette(highlight); }")
        QtCore.QTimer.singleShot(duration_ms, lambda: self.setStyleSheet(original))

    def to_series_spec(self) -> SeriesSpec:
        label = self.label_edit.text().strip() or f"Series {self._index + 1}"
        return SeriesSpec(
            label=label,
            use_x_std=bool(self.x_std_check.isChecked()),
            use_y_std=bool(self.y_std_check.isChecked()),
            inline=SeriesInlineData(
                x_text=self.x_edit.toPlainText(),
                y_text=self.y_edit.toPlainText(),
                z_text=self.z_edit.toPlainText(),
                x_std_text=self.x_std_edit.toPlainText(),
                y_std_text=self.y_std_edit.toPlainText(),
            ),
            style=SeriesStyleSpec(
                color=self.color_edit.text().strip(),
                marker=str(self.marker_combo.currentData()),
                marker_size=float(self.marker_size.value()),
                line_width=float(self.line_width.value()),
                line_style=str(self.line_style.currentData()),
            ),
        )

    def from_series_spec(self, spec: SeriesSpec) -> None:
        self.label_edit.setText(spec.label)
        self.x_edit.setPlainText(spec.inline.x_text)
        self.y_edit.setPlainText(spec.inline.y_text)
        self.z_edit.setPlainText(getattr(spec.inline, "z_text", ""))

        self.x_std_edit.setPlainText(getattr(spec.inline, "x_std_text", ""))
        self.y_std_edit.setPlainText(getattr(spec.inline, "y_std_text", ""))
        self.x_std_check.setChecked(bool(getattr(spec, "use_x_std", False)))
        self.y_std_check.setChecked(bool(getattr(spec, "use_y_std", False)))

        self.color_edit.setText(spec.style.color)

        marker_code = spec.style.marker
        idx = self.marker_combo.findData(marker_code)
        self.marker_combo.setCurrentIndex(idx if idx >= 0 else 0)

        self.marker_size.setValue(float(spec.style.marker_size))
        self.line_width.setValue(float(spec.style.line_width))

        ls_code = spec.style.line_style
        idx = self.line_style.findData(ls_code)
        self.line_style.setCurrentIndex(idx if idx >= 0 else 0)

    def apply_plot_type(self, plot_type: PlotType) -> None:
        m = meta_for(plot_type)

        # Input visibility
        self.x_label.setVisible(m.requires_x)
        self.x_edit.setVisible(m.requires_x)

        # NEW: z input visibility (3D)
        requires_z = bool(getattr(m, "requires_z", False))
        self.z_label.setVisible(requires_z)
        self.z_edit.setVisible(requires_z)

        # For y-only plots, tweak placeholder
        if not m.requires_x:
            self.y_edit.setPlaceholderText("values, e.g.\n0.1, 0.4, 0.2, 0.9")
        else:
            self.y_edit.setPlaceholderText("y values, e.g.\n0.1, 0.4, 0.2, 0.9")

        # Std visibility depends on plot type capability + whether x/y exists
        show_x_std = bool(m.supports_x_std and m.requires_x)
        show_y_std = bool(m.supports_y_std and m.requires_y)

        self.x_std_check.setVisible(show_x_std)
        self.x_std_edit.setVisible(show_x_std)
        self.y_std_check.setVisible(show_y_std)
        self.y_std_edit.setVisible(show_y_std)

        if not show_x_std:
            self.x_std_check.setChecked(False)
            self.x_std_edit.setEnabled(False)
        else:
            self.x_std_edit.setEnabled(self.x_std_check.isChecked() and self.x_std_check.isEnabled())

        if not show_y_std:
            self.y_std_check.setChecked(False)
            self.y_std_edit.setEnabled(False)
        else:
            self.y_std_edit.setEnabled(self.y_std_check.isChecked() and self.y_std_check.isEnabled())

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

        # Plot type
        self.plot_type_combo = QtWidgets.QComboBox()
        self.plot_type_combo.addItem("Line", PlotType.LINE.value)
        self.plot_type_combo.addItem("Scatter", PlotType.SCATTER.value)
        self.plot_type_combo.addItem("Step", PlotType.STEP.value)
        self.plot_type_combo.addItem("Area", PlotType.AREA.value)
        self.plot_type_combo.addItem("Bar", PlotType.BAR.value)
        self.plot_type_combo.addItem("Histogram", PlotType.HIST.value)
        self.plot_type_combo.addItem("Box", PlotType.BOX.value)
        self.plot_type_combo.addItem("Violin", PlotType.VIOLIN.value)
        
        self.plot_type_combo.addItem("Line 3D", PlotType.LINE3D.value)
        self.plot_type_combo.addItem("Scatter 3D", PlotType.SCATTER3D.value)

        # Series count
        self.series_spin = QtWidgets.QSpinBox()
        self.series_spin.setRange(1, 10)
        self.series_spin.setValue(1)

        # You got data
        self.have_data_check = QtWidgets.QCheckBox("You got data?")
        self.have_data_check.setChecked(False)

        top_form = QtWidgets.QFormLayout()
        top_form.addRow("Plot type", self.plot_type_combo)
        top_form.addRow("Series", self.series_spin)
        top_form.addRow("", self.have_data_check)
        controls_layout.addLayout(top_form)

        # Style controls
        style_box = QtWidgets.QGroupBox("Style")
        style_form = QtWidgets.QFormLayout()

        # Font applies to whole graph
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

        self.grid_check = QtWidgets.QCheckBox("Grid")
        self.grid_check.setChecked(True)
        self.legend_check = QtWidgets.QCheckBox("Legend")
        self.legend_check.setChecked(True)

        self.font_spin = QtWidgets.QSpinBox()
        self.font_spin.setRange(7, 24)
        self.font_spin.setValue(11)

        style_form.addRow("Font", self.font_combo)
        style_form.addRow("Title", title_row_widget)
        style_form.addRow("x label", self.xlab_edit)
        style_form.addRow("y label", self.ylab_edit)
        style_form.addRow("", self.grid_check)
        style_form.addRow("", self.legend_check)
        style_form.addRow("Font size", self.font_spin)

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
        controls_layout.addWidget(self.series_scroll, 1)

        # Export buttons
        btn_row = QtWidgets.QHBoxLayout()
        self.export_fig_btn = QtWidgets.QPushButton("Export figure")
        self.export_code_btn = QtWidgets.QPushButton("Export code")
        btn_row.addWidget(self.export_fig_btn)
        btn_row.addWidget(self.export_code_btn)
        controls_layout.addLayout(btn_row)

        # Issue panel
        self.issues_box = QtWidgets.QGroupBox("Messages")
        self.issues_text = QtWidgets.QPlainTextEdit()
        self.issues_text.setReadOnly(True)
        issues_layout = QtWidgets.QVBoxLayout()
        issues_layout.addWidget(self.issues_text)
        self.issues_box.setLayout(issues_layout)
        controls_layout.addWidget(self.issues_box)

        # Right: preview
        self.preview = MatplotlibPreview()

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(controls)
        splitter.addWidget(self.preview)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)

        self.setStatusBar(QtWidgets.QStatusBar(self))

        # Wire events
        self.plot_type_combo.currentIndexChanged.connect(self._on_plot_type_changed)
        self.series_spin.valueChanged.connect(self._on_series_count_changed)
        self.have_data_check.toggled.connect(self._on_have_data_toggled)

        for w in [self.title_edit, self.xlab_edit, self.ylab_edit]:
            w.textChanged.connect(self._schedule_rebuild)
        self.grid_check.toggled.connect(self._schedule_rebuild)
        self.legend_check.toggled.connect(self._schedule_rebuild)
        self.font_spin.valueChanged.connect(self._schedule_rebuild)

        self.font_combo.currentTextChanged.connect(self._schedule_rebuild)
        self.title_bold.toggled.connect(self._schedule_rebuild)
        self.title_italic.toggled.connect(self._schedule_rebuild)
        self.title_underline.toggled.connect(self._schedule_rebuild)

        self.export_fig_btn.clicked.connect(self._export_figure)
        self.export_code_btn.clicked.connect(self._export_code)

        self._rebuild_series_editors()
        self._rebuild_preview()

    def _schedule_rebuild(self,) -> None:
        self._debounce_timer.start(120)

    def _current_plot_type(self) -> PlotType:
        return PlotType(self.plot_type_combo.currentData())

    def _on_plot_type_changed(self) -> None:
        pt = self._current_plot_type()
        for ed in getattr(self, "_series_editors", []):
            ed.apply_plot_type(pt)
        self._schedule_rebuild()

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
        self._set_series_data_enabled(checked)
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
        self._set_series_data_enabled(spec.you_got_data)

    def _set_series_data_enabled(self, enabled: bool) -> None:
        for ed in getattr(self, "_series_editors", []):
            ed.set_data_enabled(enabled)

    def _collect_spec_from_ui(self) -> PlotSpec:
        plot_type = PlotType(self.plot_type_combo.currentData())
        you_got_data = self.have_data_check.isChecked()
        series_count = int(self.series_spin.value())

        series_specs = [ed.to_series_spec() for ed in self._series_editors]
        style = StyleSpec(
            font_family=self.font_combo.currentText().strip(),
            title=self.title_edit.text(),
            title_bold=self.title_bold.isChecked(),
            title_italic=self.title_italic.isChecked(),
            title_underline=self.title_underline.isChecked(),
            x_label=self.xlab_edit.text(),
            y_label=self.ylab_edit.text(),
            show_grid=self.grid_check.isChecked(),
            show_legend=self.legend_check.isChecked(),
            base_font_size=int(self.font_spin.value()),
        )

        return PlotSpec(
            plot_type=plot_type,
            you_got_data=you_got_data,
            series_count=series_count,
            series=series_specs,
            style=style,
        ).normalised()

    def _rebuild_preview(self) -> None:
        self._spec = self._collect_spec_from_ui()
        messages = self.preview.render(self._spec)
        self.issues_text.setPlainText("No issues." if not messages else "\n".join(messages))

    def _export_figure(self) -> None:
        self._spec = self._collect_spec_from_ui()
        self._rebuild_preview()

        dlg = ExportDialog.ExportFigureDialog(self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return

        fmt, dpi, jpg_q = dlg.values()

        # pick a path based on chosen format
        default_name = f"figure.{fmt}"
        filt = {
            "png": "PNG (*.png)",
            "jpg": "JPG (*.jpg *.jpeg)",
            "pdf": "PDF (*.pdf)",
        }[fmt]

        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export figure", default_name, filt)
        if not path:
            return

        lower = path.lower()
        if fmt == "png" and not lower.endswith(".png"):
            path += ".png"
        elif fmt == "jpg" and not (lower.endswith(".jpg") or lower.endswith(".jpeg")):
            path += ".jpg"
        elif fmt == "pdf" and not lower.endswith(".pdf"):
            path += ".pdf"

        # Save via Matplotlib; JPG quality is best-effort (depends on Matplotlib/Pillow)
        if fmt in ("png", "jpg"):
            if fmt == "jpg":
                try:
                    self.preview._fig.savefig(path, dpi=dpi, bbox_inches="tight", pil_kwargs={"quality": jpg_q})
                except Exception:
                    # fallback without pil_kwargs
                    self.preview._fig.savefig(path, dpi=dpi, bbox_inches="tight")
            else:
                self.preview._fig.savefig(path, dpi=dpi, bbox_inches="tight")
        else:
            self.preview._fig.savefig(path, bbox_inches="tight")

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


def main() -> None:
    app = QtWidgets.QApplication([])
    w = MainWindow()
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
