from __future__ import annotations

from typing import Optional
from PySide6 import QtWidgets

import matplotlib
matplotlib.use("QtAgg")

class ExportFigureDialog(QtWidgets.QDialog):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export figure")
        self.setModal(True)

        self.format_combo = QtWidgets.QComboBox()
        self.format_combo.addItem("PNG", "png")
        self.format_combo.addItem("JPG", "jpg")
        self.format_combo.addItem("PDF", "pdf")

        self.dpi_spin = QtWidgets.QSpinBox()
        self.dpi_spin.setRange(72, 1200)
        self.dpi_spin.setValue(300)

        self.jpg_quality_spin = QtWidgets.QSpinBox()
        self.jpg_quality_spin.setRange(1, 95)
        self.jpg_quality_spin.setValue(95)

        form = QtWidgets.QFormLayout()
        form.addRow("Format", self.format_combo)
        form.addRow("DPI (raster)", self.dpi_spin)
        form.addRow("JPEG quality", self.jpg_quality_spin)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(btns)
        self.setLayout(layout)

        self.format_combo.currentIndexChanged.connect(self._sync_enabled)
        self._sync_enabled()

    def _sync_enabled(self) -> None:
        fmt = str(self.format_combo.currentData())
        is_jpg = fmt == "jpg"
        is_raster = fmt in ("png", "jpg")
        self.jpg_quality_spin.setEnabled(is_jpg)
        self.dpi_spin.setEnabled(is_raster)

    def values(self) -> tuple[str, int, int]:
        fmt = str(self.format_combo.currentData())
        dpi = int(self.dpi_spin.value())
        jpg_q = int(self.jpg_quality_spin.value())
        return fmt, dpi, jpg_q
