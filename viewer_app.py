"""
RAW Image Viewer — PyQt5 메인 윈도우
"""

import os
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QComboBox, QPushButton,
    QScrollArea, QStatusBar, QMenuBar, QAction,
    QFileDialog, QMessageBox, QSizePolicy, QFrame,
    QSpinBox, QCheckBox, QGroupBox,
)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QImage, QPixmap, QKeySequence

import numpy as np

from image_loader import load_raw, load_mipi_image


FORMATS = [
    "YUV420p",
    "NV12",
    "NV21",
    "RGB24",
    "RGB10",
    "RGB12",
]

ZOOM_PRESETS = {
    Qt.Key_1: 0.50,
    Qt.Key_2: 1.00,
    Qt.Key_3: 1.50,
    Qt.Key_4: 2.00,
    Qt.Key_5: 2.50,
    Qt.Key_6: 3.00,
}

ZOOM_STEP = 0.10
ZOOM_MIN  = 0.10
ZOOM_MAX  = 8.00


class ImageLabel(QLabel):
    """줌·스크롤을 지원하는 이미지 표시 위젯"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self._pixmap_orig: QPixmap | None = None

    def set_pixmap(self, pixmap):
        self._pixmap_orig = pixmap
        self._apply_zoom(1.0)  # 내부용; 실제 zoom은 viewer_app 에서 관리

    def render_zoom(self, zoom: float):
        if self._pixmap_orig is None:
            return
        w = int(self._pixmap_orig.width()  * zoom)
        h = int(self._pixmap_orig.height() * zoom)
        scaled = self._pixmap_orig.scaled(
            w, h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.setPixmap(scaled)
        self.resize(scaled.size())

    def _apply_zoom(self, zoom: float):
        self.render_zoom(zoom)

    def has_image(self) -> bool:
        return self._pixmap_orig is not None


class RawViewerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAW Image Viewer")
        self.resize(1100, 700)

        self._file_path = ""
        self._zoom = 1.0
        self._bgr_array = None
        self._mipi_demosaic = False  # False=Raw Bayer, True=Demosaic→RGB

        self._build_menu()
        self._build_ui()
        self._build_statusbar()

    # ── 메뉴 ─────────────────────────────────────────────────────────────────

    def _build_menu(self):
        menu_bar: QMenuBar = self.menuBar()
        file_menu = menu_bar.addMenu("File")

        open_action = QAction("Open Raw File…", self)
        open_action.setShortcut(QKeySequence("Ctrl+O"))
        open_action.triggered.connect(self._open_file)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence("Alt+F4"))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(8)

        # 사이드 패널
        side = self._build_side_panel()
        main_layout.addWidget(side)

        # 이미지 뷰
        self._scroll = QScrollArea()
        self._scroll.setAlignment(Qt.AlignCenter)
        self._scroll.setWidgetResizable(False)
        self._scroll.setFrameShape(QFrame.StyledPanel)

        self._img_label = ImageLabel()
        self._img_label.setText("이미지를 불러오세요\n(File > Open Raw File)")
        self._img_label.setStyleSheet("color: #888; font-size: 14px;")
        self._img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._scroll.setWidget(self._img_label)

        # 마우스 휠 이벤트를 스크롤 영역에 설치
        self._scroll.viewport().installEventFilter(self)

        main_layout.addWidget(self._scroll, stretch=1)

    def _build_side_panel(self) -> QWidget:
        panel = QFrame()
        panel.setFrameShape(QFrame.StyledPanel)
        panel.setFixedWidth(230)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 10, 8, 10)
        layout.setSpacing(8)

        # Width
        layout.addWidget(QLabel("Width (px)"))
        self._width_spin = QSpinBox()
        self._width_spin.setRange(1, 65535)
        self._width_spin.setValue(1920)
        layout.addWidget(self._width_spin)

        # Height
        layout.addWidget(QLabel("Height (px)"))
        self._height_spin = QSpinBox()
        self._height_spin.setRange(1, 65535)
        self._height_spin.setValue(1080)
        layout.addWidget(self._height_spin)

        # Format
        layout.addWidget(QLabel("Format"))
        self._fmt_combo = QComboBox()
        self._fmt_combo.addItems(FORMATS)
        layout.addWidget(self._fmt_combo)

        # ── MIPI Override ─────────────────────────────────────────────────
        mipi_group = QGroupBox("MIPI Override")
        mipi_layout = QVBoxLayout(mipi_group)
        mipi_layout.setContentsMargins(6, 8, 6, 8)
        mipi_layout.setSpacing(6)

        self._mipi_override_cb = QCheckBox("MIPI로 강제 해석")
        self._mipi_override_cb.setChecked(False)
        self._mipi_override_cb.stateChanged.connect(self._on_mipi_override_changed)
        mipi_layout.addWidget(self._mipi_override_cb)

        depth_row = QHBoxLayout()
        depth_row.addWidget(QLabel("Bit depth:"))
        self._mipi_depth_combo = QComboBox()
        self._mipi_depth_combo.addItems(["MIPI_RAW8", "MIPI_RAW10", "MIPI_RAW12", "MIPI_RAW14"])
        self._mipi_depth_combo.setCurrentText("MIPI_RAW10")
        depth_row.addWidget(self._mipi_depth_combo)
        mipi_layout.addLayout(depth_row)

        bayer_row = QHBoxLayout()
        bayer_row.addWidget(QLabel("Bayer:"))
        self._mipi_bayer_combo = QComboBox()
        self._mipi_bayer_combo.addItems(["RGGB", "GRBG", "GBRG", "BGGR"])
        bayer_row.addWidget(self._mipi_bayer_combo)
        mipi_layout.addLayout(bayer_row)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        self._btn_raw_bayer = QPushButton("Raw Bayer")
        self._btn_raw_bayer.setCheckable(True)
        self._btn_raw_bayer.setChecked(True)
        self._btn_raw_bayer.clicked.connect(lambda: self._set_mipi_demosaic(False))
        self._btn_demosaic = QPushButton("Demosaic")
        self._btn_demosaic.setCheckable(True)
        self._btn_demosaic.setChecked(False)
        self._btn_demosaic.clicked.connect(lambda: self._set_mipi_demosaic(True))
        btn_row.addWidget(self._btn_raw_bayer)
        btn_row.addWidget(self._btn_demosaic)
        mipi_layout.addLayout(btn_row)

        layout.addWidget(mipi_group)

        # 초기 상태: override OFF → MIPI 컨트롤 비활성
        self._set_mipi_controls_enabled(False)

        # Zoom 표시
        layout.addSpacing(6)
        self._zoom_label = QLabel("Zoom: 100%")
        self._zoom_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._zoom_label)

        # 줌 버튼 행
        zoom_row = QHBoxLayout()
        btn_zoom_out = QPushButton("-")
        btn_zoom_out.setFixedWidth(40)
        btn_zoom_out.clicked.connect(lambda: self._change_zoom(-ZOOM_STEP))
        btn_zoom_in = QPushButton("+")
        btn_zoom_in.setFixedWidth(40)
        btn_zoom_in.clicked.connect(lambda: self._change_zoom(+ZOOM_STEP))
        zoom_row.addWidget(btn_zoom_out)
        zoom_row.addStretch()
        zoom_row.addWidget(btn_zoom_in)
        layout.addLayout(zoom_row)

        # Load 버튼
        layout.addSpacing(6)
        self._load_btn = QPushButton("Load / Render")
        self._load_btn.setFixedHeight(36)
        self._load_btn.clicked.connect(self._load_and_render)
        layout.addWidget(self._load_btn)

        layout.addStretch()

        # 단축키 안내
        hint = QLabel(
            "숫자키 1~6: 고정 줌\n"
            "휠: 줌 인/아웃\n"
            "Ctrl+O: 파일 열기"
        )
        hint.setStyleSheet("color: #999; font-size: 10px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        return panel

    def _set_mipi_controls_enabled(self, enabled: bool):
        self._mipi_depth_combo.setEnabled(enabled)
        self._mipi_bayer_combo.setEnabled(enabled)
        self._btn_raw_bayer.setEnabled(enabled)
        self._btn_demosaic.setEnabled(enabled)

    def _on_mipi_override_changed(self, state):
        enabled = (state == Qt.Checked)
        self._set_mipi_controls_enabled(enabled)
        if self._file_path:
            self._load_and_render()

    def _set_mipi_demosaic(self, demosaic: bool):
        self._mipi_demosaic = demosaic
        self._btn_raw_bayer.setChecked(not demosaic)
        self._btn_demosaic.setChecked(demosaic)
        if self._file_path:
            self._load_and_render()

    # ── 상태바 ────────────────────────────────────────────────────────────────

    def _build_statusbar(self):
        self._statusbar: QStatusBar = self.statusBar()
        self._status_file  = QLabel("파일: —")
        self._status_fmt   = QLabel("포맷: —")
        self._status_res   = QLabel("해상도: —")
        self._status_zoom  = QLabel("줌: 100%")

        for lbl in (self._status_file, self._status_fmt,
                    self._status_res, self._status_zoom):
            self._statusbar.addPermanentWidget(lbl)

    def _update_status(self):
        fname = os.path.basename(self._file_path) if self._file_path else "—"
        self._status_file.setText(f"파일: {fname}")
        if self._mipi_override_cb.isChecked():
            mode = "Demosaic" if self._mipi_demosaic else "Raw Bayer"
            fmt_str = f"{self._mipi_depth_combo.currentText()} [{self._mipi_bayer_combo.currentText()}] {mode}"
        else:
            fmt_str = self._fmt_combo.currentText()
        self._status_fmt.setText(f"포맷: {fmt_str}")
        w = self._width_spin.value()
        h = self._height_spin.value()
        self._status_res.setText(f"해상도: {w}×{h}")
        pct = int(self._zoom * 100)
        self._status_zoom.setText(f"줌: {pct}%")
        self._zoom_label.setText(f"Zoom: {pct}%")

    # ── 파일 열기 ─────────────────────────────────────────────────────────────

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "RAW 파일 열기",
            "",
            "RAW Files (*.raw *.bin *.yuv *.rgb *.data);;All Files (*)",
        )
        if not path:
            return
        self._file_path = path
        self._load_and_render()

    # ── 로딩 + 렌더링 ─────────────────────────────────────────────────────────

    def _load_and_render(self):
        if not self._file_path:
            QMessageBox.information(self, "안내", "먼저 RAW 파일을 선택해 주세요.")
            return

        w = self._width_spin.value()
        h = self._height_spin.value()

        try:
            if self._mipi_override_cb.isChecked():
                mipi_fmt = self._mipi_depth_combo.currentText()
                bayer    = self._mipi_bayer_combo.currentText()
                bgr = load_mipi_image(self._file_path, w, h, mipi_fmt, bayer, self._mipi_demosaic)
            else:
                fmt = self._fmt_combo.currentText()
                bgr = load_raw(self._file_path, w, h, fmt)
        except Exception as e:
            QMessageBox.critical(self, "로딩 오류", str(e))
            return

        self._bgr_array = bgr

        # BGR → RGB → QImage → QPixmap
        rgb = bgr[:, :, ::-1].copy()
        qimg = QImage(
            rgb.data,
            rgb.shape[1], rgb.shape[0],
            rgb.strides[0],
            QImage.Format_RGB888,
        )
        pixmap = QPixmap.fromImage(qimg)
        self._img_label.set_pixmap(pixmap)
        self._img_label.setStyleSheet("")  # 안내 텍스트 스타일 제거

        self._apply_zoom(self._zoom)
        self._update_status()

    # ── 줌 ───────────────────────────────────────────────────────────────────

    def _apply_zoom(self, zoom: float):
        zoom = max(ZOOM_MIN, min(ZOOM_MAX, zoom))
        self._zoom = zoom
        if self._img_label.has_image():
            self._img_label.render_zoom(zoom)
            self._center_scrollbars()
        self._update_status()

    def _change_zoom(self, delta: float):
        self._apply_zoom(self._zoom + delta)

    def _center_scrollbars(self):
        """줌 변경 후 스크롤바를 중앙으로"""
        h_bar = self._scroll.horizontalScrollBar()
        v_bar = self._scroll.verticalScrollBar()
        h_bar.setValue((h_bar.minimum() + h_bar.maximum()) // 2)
        v_bar.setValue((v_bar.minimum() + v_bar.maximum()) // 2)

    # ── 이벤트 핸들러 ─────────────────────────────────────────────────────────

    def eventFilter(self, source, event):
        """스크롤 영역 viewport의 휠 이벤트를 가로채 줌에 사용"""
        from PyQt5.QtCore import QEvent
        if source is self._scroll.viewport() and event.type() == QEvent.Wheel:
            delta = event.angleDelta().y()
            if delta > 0:
                self._change_zoom(+ZOOM_STEP)
            else:
                self._change_zoom(-ZOOM_STEP)
            return True  # 이벤트 소비 (스크롤 방지)
        return super().eventFilter(source, event)

    def keyPressEvent(self, event):
        key = event.key()
        if key in ZOOM_PRESETS:
            self._apply_zoom(ZOOM_PRESETS[key])
        else:
            super().keyPressEvent(event)
