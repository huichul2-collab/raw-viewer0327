"""
RAW Image Viewer — PyQt5 메인 윈도우
"""

import os
import re

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton,
    QScrollArea, QStatusBar, QMenuBar, QAction,
    QFileDialog, QMessageBox, QSizePolicy, QFrame,
    QSpinBox, QSlider,
)
from PyQt5.QtCore import Qt, QRect, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QKeySequence, QPainter, QColor, QFont

import numpy as np

from image_loader import load_raw, apply_gamma, detect_format_from_ext


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
    Qt.Key_5: 4.00,
    Qt.Key_6: 8.00,
    Qt.Key_7: 16.00,
    Qt.Key_8: 32.00,
}

ZOOM_STEP = 0.25
ZOOM_MIN  = 0.10
ZOOM_MAX  = 32.00

# 픽셀 오버레이를 활성화할 최소 줌 배율 (800%)
PIXEL_OVERLAY_ZOOM_THRESHOLD = 8.0

# 픽셀 오버레이 시 표시 크기 상한 (성능 보호)
PIXEL_OVERLAY_MAX_DIM = 8192


# ── 히스토그램 창 ─────────────────────────────────────────────────────────────

class HistogramWindow(QWidget):
    """이미지 렌더링 시 자동 갱신되는 히스토그램 창"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Histogram")
        self.setFixedSize(320, 240)
        self.setWindowFlags(self.windowFlags() | Qt.Tool)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._has_mpl = False
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            self._fig = Figure(figsize=(3.1, 2.2), dpi=96, tight_layout=True)
            self._canvas = FigureCanvas(self._fig)
            layout.addWidget(self._canvas)
            self._has_mpl = True
        except ImportError:
            lbl = QLabel("matplotlib이 설치되지 않았습니다.\npip install matplotlib")
            lbl.setAlignment(Qt.AlignCenter)
            layout.addWidget(lbl)

    def update_histogram(self, bgr_array: np.ndarray, fmt: str):
        if not self._has_mpl or bgr_array is None:
            return

        import cv2

        self._fig.clear()
        ax = self._fig.add_subplot(111)
        is_yuv = fmt.upper() in ('YUV420P', 'NV12', 'NV21')

        if is_yuv:
            ycrcb = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2YCrCb)
            y = ycrcb[:, :, 0].ravel()
            ax.hist(y, bins=256, range=(0, 256), color='#888888', alpha=0.9)
            ax.set_title('Y Channel', fontsize=9)
        else:
            for ch_idx, (color, label) in enumerate(
                zip(('#4488ff', '#44bb44', '#ff4444'), ('B', 'G', 'R'))
            ):
                ch = bgr_array[:, :, ch_idx].ravel()
                ax.hist(ch, bins=256, range=(0, 256), color=color, alpha=0.55, label=label)
            ax.legend(loc='upper right', fontsize=7)
            ax.set_title('RGB Channels', fontsize=9)

        ax.set_xlim(0, 255)
        ax.set_ylabel('Count', fontsize=7)
        ax.tick_params(labelsize=7)
        self._canvas.draw()


# ── 감마 보정 창 ──────────────────────────────────────────────────────────────

class GammaWindow(QWidget):
    """실시간 감마 보정 슬라이더 창"""

    gamma_changed = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gamma Correction")
        self.setFixedSize(300, 130)
        self.setWindowFlags(self.windowFlags() | Qt.Tool)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self._label = QLabel("Gamma: 1.0")
        self._label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._label)

        # 내부 범위: 1~30 (실제값 0.1~3.0 × 10)
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(1, 30)
        self._slider.setValue(10)
        self._slider.setTickInterval(5)
        self._slider.setTickPosition(QSlider.TicksBelow)
        self._slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self._slider)

        reset_btn = QPushButton("Reset (1.0)")
        reset_btn.clicked.connect(lambda: self._slider.setValue(10))
        layout.addWidget(reset_btn)

    def _on_slider_changed(self, value: int):
        gamma = value / 10.0
        self._label.setText(f"Gamma: {gamma:.1f}")
        self.gamma_changed.emit(gamma)

    def get_gamma(self) -> float:
        return self._slider.value() / 10.0


# ── 이미지 레이블 (줌 + 픽셀 오버레이) ───────────────────────────────────────

class ImageLabel(QLabel):
    """줌·픽셀값 오버레이를 지원하는 이미지 표시 위젯"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self._pixmap_orig: QPixmap | None = None
        self._bgr_array: np.ndarray | None = None
        self._fmt: str = ""

    def set_pixmap(self, pixmap: QPixmap):
        self._pixmap_orig = pixmap

    def set_image_data(self, bgr_array: np.ndarray, fmt: str):
        self._bgr_array = bgr_array
        self._fmt = fmt

    def render_zoom(self, zoom: float):
        if self._pixmap_orig is None:
            return
        w = int(self._pixmap_orig.width() * zoom)
        h = int(self._pixmap_orig.height() * zoom)
        scaled = self._pixmap_orig.scaled(
            w, h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        if zoom >= PIXEL_OVERLAY_ZOOM_THRESHOLD and self._bgr_array is not None:
            scaled = self._draw_pixel_overlay(scaled, zoom)

        self.setPixmap(scaled)
        self.resize(scaled.size())

    def _draw_pixel_overlay(self, pixmap: QPixmap, zoom: float) -> QPixmap:
        bgr = self._bgr_array
        img_h, img_w = bgr.shape[:2]

        # 성능 보호: 표시 크기가 임계치 초과이면 스킵
        if img_w * zoom > PIXEL_OVERLAY_MAX_DIM or img_h * zoom > PIXEL_OVERLAY_MAX_DIM:
            return pixmap

        is_yuv = self._fmt.upper() in ('YUV420P', 'NV12', 'NV21')
        font_size = max(6, min(int(zoom * 0.35), 14))

        result = QPixmap(pixmap)
        painter = QPainter(result)
        font = QFont("Monospace")
        font.setPixelSize(font_size)
        painter.setFont(font)

        for py in range(img_h):
            for px in range(img_w):
                dx = int(px * zoom)
                dy = int(py * zoom)
                bw = int((px + 1) * zoom) - dx
                bh = int((py + 1) * zoom) - dy

                b_val = int(bgr[py, px, 0])
                g_val = int(bgr[py, px, 1])
                r_val = int(bgr[py, px, 2])

                if is_yuv:
                    lum = int(0.299 * r_val + 0.587 * g_val + 0.114 * b_val)
                    text = str(lum)
                    brightness = lum
                else:
                    text = f"{r_val},{g_val},{b_val}"
                    brightness = int(0.299 * r_val + 0.587 * g_val + 0.114 * b_val)

                painter.setPen(
                    QColor(0, 0, 0) if brightness > 128 else QColor(255, 255, 255)
                )
                painter.drawText(QRect(dx, dy, bw, bh), Qt.AlignCenter, text)

        painter.end()
        return result

    def has_image(self) -> bool:
        return self._pixmap_orig is not None


# ── 메인 윈도우 ───────────────────────────────────────────────────────────────

class RawViewerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAW Image Viewer")
        self.resize(1100, 700)
        self.setAcceptDrops(True)

        self._file_path = ""
        self._zoom = 1.0
        self._raw_bgr_array = None   # 파일에서 읽은 원본 (감마 미적용)
        self._bgr_array = None       # 감마 적용 후 (표시·히스토그램용)
        self._gamma = 1.0

        self._hist_win = HistogramWindow()
        self._gamma_win = GammaWindow()
        self._gamma_win.gamma_changed.connect(self._on_gamma_changed)

        self._build_menu()
        self._build_ui()
        self._build_statusbar()
        self._connect_auto_render()

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

        settings_menu = menu_bar.addMenu("Settings")

        gamma_action = QAction("Gamma Correction…", self)
        gamma_action.triggered.connect(self._show_gamma_window)
        settings_menu.addAction(gamma_action)

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(8)

        side = self._build_side_panel()
        main_layout.addWidget(side)

        self._scroll = QScrollArea()
        self._scroll.setAlignment(Qt.AlignCenter)
        self._scroll.setWidgetResizable(False)
        self._scroll.setFrameShape(QFrame.StyledPanel)

        self._img_label = ImageLabel()
        self._img_label.setText(
            "이미지를 불러오세요\n(File > Open Raw File 또는 파일 드래그앤드롭)"
        )
        self._img_label.setStyleSheet("color: #888; font-size: 14px;")
        self._img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._scroll.setWidget(self._img_label)

        self._scroll.viewport().installEventFilter(self)

        main_layout.addWidget(self._scroll, stretch=1)

    def _build_side_panel(self) -> QWidget:
        panel = QFrame()
        panel.setFrameShape(QFrame.StyledPanel)
        panel.setFixedWidth(200)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 10, 8, 10)
        layout.setSpacing(8)

        layout.addWidget(QLabel("Width (px)"))
        self._width_spin = QSpinBox()
        self._width_spin.setRange(1, 65535)
        self._width_spin.setValue(1920)
        layout.addWidget(self._width_spin)

        layout.addWidget(QLabel("Height (px)"))
        self._height_spin = QSpinBox()
        self._height_spin.setRange(1, 65535)
        self._height_spin.setValue(1080)
        layout.addWidget(self._height_spin)

        layout.addWidget(QLabel("Format"))
        self._fmt_combo = QComboBox()
        self._fmt_combo.addItems(FORMATS)
        layout.addWidget(self._fmt_combo)

        layout.addSpacing(10)
        self._zoom_label = QLabel("Zoom: 100%")
        self._zoom_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._zoom_label)

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

        layout.addStretch()

        hint = QLabel(
            "숫자키 1~8: 고정 줌\n"
            "휠: 줌 인/아웃\n"
            "Ctrl+O: 파일 열기\n"
            "Settings > Gamma"
        )
        hint.setStyleSheet("color: #999; font-size: 10px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        return panel

    # ── 자동 렌더링 연결 ──────────────────────────────────────────────────────

    def _connect_auto_render(self):
        self._width_spin.valueChanged.connect(self._on_param_changed)
        self._height_spin.valueChanged.connect(self._on_param_changed)
        self._fmt_combo.currentIndexChanged.connect(self._on_param_changed)

    def _on_param_changed(self):
        if self._file_path:
            self._load_and_render()

    # ── 상태바 ────────────────────────────────────────────────────────────────

    def _build_statusbar(self):
        self._statusbar: QStatusBar = self.statusBar()
        self._status_file = QLabel("파일: —")
        self._status_fmt  = QLabel("포맷: —")
        self._status_res  = QLabel("해상도: —")
        self._status_zoom = QLabel("줌: 100%")

        for lbl in (self._status_file, self._status_fmt,
                    self._status_res, self._status_zoom):
            self._statusbar.addPermanentWidget(lbl)

    def _update_status(self):
        fname = os.path.basename(self._file_path) if self._file_path else "—"
        self._status_file.setText(f"파일: {fname}")
        self._status_fmt.setText(f"포맷: {self._fmt_combo.currentText()}")
        w = self._width_spin.value()
        h = self._height_spin.value()
        self._status_res.setText(f"해상도: {w}×{h}")
        pct = int(self._zoom * 100)
        self._status_zoom.setText(f"줌: {pct}%")
        self._zoom_label.setText(f"Zoom: {pct}%")

    # ── 파일 확장자 → 포맷 자동 설정 ─────────────────────────────────────────

    def _set_format_from_ext(self, path: str):
        fmt = detect_format_from_ext(path)
        if fmt is None:
            return
        idx = self._fmt_combo.findText(fmt)
        if idx >= 0:
            self._fmt_combo.blockSignals(True)
            self._fmt_combo.setCurrentIndex(idx)
            self._fmt_combo.blockSignals(False)

    # ── 파일명 → 해상도 파싱 ─────────────────────────────────────────────────

    def _parse_resolution_from_filename(self, path: str):
        name = os.path.basename(path)
        m = re.search(r'(\d{2,5})[x_×](\d{2,5})', name, re.IGNORECASE)
        if not m:
            return
        w, h = int(m.group(1)), int(m.group(2))
        if 1 <= w <= 65535 and 1 <= h <= 65535:
            self._width_spin.blockSignals(True)
            self._height_spin.blockSignals(True)
            self._width_spin.setValue(w)
            self._height_spin.setValue(h)
            self._width_spin.blockSignals(False)
            self._height_spin.blockSignals(False)

    # ── 파일 열기 ─────────────────────────────────────────────────────────────

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "RAW 파일 열기",
            "",
            "RAW Files (*.raw *.bin *.yuv *.rgb *.data *.nv12 *.nv21);;All Files (*)",
        )
        if not path:
            return
        self._file_path = path
        self._set_format_from_ext(path)
        self._parse_resolution_from_filename(path)
        self._load_and_render()

    # ── 드래그앤드롭 ──────────────────────────────────────────────────────────

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self._file_path = path
            self._set_format_from_ext(path)
            self._parse_resolution_from_filename(path)
            self._load_and_render()

    # ── 로딩 + 렌더링 ─────────────────────────────────────────────────────────

    def _load_and_render(self):
        if not self._file_path:
            return

        w   = self._width_spin.value()
        h   = self._height_spin.value()
        fmt = self._fmt_combo.currentText()

        try:
            bgr = load_raw(self._file_path, w, h, fmt)
        except Exception as e:
            QMessageBox.critical(self, "로딩 오류", str(e))
            return

        self._raw_bgr_array = bgr
        self._bgr_array = apply_gamma(bgr, self._gamma)

        self._render_bgr(self._bgr_array, fmt)

    def _render_bgr(self, bgr: np.ndarray, fmt: str):
        """BGR 배열을 QPixmap으로 변환해 표시하고 히스토그램을 갱신"""
        self._img_label.set_image_data(bgr, fmt)

        rgb = bgr[:, :, ::-1].copy()
        qimg = QImage(
            rgb.data,
            rgb.shape[1], rgb.shape[0],
            rgb.strides[0],
            QImage.Format_RGB888,
        )
        pixmap = QPixmap.fromImage(qimg)
        self._img_label.set_pixmap(pixmap)
        self._img_label.setStyleSheet("")

        self._apply_zoom(self._zoom)
        self._update_status()
        self._hist_win.update_histogram(bgr, fmt)

    # ── 감마 보정 ─────────────────────────────────────────────────────────────

    def _on_gamma_changed(self, gamma: float):
        self._gamma = gamma
        if self._raw_bgr_array is None:
            return
        fmt = self._fmt_combo.currentText()
        self._bgr_array = apply_gamma(self._raw_bgr_array, gamma)
        self._render_bgr(self._bgr_array, fmt)

    def _show_gamma_window(self):
        geo = self.frameGeometry()
        self._gamma_win.move(geo.right() + 10, geo.top())
        self._gamma_win.show()
        self._gamma_win.raise_()

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
        h_bar = self._scroll.horizontalScrollBar()
        v_bar = self._scroll.verticalScrollBar()
        h_bar.setValue((h_bar.minimum() + h_bar.maximum()) // 2)
        v_bar.setValue((v_bar.minimum() + v_bar.maximum()) // 2)

    # ── 이벤트 ───────────────────────────────────────────────────────────────

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(100, self._position_and_show_histogram)

    def _position_and_show_histogram(self):
        geo = self.frameGeometry()
        self._hist_win.move(geo.right() + 10, geo.top())
        self._hist_win.show()

    def closeEvent(self, event):
        self._hist_win.close()
        self._gamma_win.close()
        super().closeEvent(event)

    def eventFilter(self, source, event):
        from PyQt5.QtCore import QEvent
        if source is self._scroll.viewport() and event.type() == QEvent.Wheel:
            delta = event.angleDelta().y()
            if delta > 0:
                self._change_zoom(+ZOOM_STEP)
            else:
                self._change_zoom(-ZOOM_STEP)
            return True
        return super().eventFilter(source, event)

    def keyPressEvent(self, event):
        key = event.key()
        if key in ZOOM_PRESETS:
            self._apply_zoom(ZOOM_PRESETS[key])
        else:
            super().keyPressEvent(event)
