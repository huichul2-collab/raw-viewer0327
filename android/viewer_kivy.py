"""
RawViewer Android — Kivy UI
Windows 버전(viewer_app.py)과 동일 기능, Android 터치 UI 패턴으로 구현.
"""

import os
import threading

import numpy as np
import cv2

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.scatter import Scatter
from kivy.uix.image import Image as KivyImage
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.spinner import Spinner
from kivy.uix.textinput import TextInput
from kivy.uix.checkbox import CheckBox
from kivy.uix.slider import Slider
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.scrollview import ScrollView
from kivy.uix.togglebutton import ToggleButton
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.metrics import dp

from image_loader import (
    load_raw, load_standard_image, load_mipi_image,
    apply_gamma, detect_format_from_ext, parse_resolution_from_filename,
    is_standard_image, FORMATS, MIPI_FORMATS, BAYER_PATTERNS,
)

# 히스토그램은 matplotlib 사용 (없으면 비활성화)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io
    _MATPLOTLIB_OK = True
except ImportError:
    _MATPLOTLIB_OK = False


# ── 상수 ─────────────────────────────────────────────────────────────────────

ZOOM_STEP = 0.10
ZOOM_MIN  = 0.10
ZOOM_MAX  = 8.00

DRAWER_WIDTH = dp(220)

_STYLE = {
    "top_bar_height": dp(48),
    "bottom_bar_height": dp(56),
    "btn_color": (0.2, 0.2, 0.2, 1),
    "accent": (0.13, 0.59, 0.95, 1),
    "bg_dark": (0.12, 0.12, 0.12, 1),
    "bg_drawer": (0.16, 0.16, 0.16, 1),
    "text": (0.95, 0.95, 0.95, 1),
    "hint": (0.5, 0.5, 0.5, 1),
}


# ── 유틸: numpy BGR → Kivy Texture ───────────────────────────────────────────

def numpy_to_texture(arr_bgr: np.ndarray) -> Texture:
    """BGR numpy 배열 → Kivy Texture (RGB, 상하 반전)"""
    arr_rgb = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB)
    arr_rgb = np.ascontiguousarray(np.flipud(arr_rgb))
    h, w, _ = arr_rgb.shape
    texture = Texture.create(size=(w, h), colorfmt="rgb")
    texture.blit_buffer(arr_rgb.tobytes(), colorfmt="rgb", bufferfmt="ubyte")
    return texture


def make_histogram_texture(arr_bgr: np.ndarray, size=(256, 128)) -> Texture | None:
    """BGR 이미지의 RGB 히스토그램을 Kivy Texture로 반환"""
    if not _MATPLOTLIB_OK:
        return None
    try:
        fig, ax = plt.subplots(figsize=(size[0] / 72, size[1] / 72), dpi=72)
        fig.patch.set_facecolor("#1e1e1e")
        ax.set_facecolor("#1e1e1e")
        colors = ("blue", "green", "red")
        for i, c in enumerate(colors):
            hist = cv2.calcHist([arr_bgr], [i], None, [256], [0, 256])
            ax.plot(hist.flatten(), color=c, linewidth=0.8, alpha=0.8)
        ax.set_xlim(0, 255)
        ax.tick_params(colors="gray", labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        plt.tight_layout(pad=0.2)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        png_bytes = buf.read()

        nparr = np.frombuffer(png_bytes, dtype=np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        return numpy_to_texture(img)
    except Exception:
        return None


# ── 사이드 드로어 ─────────────────────────────────────────────────────────────

class DrawerPanel(ScrollView):
    """접이식 설정 패널 (왼쪽)"""

    def __init__(self, on_load_cb, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, 1)
        self.width = DRAWER_WIDTH
        self.do_scroll_x = False

        self._on_load = on_load_cb

        content = BoxLayout(
            orientation="vertical",
            size_hint_y=None,
            spacing=dp(6),
            padding=[dp(10), dp(10), dp(10), dp(10)],
        )
        content.bind(minimum_height=content.setter("height"))

        def add_label(text):
            lbl = Label(
                text=text,
                size_hint_y=None,
                height=dp(22),
                color=_STYLE["hint"],
                halign="left",
                text_size=(DRAWER_WIDTH - dp(20), None),
            )
            content.add_widget(lbl)

        # ── 파일 열기 버튼 ────────────────────────────────────────────────────
        self._open_btn = Button(
            text="[파일 열기]",
            size_hint_y=None,
            height=dp(44),
            background_color=_STYLE["accent"],
        )
        self._open_btn.bind(on_release=self._open_file_chooser)
        content.add_widget(self._open_btn)

        self._filename_lbl = Label(
            text="파일 없음",
            size_hint_y=None,
            height=dp(30),
            color=_STYLE["hint"],
            halign="left",
            text_size=(DRAWER_WIDTH - dp(20), None),
            font_size=dp(10),
        )
        content.add_widget(self._filename_lbl)

        # ── Width / Height ────────────────────────────────────────────────────
        add_label("Width (px)")
        self._width_input = TextInput(
            text="1920",
            multiline=False,
            size_hint_y=None,
            height=dp(38),
            input_filter="int",
        )
        content.add_widget(self._width_input)

        add_label("Height (px)")
        self._height_input = TextInput(
            text="1080",
            multiline=False,
            size_hint_y=None,
            height=dp(38),
            input_filter="int",
        )
        content.add_widget(self._height_input)

        # ── Format 스피너 ─────────────────────────────────────────────────────
        add_label("Format")
        self._fmt_spinner = Spinner(
            text="MIPI_RAW10",
            values=FORMATS,
            size_hint_y=None,
            height=dp(40),
        )
        self._fmt_spinner.bind(text=self._on_format_changed)
        content.add_widget(self._fmt_spinner)

        # ── MIPI Override 구분선 ──────────────────────────────────────────────
        self._mipi_sep = Label(
            text="── MIPI Override ──",
            size_hint_y=None,
            height=dp(24),
            color=_STYLE["hint"],
            font_size=dp(10),
        )
        content.add_widget(self._mipi_sep)

        # MIPI 체크박스
        mipi_row = BoxLayout(size_hint_y=None, height=dp(36), spacing=dp(6))
        self._mipi_cb = CheckBox(size_hint_x=None, width=dp(36))
        mipi_row.add_widget(self._mipi_cb)
        mipi_row.add_widget(Label(text="MIPI 수동 설정", color=_STYLE["text"]))
        content.add_widget(mipi_row)

        # Bit depth
        add_label("Bit Depth")
        self._bit_spinner = Spinner(
            text="10",
            values=["8", "10", "12", "14"],
            size_hint_y=None,
            height=dp(40),
        )
        content.add_widget(self._bit_spinner)

        # Bayer Pattern
        add_label("Bayer Pattern")
        self._bayer_spinner = Spinner(
            text="RGGB",
            values=BAYER_PATTERNS,
            size_hint_y=None,
            height=dp(40),
        )
        content.add_widget(self._bayer_spinner)

        # Raw Bayer / Demosaic 토글
        mode_row = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(4))
        self._raw_btn = ToggleButton(
            text="Raw Bayer",
            group="demosaic_mode",
            size_hint_x=0.5,
            state="normal",
        )
        self._demosaic_btn = ToggleButton(
            text="Demosaic",
            group="demosaic_mode",
            size_hint_x=0.5,
            state="down",
        )
        mode_row.add_widget(self._raw_btn)
        mode_row.add_widget(self._demosaic_btn)
        content.add_widget(mode_row)

        # ── Load / Render 버튼 ────────────────────────────────────────────────
        self._load_btn = Button(
            text="Load / Render",
            size_hint_y=None,
            height=dp(48),
            background_color=(0.2, 0.7, 0.3, 1),
        )
        self._load_btn.bind(on_release=lambda *_: self._on_load())
        content.add_widget(self._load_btn)

        self.add_widget(content)
        self._content = content

        # 초기 MIPI 관련 위젯 가시성
        self._on_format_changed(self._fmt_spinner, self._fmt_spinner.text)

    # ── 프로퍼티 ──────────────────────────────────────────────────────────────

    @property
    def file_path(self) -> str:
        return getattr(self, "_file_path", "")

    @property
    def width_val(self) -> int:
        try:
            return max(1, int(self._width_input.text))
        except ValueError:
            return 1920

    @property
    def height_val(self) -> int:
        try:
            return max(1, int(self._height_input.text))
        except ValueError:
            return 1080

    @property
    def fmt_val(self) -> str:
        return self._fmt_spinner.text

    @property
    def bayer_val(self) -> str:
        return self._bayer_spinner.text

    @property
    def demosaic_mode(self) -> bool:
        return self._demosaic_btn.state == "down"

    @property
    def mipi_bits(self) -> int:
        return int(self._bit_spinner.text)

    @property
    def mipi_override(self) -> bool:
        return self._mipi_cb.active

    # ── 내부 ──────────────────────────────────────────────────────────────────

    def _on_format_changed(self, spinner, text):
        is_mipi = text.upper() in MIPI_FORMATS
        # MIPI 관련 위젯은 항상 표시, Override 체크박스는 MIPI 아닐 때만 의미 없음
        pass  # 항상 표시 상태 유지 (간소화)

    def _open_file_chooser(self, *args):
        content = BoxLayout(orientation="vertical", spacing=dp(6))

        start_path = "/sdcard" if os.path.exists("/sdcard") else os.path.expanduser("~")
        fc = FileChooserListView(
            path=start_path,
            filters=["*.raw", "*.raw8", "*.raw10", "*.raw12", "*.raw14",
                     "*.mipi", "*.bin", "*.yuv", "*.rgb", "*.data",
                     "*.png", "*.jpg", "*.jpeg", "*.bmp"],
            size_hint_y=1,
        )
        content.add_widget(fc)

        btn_row = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(6))
        cancel_btn = Button(text="취소")
        select_btn = Button(text="선택", background_color=_STYLE["accent"])
        btn_row.add_widget(cancel_btn)
        btn_row.add_widget(select_btn)
        content.add_widget(btn_row)

        popup = Popup(
            title="파일 선택",
            content=content,
            size_hint=(0.95, 0.9),
        )

        def on_select(*_):
            if fc.selection:
                path = fc.selection[0]
                self._file_path = path
                self._filename_lbl.text = os.path.basename(path)

                # 확장자로 포맷 자동 설정
                auto_fmt = detect_format_from_ext(path)
                if auto_fmt and auto_fmt in FORMATS:
                    self._fmt_spinner.text = auto_fmt

                # 파일명에서 해상도 파싱
                res = parse_resolution_from_filename(path)
                if res:
                    self._width_input.text = str(res[0])
                    self._height_input.text = str(res[1])

                popup.dismiss()
                self._on_load()

        select_btn.bind(on_release=on_select)
        cancel_btn.bind(on_release=popup.dismiss)
        popup.open()

    def set_filename_label(self, text: str):
        self._filename_lbl.text = text


# ── 이미지 뷰어 영역 (ScatterLayout 기반) ─────────────────────────────────────

class ImageViewer(FloatLayout):
    """핀치줌 + 드래그를 지원하는 이미지 표시 영역"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._scatter = Scatter(
            do_rotation=False,
            do_translation=True,
            do_scale=True,
            scale_min=ZOOM_MIN,
            scale_max=ZOOM_MAX,
        )
        self._img = KivyImage(allow_stretch=True, keep_ratio=True)
        self._scatter.add_widget(self._img)
        self.add_widget(self._scatter)

        self._placeholder = Label(
            text="파일을 열어 이미지를 표시합니다\n(드로어 > 파일 열기)",
            color=_STYLE["hint"],
            halign="center",
            pos_hint={"center_x": 0.5, "center_y": 0.5},
        )
        self.add_widget(self._placeholder)

    def set_texture(self, texture: Texture):
        self._img.texture = texture
        w, h = texture.size
        self._img.size = (w, h)
        self._scatter.size = (w, h)
        # 초기 스케일: 화면에 맞게
        if self.width > 0 and self.height > 0:
            scale = min(self.width / w, self.height / h, 1.0)
            self._scatter.scale = scale
            self._scatter.pos = (
                (self.width - w * scale) / 2,
                (self.height - h * scale) / 2,
            )
        # 플레이스홀더 숨김
        if self._placeholder.parent:
            self.remove_widget(self._placeholder)

    def get_zoom(self) -> float:
        return self._scatter.scale

    def set_zoom(self, zoom: float):
        zoom = max(ZOOM_MIN, min(ZOOM_MAX, zoom))
        cx = self.width / 2
        cy = self.height / 2
        self._scatter.apply_transform(
            self._scatter.get_inverse_previous_touch_pos(None),
            anchor=(cx, cy),
        )
        self._scatter.scale = zoom

    def zoom_step(self, delta: float):
        new_scale = max(ZOOM_MIN, min(ZOOM_MAX, self._scatter.scale + delta))
        self._scatter.scale = new_scale


# ── 히스토그램 팝업 ───────────────────────────────────────────────────────────

class HistogramPopup(Popup):
    def __init__(self, bgr_arr: np.ndarray, **kwargs):
        super().__init__(
            title="Histogram",
            size_hint=(0.8, 0.5),
            **kwargs,
        )
        self._img = KivyImage(allow_stretch=True, keep_ratio=False)
        self.content = self._img

        def _compute(*_):
            tex = make_histogram_texture(bgr_arr, size=(512, 256))
            if tex:
                self._img.texture = tex
            else:
                self.title = "Histogram (matplotlib 없음)"

        Clock.schedule_once(_compute, 0.1)


# ── 메인 레이아웃 ─────────────────────────────────────────────────────────────

class RawViewerLayout(BoxLayout):
    """앱 루트 레이아웃"""

    def __init__(self, **kwargs):
        super().__init__(orientation="vertical", **kwargs)

        self._bgr_array: np.ndarray | None = None
        self._drawer_open = True
        self._loading = False

        # ── Top Bar ───────────────────────────────────────────────────────────
        top_bar = BoxLayout(
            orientation="horizontal",
            size_hint_y=None,
            height=_STYLE["top_bar_height"],
            spacing=dp(4),
            padding=[dp(4), dp(4), dp(4), dp(4)],
        )

        menu_btn = Button(
            text="≡",
            size_hint_x=None,
            width=dp(44),
            font_size=dp(22),
            background_color=_STYLE["btn_color"],
        )
        menu_btn.bind(on_release=self._toggle_drawer)
        top_bar.add_widget(menu_btn)

        self._title_lbl = Label(
            text="RawViewer",
            font_size=dp(16),
            color=_STYLE["text"],
            halign="left",
            size_hint_x=1,
        )
        top_bar.add_widget(self._title_lbl)

        hist_btn = Button(
            text="Hist",
            size_hint_x=None,
            width=dp(60),
            background_color=_STYLE["btn_color"],
        )
        hist_btn.bind(on_release=self._show_histogram)
        top_bar.add_widget(hist_btn)

        self.add_widget(top_bar)

        # ── 중간 영역 (드로어 + 이미지) ──────────────────────────────────────
        mid = BoxLayout(orientation="horizontal", size_hint_y=1)

        self._drawer = DrawerPanel(on_load_cb=self._load_and_render)
        mid.add_widget(self._drawer)

        self._viewer = ImageViewer()
        mid.add_widget(self._viewer)

        self.add_widget(mid)

        # ── Bottom Bar ────────────────────────────────────────────────────────
        bottom_bar = BoxLayout(
            orientation="horizontal",
            size_hint_y=None,
            height=_STYLE["bottom_bar_height"],
            spacing=dp(6),
            padding=[dp(8), dp(6), dp(8), dp(6)],
        )

        zoom_out_btn = Button(
            text="−",
            size_hint_x=None,
            width=dp(44),
            font_size=dp(20),
            background_color=_STYLE["btn_color"],
        )
        zoom_out_btn.bind(on_release=lambda *_: self._zoom_step(-ZOOM_STEP))
        bottom_bar.add_widget(zoom_out_btn)

        self._zoom_lbl = Label(
            text="100%",
            size_hint_x=None,
            width=dp(60),
            color=_STYLE["text"],
        )
        bottom_bar.add_widget(self._zoom_lbl)

        zoom_in_btn = Button(
            text="+",
            size_hint_x=None,
            width=dp(44),
            font_size=dp(20),
            background_color=_STYLE["btn_color"],
        )
        zoom_in_btn.bind(on_release=lambda *_: self._zoom_step(+ZOOM_STEP))
        bottom_bar.add_widget(zoom_in_btn)

        # 감마 슬라이더
        bottom_bar.add_widget(Label(
            text="γ",
            size_hint_x=None,
            width=dp(20),
            color=_STYLE["hint"],
        ))
        self._gamma_slider = Slider(
            min=0.2,
            max=3.0,
            value=1.0,
            step=0.05,
            size_hint_x=1,
        )
        self._gamma_slider.bind(value=self._on_gamma_changed)
        bottom_bar.add_widget(self._gamma_slider)

        self._gamma_lbl = Label(
            text="1.00",
            size_hint_x=None,
            width=dp(40),
            color=_STYLE["text"],
        )
        bottom_bar.add_widget(self._gamma_lbl)

        self.add_widget(bottom_bar)

        # 상태
        self._gamma_apply_event = None

    # ── 드로어 토글 ───────────────────────────────────────────────────────────

    def _toggle_drawer(self, *args):
        if self._drawer_open:
            self._drawer.width = 0
            self._drawer.opacity = 0
            self._drawer_open = False
        else:
            self._drawer.width = DRAWER_WIDTH
            self._drawer.opacity = 1
            self._drawer_open = True

    # ── 이미지 로딩 ───────────────────────────────────────────────────────────

    def _load_and_render(self):
        if self._loading:
            return
        path = self._drawer.file_path
        if not path:
            self._show_error("파일을 먼저 선택하세요.")
            return

        self._loading = True
        self._title_lbl.text = "로딩 중…"

        def _worker():
            try:
                if is_standard_image(path):
                    bgr = load_standard_image(path)
                elif self._drawer.mipi_override and self._drawer.fmt_val.upper() in MIPI_FORMATS:
                    bgr = load_mipi_image(
                        path,
                        self._drawer.width_val,
                        self._drawer.height_val,
                        self._drawer.fmt_val,
                        self._drawer.bayer_val,
                        demosaic=self._drawer.demosaic_mode,
                    )
                else:
                    bgr = load_raw(
                        path,
                        self._drawer.width_val,
                        self._drawer.height_val,
                        self._drawer.fmt_val,
                        self._drawer.bayer_val,
                    )
                Clock.schedule_once(lambda dt: self._on_loaded(bgr), 0)
            except Exception as e:
                Clock.schedule_once(lambda dt: self._on_load_error(str(e)), 0)

        threading.Thread(target=_worker, daemon=True).start()

    def _on_loaded(self, bgr: np.ndarray):
        self._bgr_array = bgr
        gamma = self._gamma_slider.value
        display = apply_gamma(bgr, gamma) if abs(gamma - 1.0) > 0.01 else bgr
        texture = numpy_to_texture(display)
        self._viewer.set_texture(texture)

        fname = os.path.basename(self._drawer.file_path)
        w, h = bgr.shape[1], bgr.shape[0]
        self._title_lbl.text = f"{fname}  {w}×{h}  {self._drawer.fmt_val}"
        self._loading = False

    def _on_load_error(self, msg: str):
        self._title_lbl.text = "오류"
        self._loading = False
        self._show_error(msg)

    def _show_error(self, msg: str):
        popup = Popup(
            title="오류",
            content=Label(text=msg, text_size=(dp(300), None), halign="left"),
            size_hint=(0.85, 0.4),
        )
        popup.open()

    # ── 줌 ───────────────────────────────────────────────────────────────────

    def _zoom_step(self, delta: float):
        self._viewer.zoom_step(delta)
        pct = int(self._viewer.get_zoom() * 100)
        self._zoom_lbl.text = f"{pct}%"

    # ── 감마 ─────────────────────────────────────────────────────────────────

    def _on_gamma_changed(self, slider, value):
        self._gamma_lbl.text = f"{value:.2f}"
        # 슬라이더 이동이 끝난 후 0.3초 뒤에 재렌더링 (디바운스)
        if self._gamma_apply_event:
            self._gamma_apply_event.cancel()
        self._gamma_apply_event = Clock.schedule_once(self._apply_gamma_render, 0.3)

    def _apply_gamma_render(self, *args):
        if self._bgr_array is None:
            return
        gamma = self._gamma_slider.value
        display = apply_gamma(self._bgr_array, gamma)
        texture = numpy_to_texture(display)
        self._viewer.set_texture(texture)

    # ── 히스토그램 ────────────────────────────────────────────────────────────

    def _show_histogram(self, *args):
        if self._bgr_array is None:
            self._show_error("먼저 이미지를 불러오세요.")
            return
        HistogramPopup(self._bgr_array).open()
