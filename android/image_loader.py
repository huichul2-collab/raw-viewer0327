"""
RAW 이미지 파일 로딩 및 포맷 변환 모듈 (Android/Kivy용)
PyQt5 의존성 없음 — numpy + cv2만 사용

지원 포맷: YUV420p, NV12, NV21, RGB24, RGB10, RGB12,
          MIPI_RAW8, MIPI_RAW10, MIPI_RAW12, MIPI_RAW14
"""

import os
import numpy as np
import cv2


# ── 상수 ─────────────────────────────────────────────────────────────────────

BAYER_CODES = {
    "RGGB": cv2.COLOR_BAYER_RG2BGR,
    "GRBG": cv2.COLOR_BAYER_GR2BGR,
    "GBRG": cv2.COLOR_BAYER_GB2BGR,
    "BGGR": cv2.COLOR_BAYER_BG2BGR,
}

STANDARD_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

EXT_FORMAT_MAP = {
    ".raw8":  "MIPI_RAW8",
    ".raw10": "MIPI_RAW10",
    ".raw12": "MIPI_RAW12",
    ".raw14": "MIPI_RAW14",
    ".mipi":  "MIPI_RAW10",
}

FORMATS = [
    "YUV420p", "NV12", "NV21",
    "RGB24", "RGB10", "RGB12",
    "MIPI_RAW8", "MIPI_RAW10", "MIPI_RAW12", "MIPI_RAW14",
]

MIPI_FORMATS = {"MIPI_RAW8", "MIPI_RAW10", "MIPI_RAW12", "MIPI_RAW14"}
BAYER_PATTERNS = ["RGGB", "GRBG", "GBRG", "BGGR"]


# ── 유틸리티 ─────────────────────────────────────────────────────────────────

def is_standard_image(file_path: str) -> bool:
    """표준 이미지 파일(PNG/JPG 등) 여부 확인"""
    ext = os.path.splitext(file_path)[1].lower()
    return ext in STANDARD_EXTS


def detect_format_from_ext(file_path: str) -> str | None:
    """파일 확장자로부터 포맷 자동 감지"""
    ext = os.path.splitext(file_path)[1].lower()
    return EXT_FORMAT_MAP.get(ext)


def parse_resolution_from_filename(filename: str):
    """
    파일명에서 WxH 또는 W_H 형태의 해상도 파싱.
    예: frame_1920x1080.raw10 → (1920, 1080)
    실패 시 None 반환.
    """
    import re
    basename = os.path.basename(filename)
    m = re.search(r'(\d{3,5})[x_](\d{3,5})', basename, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def apply_gamma(bgr: np.ndarray, gamma: float) -> np.ndarray:
    """BGR 이미지에 감마 보정 적용 (gamma=1.0이면 원본 유지)"""
    if abs(gamma - 1.0) < 0.001:
        return bgr
    inv = 1.0 / gamma
    lut = np.array([((i / 255.0) ** inv) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(bgr, lut)


# ── 표준 이미지 로딩 ─────────────────────────────────────────────────────────

def load_standard_image(file_path: str) -> np.ndarray:
    """PNG/JPG 등 표준 포맷 → BGR numpy 배열"""
    bgr = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {file_path}")
    return bgr


# ── RAW 메인 로딩 ─────────────────────────────────────────────────────────────

def load_raw(file_path: str, width: int, height: int, fmt: str,
             bayer_pattern: str = "RGGB") -> np.ndarray:
    """
    RAW 파일을 읽어 BGR888 numpy 배열로 반환.

    Parameters
    ----------
    file_path     : str
    width         : int
    height        : int
    fmt           : str  - 'YUV420p' | 'NV12' | 'NV21' | 'RGB24' | 'RGB10' | 'RGB12'
                          | 'MIPI_RAW8' | 'MIPI_RAW10' | 'MIPI_RAW12' | 'MIPI_RAW14'
    bayer_pattern : str  - 'RGGB' | 'GRBG' | 'GBRG' | 'BGGR'

    Returns
    -------
    np.ndarray  shape=(height, width, 3), dtype=uint8, BGR 채널 순
    """
    fmt = fmt.upper().replace("YUV420P", "YUV420P")
    if fmt == "YUV420P":
        fmt = "YUV420P"

    expected = _expected_bytes(width, height, fmt)
    with open(file_path, "rb") as f:
        raw_bytes = f.read(expected)

    if len(raw_bytes) < expected:
        raise ValueError(
            f"파일 크기 부족: 필요 {expected} bytes, 실제 {len(raw_bytes)} bytes"
        )

    if fmt == "YUV420P":
        return _yuv420p_to_bgr(raw_bytes, width, height)
    elif fmt == "NV12":
        return _nv12_to_bgr(raw_bytes, width, height)
    elif fmt == "NV21":
        return _nv21_to_bgr(raw_bytes, width, height)
    elif fmt == "RGB24":
        return _rgb24_to_bgr(raw_bytes, width, height)
    elif fmt == "RGB10":
        return _rgb10_to_bgr(raw_bytes, width, height)
    elif fmt == "RGB12":
        return _rgb12_to_bgr(raw_bytes, width, height)
    elif fmt == "MIPI_RAW8":
        bayer = unpack_mipi_raw8(np.frombuffer(raw_bytes, dtype=np.uint8), width, height)
        return bayer_to_rgb(bayer, bayer_pattern, bits=8)
    elif fmt == "MIPI_RAW10":
        bayer = unpack_mipi_raw10(np.frombuffer(raw_bytes, dtype=np.uint8), width, height)
        return bayer_to_rgb(bayer, bayer_pattern, bits=10)
    elif fmt == "MIPI_RAW12":
        bayer = unpack_mipi_raw12(np.frombuffer(raw_bytes, dtype=np.uint8), width, height)
        return bayer_to_rgb(bayer, bayer_pattern, bits=12)
    elif fmt == "MIPI_RAW14":
        bayer = unpack_mipi_raw14(np.frombuffer(raw_bytes, dtype=np.uint8), width, height)
        return bayer_to_rgb(bayer, bayer_pattern, bits=14)
    else:
        raise ValueError(f"지원하지 않는 포맷: {fmt}")


def load_mipi_image(file_path: str, width: int, height: int, fmt: str,
                    bayer_pattern: str = "RGGB",
                    demosaic: bool = True) -> np.ndarray:
    """
    MIPI RAW 파일 로딩. demosaic=False 이면 Raw Bayer(그레이스케일)로 반환.
    """
    fmt_up = fmt.upper()
    bits_map = {"MIPI_RAW8": 8, "MIPI_RAW10": 10, "MIPI_RAW12": 12, "MIPI_RAW14": 14}
    bits = bits_map.get(fmt_up, 10)

    expected = _expected_bytes(width, height, fmt_up)
    with open(file_path, "rb") as f:
        raw_bytes = f.read(expected)

    if len(raw_bytes) < expected:
        raise ValueError(
            f"파일 크기 부족: 필요 {expected} bytes, 실제 {len(raw_bytes)} bytes"
        )

    data = np.frombuffer(raw_bytes, dtype=np.uint8)

    unpack_fn = {
        "MIPI_RAW8":  unpack_mipi_raw8,
        "MIPI_RAW10": unpack_mipi_raw10,
        "MIPI_RAW12": unpack_mipi_raw12,
        "MIPI_RAW14": unpack_mipi_raw14,
    }[fmt_up]

    bayer = unpack_fn(data, width, height)

    if demosaic:
        return bayer_to_rgb(bayer, bayer_pattern, bits=bits)
    else:
        # Raw Bayer → 8bit 그레이스케일 3채널
        shift = bits - 8
        gray = (bayer >> shift).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ── 내부 헬퍼 ────────────────────────────────────────────────────────────────

def _expected_bytes(width: int, height: int, fmt: str) -> int:
    pixels = width * height
    if fmt in ("YUV420P", "NV12", "NV21"):
        return pixels * 3 // 2
    elif fmt == "RGB24":
        return pixels * 3
    elif fmt in ("RGB10", "RGB12"):
        return pixels * 6
    elif fmt == "MIPI_RAW8":
        return pixels
    elif fmt == "MIPI_RAW10":
        return pixels * 5 // 4
    elif fmt == "MIPI_RAW12":
        return pixels * 3 // 2
    elif fmt == "MIPI_RAW14":
        return pixels * 7 // 4
    raise ValueError(f"Unknown fmt: {fmt}")


def _yuv420p_to_bgr(data: bytes, width: int, height: int) -> np.ndarray:
    yuv = np.frombuffer(data, dtype=np.uint8).reshape((height * 3 // 2, width))
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)


def _nv12_to_bgr(data: bytes, width: int, height: int) -> np.ndarray:
    yuv = np.frombuffer(data, dtype=np.uint8).reshape((height * 3 // 2, width))
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)


def _nv21_to_bgr(data: bytes, width: int, height: int) -> np.ndarray:
    yuv = np.frombuffer(data, dtype=np.uint8).reshape((height * 3 // 2, width))
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)


def _rgb24_to_bgr(data: bytes, width: int, height: int) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _rgb10_to_bgr(data: bytes, width: int, height: int) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint16).reshape((height, width, 3))
    arr8 = (arr >> 2).astype(np.uint8)
    return cv2.cvtColor(arr8, cv2.COLOR_RGB2BGR)


def _rgb12_to_bgr(data: bytes, width: int, height: int) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint16).reshape((height, width, 3))
    arr8 = (arr >> 4).astype(np.uint8)
    return cv2.cvtColor(arr8, cv2.COLOR_RGB2BGR)


# ── MIPI RAW 언패킹 ───────────────────────────────────────────────────────────

def unpack_mipi_raw8(data: np.ndarray, width: int, height: int) -> np.ndarray:
    """MIPI RAW8: 1픽셀 = 1바이트"""
    return data.reshape(height, width).astype(np.uint16)


def unpack_mipi_raw10(data: np.ndarray, width: int, height: int) -> np.ndarray:
    """MIPI RAW10: 4픽셀 → 5바이트"""
    data = data.reshape(-1, 5)
    out = np.zeros((data.shape[0], 4), dtype=np.uint16)
    out[:, 0] = (data[:, 0].astype(np.uint16) << 2) | (data[:, 4] & 0x03)
    out[:, 1] = (data[:, 1].astype(np.uint16) << 2) | ((data[:, 4] >> 2) & 0x03)
    out[:, 2] = (data[:, 2].astype(np.uint16) << 2) | ((data[:, 4] >> 4) & 0x03)
    out[:, 3] = (data[:, 3].astype(np.uint16) << 2) | ((data[:, 4] >> 6) & 0x03)
    return out.reshape(height, width)


def unpack_mipi_raw12(data: np.ndarray, width: int, height: int) -> np.ndarray:
    """MIPI RAW12: 2픽셀 → 3바이트"""
    data = data.reshape(-1, 3)
    out = np.zeros((data.shape[0], 2), dtype=np.uint16)
    out[:, 0] = (data[:, 0].astype(np.uint16) << 4) | (data[:, 2] & 0x0F)
    out[:, 1] = (data[:, 1].astype(np.uint16) << 4) | ((data[:, 2] >> 4) & 0x0F)
    return out.reshape(height, width)


def unpack_mipi_raw14(data: np.ndarray, width: int, height: int) -> np.ndarray:
    """MIPI RAW14: 4픽셀 → 7바이트"""
    data = data.reshape(-1, 7)
    out = np.zeros((data.shape[0], 4), dtype=np.uint16)
    out[:, 0] = (data[:, 0].astype(np.uint16) << 6) | (data[:, 4] & 0x3F)
    out[:, 1] = (data[:, 1].astype(np.uint16) << 6) | (data[:, 5] & 0x3F)
    out[:, 2] = (data[:, 2].astype(np.uint16) << 6) | ((data[:, 4] >> 6) | ((data[:, 5] & 0x0F) << 2))
    out[:, 3] = (data[:, 3].astype(np.uint16) << 6) | ((data[:, 5] >> 4) | ((data[:, 6] & 0x03) << 4))
    return out.reshape(height, width)


def bayer_to_rgb(bayer_uint16: np.ndarray, bayer_pattern: str = "RGGB",
                 bits: int = 10) -> np.ndarray:
    """언패킹된 Bayer uint16 → Demosaicing → BGR8"""
    shift = bits - 8
    img8 = (bayer_uint16 >> shift).astype(np.uint8)
    code = BAYER_CODES.get(bayer_pattern.upper(), cv2.COLOR_BAYER_RG2BGR)
    return cv2.cvtColor(img8, code)
