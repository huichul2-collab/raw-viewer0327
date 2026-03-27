"""
RAW 이미지 파일 로딩 및 포맷 변환 모듈
지원 포맷: YUV420p, NV12, NV21, RGB24, RGB10, RGB12
"""

import os

import numpy as np
import cv2


def load_raw(file_path: str, width: int, height: int, fmt: str) -> np.ndarray:
    """
    RAW 파일을 읽어 BGR888 numpy 배열로 반환.

    Parameters
    ----------
    file_path : str
    width     : int  - 이미지 너비 (픽셀)
    height    : int  - 이미지 높이 (픽셀)
    fmt       : str  - 'YUV420p' | 'NV12' | 'NV21' | 'RGB24' | 'RGB10' | 'RGB12'

    Returns
    -------
    np.ndarray  shape=(height, width, 3), dtype=uint8, BGR 채널 순
    """
    fmt = fmt.upper()

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
    else:
        raise ValueError(f"지원하지 않는 포맷: {fmt}")


def _expected_bytes(width: int, height: int, fmt: str) -> int:
    """포맷별 예상 바이트 수 계산"""
    pixels = width * height
    if fmt in ("YUV420P", "NV12", "NV21"):
        return pixels * 3 // 2
    elif fmt == "RGB24":
        return pixels * 3
    elif fmt in ("RGB10", "RGB12"):
        # 16bit 컨테이너 3채널
        return pixels * 6
    raise ValueError(f"Unknown fmt: {fmt}")


# ── YUV420 planar ────────────────────────────────────────────────────────────

def _yuv420p_to_bgr(data: bytes, width: int, height: int) -> np.ndarray:
    """YUV420 planar (YYYY…UU…VV…) → BGR"""
    yuv = np.frombuffer(data, dtype=np.uint8).reshape((height * 3 // 2, width))
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
    return bgr


# ── NV12 / NV21 ─────────────────────────────────────────────────────────────

def _nv12_to_bgr(data: bytes, width: int, height: int) -> np.ndarray:
    """NV12 (YYYY… UV UV …) → BGR"""
    yuv = np.frombuffer(data, dtype=np.uint8).reshape((height * 3 // 2, width))
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
    return bgr


def _nv21_to_bgr(data: bytes, width: int, height: int) -> np.ndarray:
    """NV21 (YYYY… VU VU …) → BGR"""
    yuv = np.frombuffer(data, dtype=np.uint8).reshape((height * 3 // 2, width))
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
    return bgr


# ── RGB24 ───────────────────────────────────────────────────────────────────

def _rgb24_to_bgr(data: bytes, width: int, height: int) -> np.ndarray:
    """RGB888 packed → BGR"""
    arr = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr


# ── RGB10 / RGB12 ────────────────────────────────────────────────────────────

def _rgb10_to_bgr(data: bytes, width: int, height: int) -> np.ndarray:
    """
    RGB10 (16bit 컨테이너, 상위 10bit 유효) → BGR8
    각 채널 uint16로 읽은 뒤 >> 2 하여 8bit 변환
    """
    arr = np.frombuffer(data, dtype=np.uint16).reshape((height, width, 3))
    arr8 = (arr >> 2).astype(np.uint8)
    bgr = cv2.cvtColor(arr8, cv2.COLOR_RGB2BGR)
    return bgr


def _rgb12_to_bgr(data: bytes, width: int, height: int) -> np.ndarray:
    """
    RGB12 (16bit 컨테이너, 상위 12bit 유효) → BGR8
    각 채널 uint16로 읽은 뒤 >> 4 하여 8bit 변환
    """
    arr = np.frombuffer(data, dtype=np.uint16).reshape((height, width, 3))
    arr8 = (arr >> 4).astype(np.uint8)
    bgr = cv2.cvtColor(arr8, cv2.COLOR_RGB2BGR)
    return bgr


# ── 감마 보정 ────────────────────────────────────────────────────────────────

def apply_gamma(bgr: np.ndarray, gamma: float) -> np.ndarray:
    """
    BGR 배열에 감마 보정을 적용한 새 배열 반환.
    공식: output = (input / 255) ^ (1 / gamma) * 255

    Parameters
    ----------
    bgr   : np.ndarray  shape=(H, W, 3), dtype=uint8
    gamma : float       감마 값 (0.1 ~ 3.0). 1.0이면 원본 반환.
    """
    if abs(gamma - 1.0) < 1e-9:
        return bgr
    inv_gamma = 1.0 / gamma
    lut = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(bgr, lut)


# ── 파일 확장자 → 포맷 추정 ──────────────────────────────────────────────────

def detect_format_from_ext(filename: str) -> str | None:
    """
    파일 확장자로부터 포맷 문자열을 추정해 반환.
    인식하지 못하면 None 반환.
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext in ('.yuv', '.nv12', '.nv21'):
        return 'YUV420p'
    elif ext in ('.rgb', '.raw'):
        return 'RGB24'
    return None
