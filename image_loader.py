"""
RAW 이미지 파일 로딩 및 포맷 변환 모듈
지원 포맷: YUV420p, NV12, NV21, RGB24, RGB10, RGB12, MIPI_RAW8/10/12/14
"""

import numpy as np
import cv2


BAYER_CODES = {
    'RGGB': cv2.COLOR_BayerRG2BGR,
    'GRBG': cv2.COLOR_BayerGR2BGR,
    'GBRG': cv2.COLOR_BayerGB2BGR,
    'BGGR': cv2.COLOR_BayerBG2BGR,
}


def load_mipi_image(file_path: str, width: int, height: int,
                    mipi_fmt: str, bayer_pattern: str = 'RGGB',
                    demosaic: bool = False) -> np.ndarray:
    """
    MIPI RAW 파일을 읽어 BGR numpy 배열로 반환.

    Parameters
    ----------
    file_path     : str
    width, height : int
    mipi_fmt      : 'MIPI_RAW8' | 'MIPI_RAW10' | 'MIPI_RAW12' | 'MIPI_RAW14'
    bayer_pattern : 'RGGB' | 'GRBG' | 'GBRG' | 'BGGR'
    demosaic      : False → Bayer 그레이스케일, True → Bayer→RGB 디모자익

    Returns
    -------
    np.ndarray  shape=(height, width, 3), dtype=uint8, BGR
    """
    data = np.fromfile(file_path, dtype=np.uint8)
    bayer = _unpack_mipi(data, width, height, mipi_fmt)
    bits = {'MIPI_RAW8': 8, 'MIPI_RAW10': 10, 'MIPI_RAW12': 12, 'MIPI_RAW14': 14}[mipi_fmt]
    img8 = (bayer >> (bits - 8)).astype(np.uint8)
    if demosaic:
        return cv2.cvtColor(img8, BAYER_CODES[bayer_pattern])
    else:
        return cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)


def _unpack_mipi(data: np.ndarray, width: int, height: int, mipi_fmt: str) -> np.ndarray:
    if mipi_fmt == 'MIPI_RAW8':
        return _unpack_mipi_raw8(data, width, height)
    elif mipi_fmt == 'MIPI_RAW10':
        return _unpack_mipi_raw10(data, width, height)
    elif mipi_fmt == 'MIPI_RAW12':
        return _unpack_mipi_raw12(data, width, height)
    elif mipi_fmt == 'MIPI_RAW14':
        return _unpack_mipi_raw14(data, width, height)
    raise ValueError(f"Unknown MIPI format: {mipi_fmt}")


def _unpack_mipi_raw8(data: np.ndarray, width: int, height: int) -> np.ndarray:
    """1 byte per pixel → uint16"""
    return data[:width * height].astype(np.uint16).reshape(height, width)


def _unpack_mipi_raw10(data: np.ndarray, width: int, height: int) -> np.ndarray:
    """4 pixels per 5 bytes (MIPI CSI-2 RAW10 packing)"""
    n_groups = width * height // 4
    b = data[:n_groups * 5].reshape(-1, 5).astype(np.uint16)
    p0 = (b[:, 0] << 2) | (b[:, 4] & 0x03)
    p1 = (b[:, 1] << 2) | ((b[:, 4] >> 2) & 0x03)
    p2 = (b[:, 2] << 2) | ((b[:, 4] >> 4) & 0x03)
    p3 = (b[:, 3] << 2) | ((b[:, 4] >> 6) & 0x03)
    result = np.empty(n_groups * 4, dtype=np.uint16)
    result[0::4] = p0; result[1::4] = p1
    result[2::4] = p2; result[3::4] = p3
    return result.reshape(height, width)


def _unpack_mipi_raw12(data: np.ndarray, width: int, height: int) -> np.ndarray:
    """2 pixels per 3 bytes (MIPI CSI-2 RAW12 packing)"""
    n_groups = width * height // 2
    b = data[:n_groups * 3].reshape(-1, 3).astype(np.uint16)
    p0 = (b[:, 0] << 4) | (b[:, 2] & 0x0F)
    p1 = (b[:, 1] << 4) | ((b[:, 2] >> 4) & 0x0F)
    result = np.empty(n_groups * 2, dtype=np.uint16)
    result[0::2] = p0; result[1::2] = p1
    return result.reshape(height, width)


def _unpack_mipi_raw14(data: np.ndarray, width: int, height: int) -> np.ndarray:
    """4 pixels per 7 bytes (MIPI CSI-2 RAW14 packing)"""
    n_groups = width * height // 4
    b = data[:n_groups * 7].reshape(-1, 7).astype(np.uint16)
    p0 = (b[:, 0] << 6) | (b[:, 4] & 0x3F)
    p1 = (b[:, 1] << 6) | ((b[:, 4] & 0xC0) >> 2) | (b[:, 5] & 0x0F)
    p2 = (b[:, 2] << 6) | ((b[:, 5] & 0xF0) >> 2) | (b[:, 6] & 0x03)
    p3 = (b[:, 3] << 6) | (b[:, 6] >> 2)
    result = np.empty(n_groups * 4, dtype=np.uint16)
    result[0::4] = p0; result[1::4] = p1
    result[2::4] = p2; result[3::4] = p3
    return result.reshape(height, width)


def load_raw(file_path: str, width: int, height: int, fmt: str) -> np.ndarray:
    """
    RAW 파일을 읽어 RGB888 numpy 배열로 반환.

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
    # 상위 10bit → 8bit: >> 2
    arr8 = (arr >> 2).astype(np.uint8)
    bgr = cv2.cvtColor(arr8, cv2.COLOR_RGB2BGR)
    return bgr


def _rgb12_to_bgr(data: bytes, width: int, height: int) -> np.ndarray:
    """
    RGB12 (16bit 컨테이너, 상위 12bit 유효) → BGR8
    각 채널 uint16로 읽은 뒤 >> 4 하여 8bit 변환
    """
    arr = np.frombuffer(data, dtype=np.uint16).reshape((height, width, 3))
    # 상위 12bit → 8bit: >> 4
    arr8 = (arr >> 4).astype(np.uint8)
    bgr = cv2.cvtColor(arr8, cv2.COLOR_RGB2BGR)
    return bgr
