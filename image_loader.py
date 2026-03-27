"""
RAW 이미지 파일 로딩 및 포맷 변환 모듈
지원 포맷: YUV420p, NV12, NV21, RGB24, RGB10, RGB12,
          MIPI_RAW8, MIPI_RAW10, MIPI_RAW12, MIPI_RAW14
"""

import numpy as np
import cv2


def load_raw(file_path: str, width: int, height: int, fmt: str,
             bayer_pattern: str = "RGGB") -> np.ndarray:
    """
    RAW 파일을 읽어 RGB888 numpy 배열로 반환.

    Parameters
    ----------
    file_path     : str
    width         : int  - 이미지 너비 (픽셀)
    height        : int  - 이미지 높이 (픽셀)
    fmt           : str  - 'YUV420p' | 'NV12' | 'NV21' | 'RGB24' | 'RGB10' | 'RGB12'
                          | 'MIPI_RAW8' | 'MIPI_RAW10' | 'MIPI_RAW12' | 'MIPI_RAW14'
    bayer_pattern : str  - MIPI 포맷 시 사용: 'RGGB' | 'GRBG' | 'GBRG' | 'BGGR'

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
    elif fmt == "MIPI_RAW8":
        return pixels
    elif fmt == "MIPI_RAW10":
        # 4픽셀 → 5바이트
        return pixels * 5 // 4
    elif fmt == "MIPI_RAW12":
        # 2픽셀 → 3바이트
        return pixels * 3 // 2
    elif fmt == "MIPI_RAW14":
        # 4픽셀 → 7바이트
        return pixels * 7 // 4
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


# ── MIPI RAW 언패킹 ──────────────────────────────────────────────────────────

def unpack_mipi_raw8(data: np.ndarray, width: int, height: int) -> np.ndarray:
    """MIPI RAW8: 1픽셀 = 1바이트, 패킹 없음"""
    return data.reshape(height, width).astype(np.uint16)


def unpack_mipi_raw10(data: np.ndarray, width: int, height: int) -> np.ndarray:
    """MIPI RAW10: 4픽셀 → 5바이트 패킹 언패킹 → uint16 (0~1023)"""
    data = data.reshape(-1, 5)
    out = np.zeros((data.shape[0], 4), dtype=np.uint16)
    out[:, 0] = (data[:, 0].astype(np.uint16) << 2) | (data[:, 4] & 0x03)
    out[:, 1] = (data[:, 1].astype(np.uint16) << 2) | ((data[:, 4] >> 2) & 0x03)
    out[:, 2] = (data[:, 2].astype(np.uint16) << 2) | ((data[:, 4] >> 4) & 0x03)
    out[:, 3] = (data[:, 3].astype(np.uint16) << 2) | ((data[:, 4] >> 6) & 0x03)
    return out.reshape(height, width)


def unpack_mipi_raw12(data: np.ndarray, width: int, height: int) -> np.ndarray:
    """MIPI RAW12: 2픽셀 → 3바이트 패킹 언패킹 → uint16 (0~4095)"""
    data = data.reshape(-1, 3)
    out = np.zeros((data.shape[0], 2), dtype=np.uint16)
    out[:, 0] = (data[:, 0].astype(np.uint16) << 4) | (data[:, 2] & 0x0F)
    out[:, 1] = (data[:, 1].astype(np.uint16) << 4) | ((data[:, 2] >> 4) & 0x0F)
    return out.reshape(height, width)


def unpack_mipi_raw14(data: np.ndarray, width: int, height: int) -> np.ndarray:
    """MIPI RAW14: 4픽셀 → 7바이트 패킹 언패킹 → uint16 (0~16383)"""
    data = data.reshape(-1, 7)
    out = np.zeros((data.shape[0], 4), dtype=np.uint16)
    out[:, 0] = (data[:, 0].astype(np.uint16) << 6) | (data[:, 4] & 0x3F)
    out[:, 1] = (data[:, 1].astype(np.uint16) << 6) | (data[:, 5] & 0x3F)
    out[:, 2] = (data[:, 2].astype(np.uint16) << 6) | ((data[:, 4] >> 6) | ((data[:, 5] & 0x0F) << 2))
    out[:, 3] = (data[:, 3].astype(np.uint16) << 6) | ((data[:, 5] >> 4) | ((data[:, 6] & 0x03) << 4))
    return out.reshape(height, width)


def bayer_to_rgb(bayer_uint16: np.ndarray, bayer_pattern: str = "RGGB",
                 bits: int = 10) -> np.ndarray:
    """
    언패킹된 Bayer uint16 데이터를 Demosaicing하여 BGR8 반환.

    Parameters
    ----------
    bayer_uint16  : np.ndarray  shape=(height, width), dtype=uint16
    bayer_pattern : str         'RGGB' | 'GRBG' | 'GBRG' | 'BGGR'
    bits          : int         유효 비트 수 (8/10/12/14)

    Returns
    -------
    np.ndarray  shape=(height, width, 3), dtype=uint8, BGR 채널 순
    """
    shift = bits - 8
    img8 = (bayer_uint16 >> shift).astype(np.uint8)

    pattern_map = {
        "RGGB": cv2.COLOR_BAYER_RG2BGR,
        "GRBG": cv2.COLOR_BAYER_GR2BGR,
        "GBRG": cv2.COLOR_BAYER_GB2BGR,
        "BGGR": cv2.COLOR_BAYER_BG2BGR,
    }
    code = pattern_map.get(bayer_pattern.upper(), cv2.COLOR_BAYER_RG2BGR)
    return cv2.cvtColor(img8, code)
