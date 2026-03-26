# RAW Image Viewer

Windows에서 RAW 이미지 파일(.raw, .bin, .yuv, .rgb 등)을 직접 열어 확인하는 GUI 뷰어입니다.

---

## 요구 사항

- Python 3.10 이상 (Windows)
- pip 패키지: PyQt5, numpy, opencv-python

---

## 설치

```bash
pip install PyQt5 numpy opencv-python
```

시스템 패키지 충돌이 발생하는 경우:

```bash
pip install PyQt5 numpy opencv-python --break-system-packages
```

---

## 실행

```bash
python main.py
```

---

## 사용 방법

### 1. 파일 열기
- 메뉴바 **File > Open Raw File…** 또는 `Ctrl+O`로 파일 선택
- 지원 확장자: `.raw`, `.bin`, `.yuv`, `.rgb`, `.data`, 기타 모든 파일

### 2. 파라미터 설정 (왼쪽 사이드 패널)
| 항목 | 설명 |
|------|------|
| Width | 이미지 너비(픽셀) |
| Height | 이미지 높이(픽셀) |
| Format | 아래 지원 포맷 참조 |

### 3. Load / Render 버튼 클릭
- 설정한 파라미터로 RAW 파일을 해석하여 화면에 표시

### 4. 줌 조작
| 조작 | 동작 |
|------|------|
| 마우스 휠 위 | 줌 인 (+10%) |
| 마우스 휠 아래 | 줌 아웃 (−10%) |
| 숫자키 `1` | 50% |
| 숫자키 `2` | 100% (원본) |
| 숫자키 `3` | 150% |
| 숫자키 `4` | 200% |
| 숫자키 `5` | 250% |
| 숫자키 `6` | 300% |
| `+` 버튼 | 줌 인 |
| `-` 버튼 | 줌 아웃 |

---

## 지원 포맷

| 포맷 명 | 설명 | 파일 크기 (W×H 기준) |
|---------|------|----------------------|
| **YUV420p** | YUV420 Planar — Y 전체, U 1/4, V 1/4 순서 | W×H × 1.5 bytes |
| **NV12** | YUV420 Semi-Planar — Y 전체, UV 인터리브 | W×H × 1.5 bytes |
| **NV21** | YUV420 Semi-Planar — Y 전체, VU 인터리브 | W×H × 1.5 bytes |
| **RGB24** | RGB 8bpp per channel, packed | W×H × 3 bytes |
| **RGB10** | RGB 10bit, 16bit 컨테이너 (상위 10bit 유효) | W×H × 6 bytes |
| **RGB12** | RGB 12bit, 16bit 컨테이너 (상위 12bit 유효) | W×H × 6 bytes |

> **RGB10 / RGB12 변환 방식**
> 각 채널을 uint16로 읽은 뒤 우측 비트 시프트(>>2 또는 >>4)하여 8bit로 변환 후 표시합니다.
> 하위 비트 유효(little-endian packed) 포맷의 경우 별도 파싱 로직이 필요합니다.

---

## 파일 구조

```
20260327_raw_viewer/
├── main.py          # 진입점
├── viewer_app.py    # PyQt5 메인 윈도우
├── image_loader.py  # RAW 파일 로딩 + 포맷 변환
├── requirements.txt # 의존 패키지
└── README.md        # 이 파일
```

---

## 문제 해결

| 증상 | 원인 / 해결 |
|------|-------------|
| "파일 크기 부족" 오류 | Width/Height 값이 실제 파일과 다름 → 올바른 해상도 입력 |
| 색상이 이상함 | Format 선택이 틀림 → 다른 포맷 시도 |
| 화면이 깨짐 | RGB10/12에서 비트 배치가 다른 경우 — image_loader.py의 시프트 값 조정 |
| PyQt5 import 오류 | `pip install PyQt5` 재실행 |
