@echo off
echo === RAW Viewer Build Script ===
echo.

echo [1/2] Installing dependencies...
pip install PyQt5 numpy opencv-python pyinstaller
echo.

echo [2/2] Building EXE...
pyinstaller --onefile --windowed --name RawViewer main.py
echo.

echo === Build complete: dist\RawViewer.exe ===
pause
