"""
RAW Image Viewer — 진입점
실행: python main.py
"""

import sys
from PyQt5.QtWidgets import QApplication
from viewer_app import RawViewerWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("RAW Image Viewer")
    window = RawViewerWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
