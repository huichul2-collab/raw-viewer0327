"""
RawViewer Android — Kivy 앱 진입점
"""

from kivy.app import App
from viewer_kivy import RawViewerLayout


class RawViewerApp(App):
    def build(self):
        self.title = "RawViewer"
        return RawViewerLayout()


if __name__ == "__main__":
    RawViewerApp().run()
