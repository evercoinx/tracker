import unittest

from tracker.screen import Screen


class TestScreen(unittest.TestCase):
    def test_get_window_screens(self):
        tests = [
            {
                "name": "full hd",
                "windows": [0, 1, 2, 3],
                "left_margin": 0,
                "top_margin": 0,
                "width": 1920,
                "height": 1080,
                "coords": (
                    {"top": 0, "left": 0, "width": 960, "height": 540},
                    {"top": 0, "left": 960, "width": 960, "height": 540},
                    {"top": 540, "left": 0, "width": 960, "height": 540},
                    {"top": 540, "left": 960, "width": 960, "height": 540},
                ),
            },
            {
                "name": "hd",
                "windows": [0, 1, 2, 3],
                "left_margin": 0,
                "top_margin": 0,
                "width": 1280,
                "height": 720,
                "coords": (
                    {"top": 0, "left": 0, "width": 640, "height": 360},
                    {"top": 0, "left": 640, "width": 640, "height": 360},
                    {"top": 360, "left": 0, "width": 640, "height": 360},
                    {"top": 360, "left": 640, "width": 640, "height": 360},
                ),
            },
        ]

        for t in tests:
            with self.subTest(f"get {t['name']} window screens"):
                wc = Screen.get_window_screens(
                    t["windows"],
                    t["left_margin"],
                    t["top_margin"],
                    t["width"],
                    t["height"],
                )
                self.assertTupleEqual(tuple(wc), t["coords"])


if __name__ == "__main__":
    unittest.main()
