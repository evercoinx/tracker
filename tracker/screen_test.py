import unittest

from tracker.screen import Screen


class TestScreen(unittest.TestCase):
    def test_calculate_window_coords(self):
        tests = [
            {
                "name": "full hd",
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
            with self.subTest(f"calculate {t['name']} window coordinates"):
                wc = Screen.calculate_window_coords(
                    [0, 1, 2, 3], t["width"], t["height"], 0, 0
                )
                self.assertTupleEqual(tuple(wc), t["coords"])


if __name__ == "__main__":
    unittest.main()
