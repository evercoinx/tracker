import unittest

from tracker.grabber import Grabber


class TestGrabber(unittest.TestCase):
    def test_get_rois(self):
        tests = [
            {
                "name": "full hd",
                "width": 1920,
                "height": 1080,
                "rois": (
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
                "rois": (
                    {"top": 0, "left": 0, "width": 640, "height": 360},
                    {"top": 0, "left": 640, "width": 640, "height": 360},
                    {"top": 360, "left": 0, "width": 640, "height": 360},
                    {"top": 360, "left": 640, "width": 640, "height": 360},
                ),
            },
        ]

        for t in tests:
            with self.subTest(f"get rois for {t['name']}"):
                rois = Grabber.get_rois(t["width"], t["height"], 0, 0)
                self.assertTupleEqual(t["rois"], rois)


if __name__ == "__main__":
    unittest.main()
