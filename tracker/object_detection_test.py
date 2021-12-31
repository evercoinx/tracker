import unittest

from tracker.object_detection import ObjectDetection
from tracker.utils import Dimensions


class TestScreen(unittest.TestCase):
    def test_split_into_regions(self):
        tests = [
            {
                "dimensions": Dimensions(960, 480),
                "width_parts": 3,
                "height_parts": 2,
                "regions": [
                    (0, 0, 320, 240),
                    (320, 0, 640, 240),
                    (640, 0, 960, 240),
                    (0, 240, 320, 480),
                    (320, 240, 640, 480),
                    (640, 240, 960, 480),
                ],
            }
        ]

        for t in tests:
            with self.subTest("split regions"):
                regions = ObjectDetection.split_into_regions(
                    t["dimensions"],
                    t["width_parts"],
                    t["height_parts"],
                )
                self.assertListEqual(regions, t["regions"])


if __name__ == "__main__":
    unittest.main()
