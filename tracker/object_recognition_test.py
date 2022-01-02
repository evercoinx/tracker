import unittest

from tracker.object_recognition import ObjectRecognition
from tracker.utils import Dimensions, Point, Region


class TestScreen(unittest.TestCase):
    def test_split_into_regions(self):
        tests = [
            {
                "dimensions": Dimensions(960, 480),
                "ratio": (3, 2),
                "regions": [
                    Region(start=Point(0, 0), end=Point(320, 240)),
                    Region(start=Point(320, 0), end=Point(640, 240)),
                    Region(start=Point(640, 0), end=Point(960, 240)),
                    Region(start=Point(0, 240), end=Point(320, 480)),
                    Region(start=Point(320, 240), end=Point(640, 480)),
                    Region(start=Point(640, 240), end=Point(960, 480)),
                ],
            }
        ]

        for t in tests:
            with self.subTest("split regions"):
                regions = ObjectRecognition.split_into_regions(
                    t["dimensions"], t["ratio"]
                )
                self.assertListEqual(regions, t["regions"])


if __name__ == "__main__":
    unittest.main()
