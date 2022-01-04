import unittest

from tracker.object_recognition import ObjectRecognition
from tracker.region_detection import Point, Region


class TestScreen(unittest.TestCase):
    def test_get_player_regions(self):
        tests = [
            {
                "width": 960,
                "height": 491,
                "regions": [
                    Region(start=Point(96, 24), end=Point(384, 220)),
                    Region(start=Point(384, 24), end=Point(576, 220)),
                    Region(start=Point(576, 24), end=Point(816, 220)),
                    Region(start=Point(96, 220), end=Point(384, 392)),
                    Region(start=Point(384, 220), end=Point(576, 392)),
                    Region(start=Point(576, 220), end=Point(816, 392)),
                ],
            }
        ]

        for t in tests:
            with self.subTest("split regions"):
                regions = ObjectRecognition.get_player_regions(t["width"], t["height"])
                self.assertListEqual(regions, t["regions"])


if __name__ == "__main__":
    unittest.main()
