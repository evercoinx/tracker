from typing import List

from tracker.region_detection import Point, Region


class ObjectRecognition:
    """Recognizes objects on an window frame"""

    def get_dealer_position(self, point: Point, width: int, height: int) -> int:
        regions = self.get_player_regions(width, height)
        for i, r in enumerate(regions):
            if (r.start.x < point.x < r.end.x) and (r.start.y < point.y < r.end.y):
                return i
        return -1

    @staticmethod
    def get_player_regions(width: int, height: int) -> List[Region]:
        percents = [
            (0.10, 0.05),
            (0.40, 0.45),
            (0.40, 0.05),
            (0.60, 0.45),
            (0.60, 0.05),
            (0.85, 0.45),
            (0.10, 0.45),
            (0.40, 0.80),
            (0.40, 0.45),
            (0.60, 0.80),
            (0.60, 0.45),
            (0.85, 0.80),
        ]

        regions: List[Region] = []
        points: List[Point] = []
        for i, (x, y) in enumerate(percents):
            p = ObjectRecognition.get_scaled_point(x, y, width, height)
            points.append(p)

            if i % 2 != 0:
                r = Region(start=points[0], end=points[1])
                regions.append(r)
                points = []
        return regions

    @staticmethod
    def get_scaled_point(
        percent_x: float, percent_y: float, width: int, height: int
    ) -> Point:
        x = int(percent_x * width)
        y = int(percent_y * height)
        return Point(x, y)
