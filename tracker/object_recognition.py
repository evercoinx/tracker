from typing import Dict, List, Tuple

from tracker.region_detection import Point, Region


class ObjectRecognition:
    """Recognizes objects on an window frame"""

    PLAYER_REGION_PERCENTS = [
        # top left
        (0.125, 0.05),
        (0.40, 0.45),
        # top middle
        (0.40, 0.05),
        (0.60, 0.45),
        # top right
        (0.60, 0.05),
        (0.85, 0.45),
        # bottom left
        (0.125, 0.45),
        (0.40, 0.80),
        # bottom middle
        (0.40, 0.45),
        (0.60, 0.80),
        # bottom right
        (0.60, 0.45),
        (0.85, 0.80),
    ]

    def __init__(self) -> None:
        self.player_regions: Dict[Tuple, List[Region]] = {}

    def get_dealer_position(self, target: Region, width: int, height: int) -> int:
        regions = self.get_player_regions(width, height)
        for i, r in enumerate(regions):
            if self.point_in_region(
                point=target.start, region=r
            ) and self.point_in_region(point=target.end, region=r):
                return i
        return -1

    def get_player_regions(self, width: int, height: int) -> List[Region]:
        cache_key = (width, height)
        if cache_key in self.player_regions:
            return self.player_regions[cache_key]

        regions: List[Region] = []
        points: List[Point] = []
        for i, (x, y) in enumerate(ObjectRecognition.PLAYER_REGION_PERCENTS):
            p = self.get_scaled_point(x, y, width, height)
            points.append(p)

            if i % 2 != 0:
                r = Region(start=points[0], end=points[1])
                regions.append(r)
                points = []

        self.player_regions[cache_key] = regions
        return regions

    @staticmethod
    def point_in_region(point: Point, region: Region) -> bool:
        return (region.start.x < point.x < region.end.x) and (
            region.start.y < point.y < region.end.y
        )

    @staticmethod
    def get_scaled_point(
        percent_x: float, percent_y: float, width: int, height: int
    ) -> Point:
        x = int(percent_x * width)
        y = int(percent_y * height)
        return Point(x, y)
