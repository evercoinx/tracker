from typing import Dict, List, Tuple

from tracker.object_detection import Point, Region


class ObjectRecognition:
    """Recognizes objects on an window frame"""

    PLAYER_REGION_PERCENTAGE = [
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

    def recognize_dealer_position(self, region: Region, width: int, height: int) -> int:
        player_regions = self.get_player_regions(width, height)
        for i, r in enumerate(player_regions):
            if self.point_in_region(region.start, r) and self.point_in_region(
                region.end, r
            ):
                return i
        return -1

    def get_player_regions(self, width: int, height: int) -> List[Region]:
        cache_key = (width, height)
        if cache_key in self.player_regions:
            return self.player_regions[cache_key]

        regions: List[Region] = []
        points: List[Point] = []
        for i, (x, y) in enumerate(ObjectRecognition.PLAYER_REGION_PERCENTAGE):
            p = self.get_point_by_percentage(x, y, width, height)
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
    def get_point_by_percentage(
        width_percentage: float,
        height_percentage: float,
        total_width: int,
        total_height: int,
    ) -> Point:
        x = int(width_percentage * total_width)
        y = int(height_percentage * total_height)
        return Point(x, y)
