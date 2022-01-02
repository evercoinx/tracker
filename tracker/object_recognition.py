from typing import List, Tuple

from tracker.utils import Point, Region


class ObjectRecognition:
    """Recognizes objects on an window frame"""

    def get_dealer_position(
        self, point: Point, width: int, height: int, ratio: Tuple[int]
    ) -> int:
        regions = self.split_into_regions(width, height, ratio)
        for i, region in enumerate(regions):
            if (
                region.start.x < point.x < region.end.x
                and region.start.y < point.y < region.end.y
            ):
                return i
        return -1

    @staticmethod
    def split_into_regions(width: int, height: int, ratio: Tuple[int]) -> List[Region]:
        w = width // ratio[0]
        h = height // ratio[1]

        iterations = ratio[0] * ratio[1]
        regions: List[Region] = []

        x, y = 0, 0
        for _ in range(iterations // 2):
            r = Region(
                start=Point(x, y),
                end=Point(x + w, y + h),
            )
            regions.append(r)
            x += w

        x, y = 0, h
        for _ in range(iterations // 2):
            r = Region(
                start=Point(x, y),
                end=Point(x + w, y + h),
            )
            regions.append(r)
            x += w
        return regions
