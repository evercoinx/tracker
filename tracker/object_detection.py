from typing import List, Tuple

import cv2
import numpy as np

from tracker.error import TemplateError
from tracker.utils import Dimensions, Point, Region


class ObjectDetection:
    """Detect objects on an window frame"""

    def __init__(self, template_path: str, template_format: str) -> None:
        self.template_path = template_path
        self.template_format = template_format

        self.dealer_tmpl = cv2.imread(
            f"{self.template_path}/dealer.{self.template_format}", cv2.IMREAD_UNCHANGED
        )
        if self.dealer_tmpl is None:
            raise TemplateError("dealer template is not found")

    def get_dealer_region(self, frame: np.ndarray) -> Region:
        result = cv2.matchTemplate(frame, self.dealer_tmpl, cv2.TM_CCOEFF_NORMED)
        max_loc = cv2.minMaxLoc(result)[3]

        (start_x, start_y) = max_loc
        end_x = start_x + self.dealer_tmpl.shape[1]
        end_y = start_y + self.dealer_tmpl.shape[0]

        return Region(
            start=Point(start_x, start_y),
            end=Point(end_x, end_y),
        )

    def get_point_position(
        self, point: Point, dimensions: Dimensions, ratio: Tuple
    ) -> int:
        regions = self.split_into_regions(dimensions, ratio)
        for i, region in enumerate(regions):
            if (
                region.start.x < point.x < region.end.x
                and region.start.y < point.y < region.end.y
            ):
                return i
        return -1

    @staticmethod
    def split_into_regions(dimensions: Dimensions, ratio: Tuple) -> List[Region]:
        w = dimensions.width // ratio[0]
        h = dimensions.height // ratio[1]

        iterations = ratio[0] * ratio[1]
        regions = []

        x, y = 0, 0
        for _ in range(iterations // 2):
            r = Region(start=Point(x, y), end=Point(x + w, y + h))
            regions.append(r)
            x += w

        x, y = 0, h
        for _ in range(iterations // 2):
            r = Region(start=Point(x, y), end=Point(x + w, y + h))
            regions.append(r)
            x += w
        return regions
