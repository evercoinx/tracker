from typing import ClassVar, Dict, List, NamedTuple, Optional, Tuple

import cv2
import numpy as np
from typing_extensions import Literal

from tracker.error import ImageError


class Point(NamedTuple):
    x: int
    y: int


class Region(NamedTuple):
    start: Point
    end: Point


class ObjectDetection:
    """Detects object regions on an window frame"""

    seat_region_percentages: ClassVar[List[Tuple[float, float]]] = [
        # top left region, index 0
        (0.125, 0.05),
        (0.40, 0.45),
        # top middle region, index 1
        (0.40, 0.05),
        (0.60, 0.45),
        # top right region, index 2
        (0.60, 0.05),
        (0.85, 0.45),
        # bottom left region, index 3
        (0.125, 0.45),
        (0.40, 0.80),
        # bottom middle region, index 4
        (0.40, 0.45),
        (0.60, 0.80),
        # bottom right region, index 5
        (0.60, 0.45),
        (0.85, 0.80),
    ]

    min_template_confidence: ClassVar[float] = 0.8

    template_path: str
    image_format: str
    seat_regions_cache: Dict[Tuple[int, int], List[Region]]
    dealer_template: np.ndarray
    pocket_cards_templates: List[np.ndarray]

    def __init__(self, template_path: str, image_format: str) -> None:
        self.template_path = template_path
        self.image_format = image_format
        self.seat_regions_cache = {}

        dealer_path = f"{self.template_path}/dealer.{self.image_format}"
        dealer_template = cv2.imread(dealer_path, cv2.IMREAD_UNCHANGED)
        if dealer_template is None:
            raise ImageError("unable to read template image of dealer", dealer_path)
        self.dealer_template = dealer_template

        self.pocket_cards_templates = []
        for i in range(self.seat_count):
            card_path = f"{self.template_path}/pocket_cards_{i}.{self.image_format}"
            pocket_cards_template = cv2.imread(card_path, cv2.IMREAD_UNCHANGED)
            if pocket_cards_template is None:
                raise ImageError(
                    "unable to read template image of pocket card", card_path
                )
            self.pocket_cards_templates.append(pocket_cards_template)

    @property
    def seat_count(self) -> int:
        return len(type(self).seat_region_percentages) // 2

    @property
    def table_card_count(self) -> Literal[5]:
        return 5

    def get_seat_regions(self, frame_width: int, frame_height: int) -> List[Region]:
        cache_key = (frame_width, frame_height)
        if cache_key in self.seat_regions_cache:
            return self.seat_regions_cache[cache_key]

        regions: List[Region] = []
        points: List[Point] = []
        for i, (x, y) in enumerate(type(self).seat_region_percentages):
            p = self.get_point_by_percentage(x, y, frame_width, frame_height)
            points.append(p)

            if i % 2 != 0:
                r = Region(start=points[0], end=points[1])
                regions.append(r)
                points = []

        self.seat_regions_cache[cache_key] = regions
        return regions

    def detect_hand_number(self, frame: np.ndarray) -> Region:
        return Region(
            start=Point(73, 24),
            end=Point(174, 39),
        )

    def detect_hand_time(self, frame: np.ndarray) -> Region:
        return Region(
            start=Point(857, 22),
            end=Point(912, 36),
        )

    def detect_total_pot(self, frame: np.ndarray) -> Region:
        return Region(
            start=Point(462, 160),
            end=Point(553, 181),
        )

    def detect_seat_number(self, frame: np.ndarray, index: int) -> Region:
        start_points = [
            Point(172, 113),
            Point(433, 81),
            Point(664, 113),
            Point(138, 334),
            Point(431, 342),
            Point(682, 334),
        ]
        if index > len(start_points) - 1:
            raise ValueError(f"invalid seat number index: {index}")

        (w, h) = 199, 15
        end_point = Point(start_points[index].x + w, start_points[index].y + h)
        return Region(start=start_points[index], end=end_point)

    def detect_seat_action(self, frame: np.ndarray, index: int) -> Region:
        start_points = [
            Point(172, 100),
            Point(433, 68),
            Point(664, 100),
            Point(138, 321),
            Point(431, 328),
            Point(682, 321),
        ]
        if index > len(start_points) - 1:
            raise ValueError(f"invalid seat action index: {index}")

        (w, h) = 119, 14
        end_point = Point(start_points[index].x + w, start_points[index].y + h)
        return Region(start=start_points[index], end=end_point)

    def detect_seat_stake(self, frame: np.ndarray, index: int) -> Region:
        start_points = [
            Point(294, 154),
            Point(423, 131),
            Point(602, 153),
            Point(287, 288),
            Point(0, 0),
            Point(595, 290),
        ]
        if index > len(start_points) - 1:
            raise ValueError(f"invalid seat stake index: {index}")

        (w, h) = 56, 19
        end_point = Point(start_points[index].x + w, start_points[index].y + h)
        return Region(start=start_points[index], end=end_point)

    def detect_seat_balance(self, frame: np.ndarray, index: int) -> Region:
        start_points = [
            Point(172, 130),
            Point(433, 98),
            Point(664, 130),
            Point(138, 351),
            Point(431, 357),
            Point(682, 351),
        ]
        if index > len(start_points) - 1:
            raise ValueError(f"invalid seat balance index: {index}")

        (w, h) = 119, 16
        end_point = Point(start_points[index].x + w, start_points[index].y + h)
        return Region(start=start_points[index], end=end_point)

    def detect_table_card(self, frame: np.ndarray, index: int) -> Region:
        start_points = [
            Point(368, 185),
            Point(414, 185),
            Point(460, 185),
            Point(506, 185),
            Point(554, 185),
        ]
        if index > len(start_points) - 1:
            raise ValueError(f"invalid table card index: {index}")

        (w, h) = 38, 32
        end_point = Point(start_points[index].x + w, start_points[index].y + h)
        return Region(start=start_points[index], end=end_point)

    def detect_dealer(self, frame: np.ndarray) -> Optional[Region]:
        return self.detect_object_by_template(frame, self.dealer_template)

    def detect_pocket_cards(self, frame: np.ndarray, index: int) -> Optional[Region]:
        return self.detect_object_by_template(frame, self.pocket_cards_templates[index])

    def detect_object_by_template(
        self, frame: np.ndarray, template: np.ndarray
    ) -> Optional[Region]:
        result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val < type(self).min_template_confidence:
            return None

        (start_x, start_y) = max_loc
        start_point = Point(start_x, start_y)

        h, w = template.shape[:2]
        end_point = Point(start_point.x + w, start_point.y + h)
        return Region(start=start_point, end=end_point)

    @staticmethod
    def get_point_by_percentage(
        x_percentage: float,
        y_percentage: float,
        frame_width: int,
        frame_height: int,
    ) -> Point:
        x = int(x_percentage * frame_width)
        y = int(y_percentage * frame_height)
        return Point(x, y)
