from typing import Dict, List, NamedTuple, Optional, Tuple

import cv2
import numpy as np

from tracker.error import TemplateError


class Point(NamedTuple):
    x: int
    y: int


class Region(NamedTuple):
    start: Point
    end: Point


class ObjectDetection:
    """Detects object regions on an window frame"""

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

    MIN_TEMPLATE_CONFIDENCE = 0.8

    template_path: str
    template_format: str
    player_regions_cache: Dict[Tuple, List[Region]]
    dealer_template: np.ndarray
    hand_cards_templates: List[np.ndarray]

    def __init__(self, template_path: str, template_format: str) -> None:
        self.template_path = template_path
        self.template_format = template_format
        self.player_regions_cache: Dict[Tuple, List[Region]] = {}

        self.dealer_template = cv2.imread(
            f"{self.template_path}/dealer.{self.template_format}", cv2.IMREAD_UNCHANGED
        )
        if self.dealer_template is None:
            raise TemplateError("dealer template is not found")

        self.hand_cards_templates = []
        for i in range(6):
            hand_cards_template = cv2.imread(
                f"{self.template_path}/hand_cards_{i}.{self.template_format}",
                cv2.IMREAD_UNCHANGED,
            )
            if hand_cards_template is None:
                raise TemplateError(f"cards template #{i} is not found")
            self.hand_cards_templates.append(hand_cards_template)

    def get_player_regions(self, width: int, height: int) -> List[Region]:
        cache_key = (width, height)
        if cache_key in self.player_regions_cache:
            return self.player_regions_cache[cache_key]

        regions: List[Region] = []
        points: List[Point] = []
        for i, (x, y) in enumerate(ObjectDetection.PLAYER_REGION_PERCENTAGE):
            p = self.get_point_by_percentage(x, y, width, height)
            points.append(p)

            if i % 2 != 0:
                r = Region(start=points[0], end=points[1])
                regions.append(r)
                points = []

        self.player_regions_cache[cache_key] = regions
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

    def detect_dealer(self, frame: np.ndarray) -> Optional[Region]:
        return self.detect_object_by_template(frame, self.dealer_template)

    def detect_hand_card(self, frame: np.ndarray, index: int) -> Optional[Region]:
        return self.detect_object_by_template(frame, self.hand_cards_templates[index])

    def detect_object_by_template(
        self, frame: np.ndarray, template: np.ndarray
    ) -> Optional[Region]:
        result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val < ObjectDetection.MIN_TEMPLATE_CONFIDENCE:
            return None

        (start_x, start_y) = max_loc
        start_point = Point(start_x, start_y)

        h, w = template.shape[:2]
        end_point = Point(start_point.x + w, start_point.y + h)
        return Region(start=start_point, end=end_point)

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
