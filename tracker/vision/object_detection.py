from typing import ClassVar, Dict, List, NamedTuple, Optional, Tuple

import cv2
import numpy as np


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
            raise OSError(f"Unable to read template image of dealer at {dealer_path}")
        self.dealer_template = dealer_template

        self.pocket_cards_templates = []
        for i in range(6):
            card_path = f"{self.template_path}/pocket_{i}.{self.image_format}"
            pocket_cards_template = cv2.imread(card_path, cv2.IMREAD_UNCHANGED)
            if pocket_cards_template is None:
                raise OSError(
                    f"Unable to read template image of pocket card at {card_path}"
                )
            self.pocket_cards_templates.append(pocket_cards_template)

    def get_seat_regions(self, frame_width: int, frame_height: int) -> List[Region]:
        cache_key = (frame_width, frame_height)
        if cache_key in self.seat_regions_cache:
            return self.seat_regions_cache[cache_key]

        regions: List[Region] = []
        points: List[Point] = []
        for i, (x, y) in enumerate(type(self).seat_region_percentages):
            p = self._get_scaled_point(x, y, frame_width, frame_height)
            points.append(p)

            if i % 2 != 0:
                r = Region(start=points[0], end=points[1])
                regions.append(r)
                points = []

        self.seat_regions_cache[cache_key] = regions
        return regions

    def detect_hand_number(self, frame: np.ndarray) -> Region:
        (h, w) = frame.shape[:2]
        start = self._get_scaled_point(0.07, 0.05, w, h)
        end = self._get_scaled_point(0.185, 0.08, w, h)
        return Region(start, end)

    def detect_hand_time(self, frame: np.ndarray) -> Region:
        (h, w) = frame.shape[:2]
        start = self._get_scaled_point(0.89, 0.04, w, h)
        end = self._get_scaled_point(0.955, 0.08, w, h)
        return Region(start, end)

    def detect_total_pot(self, frame: np.ndarray) -> Region:
        (h, w) = frame.shape[:2]
        start = self._get_scaled_point(0.48, 0.32, w, h)
        end = self._get_scaled_point(0.57, 0.37, w, h)
        return Region(start, end)

    @staticmethod
    def _get_scaled_point(
        x_percent: float,
        y_percent: float,
        total_width: int,
        total_height: int,
    ) -> Point:
        x = int(x_percent * total_width)
        y = int(y_percent * total_height)
        return Point(x, y)

    def detect_seat_action(self, frame: np.ndarray, index: int) -> Region:
        start_percents = [
            (0.18, 0.205),
            (0.455, 0.14),
            (0.695, 0.205),
            (0.145, 0.655),
            (0.45, 0.665),
            (0.715, 0.655),
        ]
        return self._get_scaled_region(
            frame,
            start_percents,
            index,
            percent_width=0.12,
            percent_height=0.03,
        )

    def detect_seat_name(self, frame: np.ndarray, index: int) -> Region:
        start_percents = [
            (0.18, 0.23),
            (0.455, 0.165),
            (0.695, 0.23),
            (0.145, 0.68),
            (0.45, 0.695),
            (0.715, 0.68),
        ]
        return self._get_scaled_region(
            frame,
            start_percents,
            index,
            percent_width=0.12,
            percent_height=0.035,
        )

    def detect_seat_balance(self, frame: np.ndarray, index: int) -> Region:
        start_percents = [
            (0.18, 0.265),
            (0.455, 0.20),
            (0.695, 0.265),
            (0.145, 0.715),
            (0.45, 0.73),
            (0.715, 0.715),
        ]
        return self._get_scaled_region(
            frame,
            start_percents,
            index,
            percent_width=0.12,
            percent_height=0.035,
        )

    def detect_seat_stake(self, frame: np.ndarray, index: int) -> Region:
        start_percents = [
            (0.305, 0.315),
            (0.43, 0.27),
            (0.615, 0.315),
            (0.295, 0.585),
            (0.0, 0.0),
            (0.605, 0.59),
        ]
        return self._get_scaled_region(
            frame,
            start_percents,
            index,
            percent_width=0.075,
            percent_height=0.04,
        )

    def detect_table_card(self, frame: np.ndarray, index: int) -> Region:
        start_percents = [
            (0.384, 0.377),
            (0.432, 0.377),
            (0.48, 0.377),
            (0.528, 0.377),
            (0.578, 0.377),
        ]
        return self._get_scaled_region(
            frame,
            start_percents,
            index,
            percent_width=0.039,
            percent_height=0.065,
        )

    def _get_scaled_region(
        self,
        frame: np.ndarray,
        start_percents: List[Tuple[float, float]],
        index: int,
        *,
        percent_width: float,
        percent_height: float,
    ) -> Region:
        if index > len(start_percents) - 1:
            raise ValueError(f"invalid index of start percents: {index}")

        x_percent, y_percent = start_percents[index]
        (frame_height, frame_width) = frame.shape[:2]

        start = self._get_scaled_point(x_percent, y_percent, frame_width, frame_height)
        end = self._get_scaled_point(
            x_percent + percent_width,
            y_percent + percent_height,
            frame_width,
            frame_height,
        )
        return Region(start, end)

    def detect_dealer(self, frame: np.ndarray) -> Optional[Region]:
        return self._get_object_by_template(frame, self.dealer_template)

    def detect_pocket_cards(self, frame: np.ndarray, index: int) -> Optional[Region]:
        return self._get_object_by_template(frame, self.pocket_cards_templates[index])

    def _get_object_by_template(
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
