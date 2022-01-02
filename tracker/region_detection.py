import cv2
import numpy as np

from tracker.error import TemplateError
from tracker.utils import Point, Region


class RegionDetection:
    """Detects regions on an window frame"""

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
        start_point = Point(start_x, start_y)

        h, w = self.dealer_tmpl.shape[:2]
        end_point = self.calculate_end_point(start_point, w, h)
        return Region(start=start_point, end=end_point)

    def get_hand_number_region(self, frame: np.ndarray) -> Region:
        return Region(
            start=Point(73, 24),
            end=Point(174, 39),
        )

    def get_hand_time_region(self, frame: np.ndarray) -> Region:
        return Region(
            start=Point(857, 22),
            end=Point(912, 36),
        )

    def get_total_pot_region(self, frame: np.ndarray) -> Region:
        return Region(
            start=Point(462, 160),
            end=Point(553, 181),
        )

    def get_seat_number_region(self, frame: np.ndarray, index: int) -> Region:
        points = [
            Point(138, 334),
            Point(172, 113),
            Point(433, 81),
            Point(664, 113),
            Point(682, 334),
            Point(431, 342),
        ]
        if index > len(points) - 1:
            raise ValueError(f"invalid seat number index: {index}")

        end_point = self.calculate_end_point(points[index], width=119, height=15)
        return Region(start=points[index], end=end_point)

    def get_seat_action_region(self, frame: np.ndarray, index: int) -> Region:
        points = [
            Point(138, 321),
            Point(172, 100),
            Point(433, 68),
            Point(664, 100),
            Point(682, 321),
            Point(431, 328),
        ]
        if index > len(points) - 1:
            raise ValueError(f"invalid seat action index: {index}")

        end_point = self.calculate_end_point(points[index], width=119, height=14)
        return Region(start=points[index], end=end_point)

    def get_seat_stake_region(self, frame: np.ndarray, index: int) -> Region:
        points = [
            Point(287, 288),
            Point(294, 154),
            Point(423, 131),
            Point(602, 153),
            Point(595, 290),
            Point(0, 0),
        ]
        if index > len(points) - 1:
            raise ValueError(f"invalid seat stake index: {index}")

        end_point = self.calculate_end_point(points[index], width=56, height=19)
        return Region(start=points[index], end=end_point)

    def get_seat_balance_region(self, frame: np.ndarray, index: int) -> Region:
        points = [
            Point(138, 351),
            Point(172, 130),
            Point(433, 98),
            Point(664, 130),
            Point(682, 351),
            Point(431, 357),
        ]
        if index > len(points) - 1:
            raise ValueError(f"invalid seat balance index: {index}")

        end_point = self.calculate_end_point(points[index], width=119, height=16)
        return Region(start=points[index], end=end_point)

    @staticmethod
    def calculate_end_point(start_point: Point, width: int, height: int) -> Point:
        end_x = start_point.x + width
        end_y = start_point.y + height
        return Point(end_x, end_y)
