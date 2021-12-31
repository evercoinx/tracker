import cv2

from tracker.error import TemplateError


class ObjectDetection:
    """Detect objects on an window frame"""

    def __init__(self, template_path, template_format):
        self.template_path = template_path
        self.template_format = template_format

        self.dealer_tmpl = cv2.imread(
            f"{self.template_path}/dealer.{self.template_format}", cv2.IMREAD_UNCHANGED
        )
        if self.dealer_tmpl is None:
            raise TemplateError("dealer template is not found")

    def get_dealer_coords(self, frame):
        result = cv2.matchTemplate(frame, self.dealer_tmpl, cv2.TM_CCOEFF_NORMED)
        max_loc = cv2.minMaxLoc(result)[3]
        (start_x, start_y) = max_loc
        end_x = start_x + self.dealer_tmpl.shape[1]
        end_y = start_y + self.dealer_tmpl.shape[0]
        return (start_x, start_y, end_x, end_y)

    def point_in_region(
        self, point, frame_width, frame_height, width_parts, height_parts
    ):
        regions = self.split_into_regions(
            frame_width, frame_height, width_parts, height_parts
        )

        for i, region in enumerate(regions):
            if region[0] < point[0] < region[2] and region[1] < point[1] < region[3]:
                return i
        return -1

    @staticmethod
    def split_into_regions(frame_width, frame_height, width_parts, height_parts):
        w = frame_width // width_parts
        h = frame_height // height_parts

        iterations = width_parts * height_parts
        regions = []

        x, y = 0, 0
        for _ in range(iterations // 2):
            regions.append((x, y, x + w, y + h))
            x += w

        x, y = 0, h
        for _ in range(iterations // 2):
            regions.append((x, y, x + w, y + h))
            x += w
        return regions
