import cv2


class ObjectDetection:
    """Detect objects on an window frame"""

    def __init__(self, template_path, template_format):
        self.template_path = template_path
        self.template_format = template_format

        self.dealer_tmpl = cv2.imread(
            f"{self.template_path}/dealer.{self.template_format}", cv2.IMREAD_UNCHANGED
        )
        if self.dealer_tmpl is None:
            raise Exception("Dealer template is not found")

    def get_dealer(self, frame):
        result = cv2.matchTemplate(frame, self.dealer_tmpl, cv2.TM_CCOEFF_NORMED)
        max_loc = cv2.minMaxLoc(result)[3]
        (start_x, start_y) = max_loc
        end_x = start_x + self.dealer_tmpl.shape[1]
        end_y = start_y + self.dealer_tmpl.shape[0]
        return (start_x, start_y, end_x, end_y)
