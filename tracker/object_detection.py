import cv2


class ObjectDetection:
    """Detect objects on an window frame"""

    def __init__(self, template_path):
        self.template_path = template_path

    def get_dealer(self, frame):
        dealer_tmpl = cv2.imread(
            f"{self.template_path}/dealer.png", cv2.IMREAD_UNCHANGED
        )
        if dealer_tmpl is None:
            return None

        result = cv2.matchTemplate(frame, dealer_tmpl, cv2.TM_CCOEFF_NORMED)
        (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(result)
        (start_x, start_y) = max_loc
        end_x = start_x + dealer_tmpl.shape[1]
        end_y = start_y + dealer_tmpl.shape[0]

        return cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 0, 0), 3)
