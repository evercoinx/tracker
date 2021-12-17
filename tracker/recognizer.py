import logging

import cv2
import numpy as np


class Recognizer:
    """Provides API to recognize objects in a grayscale image"""

    def run(self, image, img_seq_num):
        prefix = "recognizer:"

        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        edged = self.apply_canny(blurred, 0.33)

        cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        largest_cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

        for c in largest_cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.05 * peri, True)

            if len(approx) == 4:
                cv2.drawContours(image, [approx], -1, (255, 255, 255), 5)
                cv2.imwrite(f"images/original/{img_seq_num}.png", image)
                cv2.imwrite(f"images/processed/{img_seq_num}.png", edged)
                logging.info(f"{prefix} image #{img_seq_num} recognized")
                break

    @staticmethod
    def apply_canny(image, sigma):
        m = np.median(image)
        lower = max(0, (1.0 - sigma) * m)
        upper = min(255, (1.0 + sigma) * m)
        return cv2.Canny(image, int(lower), int(upper))
