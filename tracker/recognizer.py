import logging

import cv2


class Recognizer:
    """Provides API to recognize objects in a given image"""

    def run(self, image, img_seq_num):
        prefix = "recognizer:"
        output_path = "images/{}.png"

        cv2.imwrite(output_path.format(img_seq_num), image)
        logging.info(f"{prefix} image #{img_seq_num} saved")
