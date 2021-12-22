import logging
from multiprocessing import current_process

import cv2
import numpy as np


class ObjectDetector:
    """Detects objects on an image"""

    def __init__(self, queue, events):
        self.queue = queue
        self.events = events

    def run(self):
        prefix = f"{current_process().name}:"
        img_seq = 1

        while True:
            try:
                idx, img = self.queue.get()

                arr = np.asarray(img, dtype=np.uint8)
                gray = cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)

                cv2.imwrite(
                    f"./images/original/table{idx+1}/{img_seq}.png",
                    gray,
                )
                logging.info(f"{prefix} table {idx+1}: image {img_seq}.png saved")

                img_seq += 1
                self.events[idx].set()
            except (KeyboardInterrupt, SystemExit):
                self.queue.close()
                logging.warn(f"{prefix} interruption; exiting...")
                return
