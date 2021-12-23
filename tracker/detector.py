import logging
from multiprocessing import current_process

import cv2
import numpy as np


class ObjectDetector:
    """Detect objects on an image"""

    def __init__(self, queue, events):
        self.queue = queue
        self.events = events

    def run(self):
        prefix = f"{current_process().name}:"
        img_seq = 1

        while True:
            try:
                win_idx, win_img = self.queue.get()
                win_arr = np.asarray(win_img, dtype=np.uint8)

                gray = cv2.cvtColor(win_arr, cv2.COLOR_BGRA2GRAY)
                cv2.imwrite(
                    f"./images/original/table{win_idx+1}/{img_seq}.png",
                    gray,
                )
                logging.info(f"{prefix} table {win_idx+1}: image {img_seq}.png saved")

                img_seq += 1
                self.events[win_idx].set()

            except (KeyboardInterrupt, SystemExit):
                self.queue.close()
                logging.warn(f"{prefix} interruption; exiting...")
                return
