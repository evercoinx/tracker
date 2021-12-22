import logging
from multiprocessing import current_process

import cv2
import numpy as np


class ObjectDetector:
    """Detects objects on an image"""

    def __init__(self, queue, events):
        self.queue = queue
        self.events = events
        self.tables = [None] * len(events)

    def run(self):
        prefix = f"{current_process().name}:"
        img_seq = 0

        while True:
            try:
                tbl_idx, tbl_img = self.queue.get()
                self.tables[tbl_idx] = tbl_img

                if all(p is not None for p in self.tables):
                    img_seq += 1

                    for idx, img in enumerate(self.tables):
                        arr = np.asarray(img, dtype=np.uint8)
                        gray = cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)

                        cv2.imwrite(
                            f"./images/original/table{idx+1}/{img_seq}.png",
                            gray,
                        )
                        logging.info(
                            f"{prefix} image #{img_seq} for table #{idx+1} saved"
                        )

                        self.tables[idx] = None
                        self.events[idx].set()

            except (KeyboardInterrupt, SystemExit):
                self.queue.close()
                logging.warn(f"{prefix} interruption; exiting...")
                return
