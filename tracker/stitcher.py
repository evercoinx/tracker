import logging
from multiprocessing import current_process

import cv2
import numpy as np


class Stitcher:
    """Provides API to stitch image parts"""

    def __init__(self, queue, events, recognizer):
        self.queue = queue
        self.events = events
        self.table_images = [None] * len(events)
        self.recognizer = recognizer

    def run(self):
        prefix = f"{current_process().name}:"
        img_seq = 0

        while True:
            try:
                table_idx, image = self.queue.get()
                self.table_images[table_idx] = image

                if all(p is not None for p in self.table_images):
                    img_seq += 1

                    for idx, img in enumerate(self.table_images):
                        arr = np.asarray(img, dtype=np.uint8)
                        gray = cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)

                        cv2.imwrite(
                            f"./images/original/table{idx+1}/{img_seq}.png",
                            gray,
                        )
                        logging.info(
                            f"{prefix} image #{img_seq} for table #{idx+1} saved"
                        )

                        self.table_images[idx] = None
                        self.events[idx].set()

            except (KeyboardInterrupt, SystemExit):
                self.queue.close()
                logging.warn(f"{prefix} interruption; exiting...")
                return
