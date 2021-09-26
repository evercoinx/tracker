import logging
from multiprocessing import current_process

import cv2
import numpy as np


class Recognition:
    """Provides API to recognize objects in a given image"""

    def __init__(self, queue, events):
        self.queue = queue
        self.events = events
        self.image_parts = [None for i in range(len(events))]

    def run(self):
        prefix = f"{current_process().name}:"
        output_file_path = "images/{}.png"
        img_idx = 1

        while True:
            try:
                idx, img = self.queue.get()

                img = np.asarray(img, dtype=np.uint8)
                gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                self.image_parts[idx] = gray

                if all(p is not None for p in self.image_parts):
                    full = self.make_full_image()
                    cv2.imwrite(output_file_path.format(img_idx), full)
                    logging.info(f"{prefix} image {img_idx} saved")

                    self.clear_image_parts()
                    img_idx += 1
            except (KeyboardInterrupt, SystemExit):
                self.queue.close()
                logging.warn(f"{prefix} interruption; exiting...")
                return

    def make_full_image(self):
        center = len(self.image_parts) // 2
        top = np.hstack(self.image_parts[:center])  # type: ignore
        bottom = np.hstack(self.image_parts[center:])  # type: ignore
        return np.vstack([top, bottom])

    def clear_image_parts(self):
        for i in range(len(self.events)):
            self.image_parts[i] = None
            self.events[i].set()
