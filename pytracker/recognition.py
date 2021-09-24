# pyright: reportMissingImports=false
import multiprocessing as mp

import cv2
import numpy as np


class Recognition:
    """Provides API to recognize objects in a given image"""

    def __init__(self, log, data_queue, event_queues):
        self.log = log
        self.data_queue = data_queue
        self.event_queues = event_queues
        self.image_parts = {i: None for i in range(len(event_queues))}

    def run(self):
        prefix = f"{mp.current_process().name}:"
        output_file_path = "images/{}.png"
        img_idx = 1

        while True:
            try:
                idx, img = self.data_queue.get()

                img = np.asarray(img, dtype=np.uint8)
                gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                self.image_parts[idx] = gray

                if all(i is not None for i in self.image_parts.values()):
                    full = self.make_full_image()
                    cv2.imwrite(output_file_path.format(img_idx), full)
                    self.log.debug(f"{prefix} image {img_idx} saved")

                    self.clear_image_parts()
                    img_idx += 1
            except (KeyboardInterrupt, SystemExit):
                self.data_queue.close()
                self.log.warn(f"{prefix} interruption; exiting...")
                return

    def make_full_image(self):
        top = np.hstack([self.image_parts[0], self.image_parts[1]])
        bottom = np.hstack([self.image_parts[2], self.image_parts[3]])
        return np.vstack([top, bottom])

    def clear_image_parts(self):
        for i in range(len(self.event_queues)):
            self.image_parts[i] = None
            self.event_queues[i].put(True)
