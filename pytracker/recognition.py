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
        self.images = {i: None for i in range(len(event_queues))}

    def run(self):
        prefix = f"{mp.current_process().name}:"
        out_path = "images/{}.png"
        img_idx = 1

        while True:
            try:
                idx, img = self.data_queue.get()

                img = np.asarray(img, dtype=np.uint8)
                gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                self.images[idx] = gray

                if (
                    self.images[0] is not None
                    and self.images[1] is not None
                    and self.images[2] is not None
                    and self.images[3] is not None
                ):
                    top = np.hstack([self.images[0], self.images[1]])
                    bottom = np.hstack([self.images[2], self.images[3]])
                    full = np.vstack([top, bottom])
                    cv2.imwrite(out_path.format(img_idx), full)

                    for i in range(4):
                        self.images[i] = None
                        self.event_queues[i].put(True)
                    self.log.debug(f"{prefix} image {img_idx} saved")

                    img_idx += 1
            except (KeyboardInterrupt, SystemExit):
                self.data_queue.close()
                self.log.warn(f"{prefix} interruption; exiting...")
                return
