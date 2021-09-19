# pyright: reportMissingImports=false
import cv2
import numpy as np


class Recognition:
    """Provides API to recognize objects in a given image"""

    def __init__(self, name, *, log, queue):
        self.name = name
        self.log = log
        self.queue = queue

    def run(self):
        prefix = f"{self.name}:"
        out_path = "images/{}.png"
        i = 0

        while True:
            try:
                img = self.queue.get()
                if img is None:
                    self.log.info(f"{prefix} no more data; exiting...")
                    return

                img = np.asarray(img, dtype=np.uint8)
                gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

                cv2.imwrite(out_path.format(i), gray)
                self.log.debug(f"{prefix} image {i} saved")

                i += 1
            except (KeyboardInterrupt, SystemExit):
                self.log.warn(f"{prefix} interruption; exiting...")
                return
