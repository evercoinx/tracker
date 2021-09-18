import cv2
import numpy as np


def process(log, queue):
    i = 0
    out_path = "screenshots/image_{}.png"

    while True:
        screenshot = queue.get()
        if screenshot is None:
            return

        img = np.asarray(screenshot)
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        cv2.imwrite(out_path.format(i), gray)
        log.debug(f"image {i} saved")

        i += 1
