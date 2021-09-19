import cv2
import numpy as np


def process(log, queue):
    name = "processor"
    out_path = "images/{}.png"
    i = 0

    while True:
        try:
            img = queue.get()
            if img is None:
                log.info(f"{name}: no more data; exiting...")
                return

            img = np.asarray(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

            cv2.imwrite(out_path.format(i), gray)
            log.debug(f"{name}: image {i} saved")

            i += 1
        except (KeyboardInterrupt, SystemExit):
            log.warn(f"{name}: interruption; exiting...")
            break
