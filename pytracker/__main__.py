# pyright: reportMissingImports=false
import argparse
import logging as log
import sys
import time

import cv2
import numpy as np
from mss.linux import MSS as mss


def main():
    # Parsing passed arguments
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-l",
        "--loglevel",
        type=str,
        default="info",
        help="set log level: debug, info, warn, error",
    )
    ap.add_argument("-d", "--display", type=str, default=":0.0", help="set display")
    args = ap.parse_args()

    # Setting up logging
    log_level = args.loglevel.upper()
    log.basicConfig(
        level=log.getLevelName(log_level),
        format="%(asctime)s - %(levelname)-8s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    with mss(display=args.display) as sct:
        # Part of the screen to capture
        display = {"top": 100, "left": 0, "width": 800, "height": 640}

        while True:
            last_time = time.time()

            # Get raw pixels from the screen, save it to a Numpy array
            img = np.array(sct.grab(display))

            # Display the picture in grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            cv2.imshow("OpenCV/Numpy grayscale", gray)
            # cv2.imwrite(f"file{i}.png", gray)

            print("fps: {}".format(1 / (time.time() - last_time)))

            # Press "q" to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    # Cleaning up resources
    cv2.destroyAllWindows()
    sys.exit(0)


if __name__ == "__main__":
    main()
