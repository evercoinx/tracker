# pyright: reportMissingImports=false
import argparse
import logging as log
import sys
from multiprocessing import Process, Queue

import cv2
import numpy as np
from mss.linux import MSS as mss


def grab(log, queue, display, area):
    with mss(display) as sct:
        for i in range(10):
            screenshot = sct.grab(area)
            queue.put(screenshot)
            log.debug(f"screenshot {i} grabbed")

    queue.put(None)


def process(log, queue):
    i = 0
    out_path = "screenshots/image_{}.png"

    while True:
        screenshot = queue.get()
        if screenshot is None:
            log.info("job finished")
            sys.exit(0)

        img = np.asarray(screenshot)
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        cv2.imwrite(out_path.format(i), gray)
        i += 1

        log.debug(f"image {i} saved")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--loglevel",
        type=str,
        default="info",
        help="log level: debug, info, warn, error",
    )
    ap.add_argument("--display", type=str, default=":0.0", help="output display")
    ap.add_argument("--top", type=int, default=0, help="top screen margin")
    ap.add_argument("--left", type=int, default=0, help="left screen margin")
    ap.add_argument("--width", type=int, default=800, help="screen width")
    ap.add_argument("--height", type=int, default=600, help="screen height")
    args = ap.parse_args()

    log_level = args.loglevel.upper()
    log.basicConfig(
        level=log.getLevelName(log_level),
        format="%(asctime)s - %(levelname)-8s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    area = {
        "top": args.top,
        "left": args.left,
        "width": args.width,
        "height": args.height,
    }
    queue = Queue()

    log.info("job started")
    Process(target=grab, args=(log, queue, args.display, area)).start()
    Process(target=process, args=(log, queue)).start()


if __name__ == "__main__":
    main()
