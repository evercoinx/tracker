import argparse
import logging
import os
import sys
from multiprocessing import Event, Process, Queue

from tracker.grabber import Grabber
from tracker.stitcher import Stitcher


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--loglevel",
        type=str,
        default="info",
        help="log level: debug, info, warn, error",
    )
    ap.add_argument("--display", type=str, required=True, help="display number")
    ap.add_argument("--width", type=int, default=1920, help="screen width")
    ap.add_argument("--height", type=int, default=1080, help="screen height")
    args = vars(ap.parse_args())

    log_level = args["loglevel"].upper()
    logging.basicConfig(
        level=logging.getLevelName(log_level),
        format="%(asctime)s - %(levelname)-7s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # The display variable has the following format: hostname:display.screen
    display = os.environ.get("DISPLAY", "")
    if not display:
        logging.critical("Display is not set")
        sys.exit(1)

    parsed_display = display.split(":")
    if args["display"] != f":{parsed_display[1]}":
        logging.critical(
            f"Display mismatch: expected {args['display']}, actual :{parsed_display[1]}"
        )
        sys.exit(1)

    rois = Grabber.get_rois(args["width"], args["height"])
    roi_count = len(rois)

    queue = Queue(roi_count)
    events = [Event() for _ in range(roi_count)]

    grabber = Grabber(queue, events)
    stitcher = Stitcher(queue, events)

    procs = []
    for (i, roi) in enumerate(rois):
        gp = Process(
            name=f"grabber-{i}", target=grabber.capture, args=(args["display"], roi, i)
        )
        procs.append(gp)

    sp = Process(name="stitcher", target=stitcher.run, args=())
    procs.append(sp)

    for p in procs:
        p.start()

    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
