import argparse
import logging
import os
import sys
from multiprocessing import Event, Process, Queue

from tracker.detector import ObjectDetector
from tracker.screen import Screen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--log-level",
        type=str,
        default="info",
        help="log level: debug, info, warn, error",
    )
    ap.add_argument(
        "--display", type=str, default=":0.0", help="display number, e.g. :10.0"
    )
    ap.add_argument("--screen-width", type=int, default=1920, help="screen width in px")
    ap.add_argument(
        "--screen-height", type=int, default=1080, help="screen height in px"
    )
    ap.add_argument(
        "--left-margin", type=int, default=0, help="left margin of screen in px"
    )
    ap.add_argument(
        "--top-margin", type=int, default=0, help="top margin of screen in px"
    )
    args = vars(ap.parse_args())

    log_level = args["log_level"].upper()
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
            f"Display mismatch: actual :{parsed_display[1]}, expected {args['display']}"
        )
        sys.exit(1)

    win_coords = Screen.calculate_window_coords(
        args["screen_width"],
        args["screen_height"],
        args["left_margin"],
        args["top_margin"],
    )
    win_count = len(win_coords)

    queue = Queue(win_count)
    events = [Event() for _ in range(win_count)]

    procs = []
    screen = Screen(queue, events)
    detector = ObjectDetector(queue, events)

    for (i, wc) in enumerate(win_coords):
        sp = Process(
            name=f"screen-{i}", target=screen.capture, args=(args["display"], wc, i)
        )
        procs.append(sp)

        dp = Process(name=f"detector-{i}", target=detector.run, args=())
        procs.append(dp)

    for p in procs:
        p.start()

    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
