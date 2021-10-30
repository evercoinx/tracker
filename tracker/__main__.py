import argparse
import logging
import os
import sys
from multiprocessing import Event, Process, Queue

from tracker.recognition import Recognition
from tracker.screen import Screen


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

    venv_path = os.environ.get("VIRTUAL_ENV", "")
    if not venv_path:
        logging.critical("Virtual environment is not set")
        sys.exit(1)

    if __package__ != os.path.basename(venv_path):
        logging.critical(
            f"Virtual environment mismatch: expected {__package__}, "
            f"acutal {os.path.basename(venv_path)}"
        )
        sys.exit(1)

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

    rois = Screen.get_rois(args["width"], args["height"])
    roi_count = len(rois)

    queue = Queue(roi_count)
    events = [Event() for _ in range(roi_count)]

    screen = Screen(queue, events)
    recognition = Recognition(queue, events)

    procs = []
    for (i, roi) in enumerate(rois):
        scr = Process(
            name=f"screen-{i}", target=screen.capture, args=(args["display"], roi, i)
        )
        procs.append(scr)

    rec = Process(name="recognition", target=recognition.run, args=())
    procs.append(rec)

    for p in procs:
        p.start()

    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
