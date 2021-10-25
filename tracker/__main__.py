import argparse
import logging
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
    ap.add_argument("--display", type=str, default=":0.0", help="display number")
    ap.add_argument("--width", type=int, default=1920, help="screen width")
    ap.add_argument("--height", type=int, default=1080, help="screen height")
    args = vars(ap.parse_args())

    log_level = args["loglevel"].upper()
    logging.basicConfig(
        level=logging.getLevelName(log_level),
        format="%(asctime)s - %(levelname)-7s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

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
