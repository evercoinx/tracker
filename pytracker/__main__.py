import argparse
import logging as log
import multiprocessing as mp

from .recognition import Recognition
from .screen import Screen, rois


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--loglevel",
        type=str,
        default="info",
        help="log level: debug, info, warn, error",
    )
    ap.add_argument("--display", type=str, default=":0.0", help="display number")
    args = vars(ap.parse_args())

    log_level = args["loglevel"].upper()
    log.basicConfig(
        level=log.getLevelName(log_level),
        format="%(asctime)s - %(levelname)-7s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    data_queue = mp.Queue(1)
    event_queues = [mp.Queue(1) for _ in range(len(rois))]

    screen = Screen(log=log, data_queue=data_queue, event_queues=event_queues)
    recognition = Recognition(log=log, data_queue=data_queue, event_queues=event_queues)

    procs = []
    for (i, roi) in enumerate(rois):
        scr = mp.Process(
            name=f"screen-{i}", target=screen.grab, args=(args["display"], roi, i)
        )
        procs.append(scr)

    rec = mp.Process(name="recognition", target=recognition.run, args=())
    procs.append(rec)

    for p in procs:
        p.start()

    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
