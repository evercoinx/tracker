# pyright: reportMissingImports=false
import argparse
import logging as log
import multiprocessing as mp

from .grabber import grab
from .processor import process


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--loglevel",
        type=str,
        default="info",
        help="log level: debug, info, warn, error",
    )
    ap.add_argument("--display", type=str, default=":0.0", help="display number")
    ap.add_argument(
        "--top", type=int, default=0, help="screen y-coordinate of upper-left corner"
    )
    ap.add_argument(
        "--left", type=int, default=0, help="screen x-coordinate of upper-left corner"
    )
    ap.add_argument("--width", type=int, default=800, help="screen width")
    ap.add_argument("--height", type=int, default=600, help="screen height")
    args = vars(ap.parse_args())

    log_level = args["loglevel"].upper()
    log.basicConfig(
        level=log.getLevelName(log_level),
        format="%(asctime)s - %(levelname)-7s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    roi = {
        "top": args["top"],
        "left": args["left"],
        "width": args["width"],
        "height": args["height"],
    }
    queue = mp.Queue()

    log.info("job started")
    grabber = mp.Process(target=grab, args=(log, queue, args["display"], roi))
    processor = mp.Process(target=process, args=(log, queue))

    grabber.start()
    processor.start()

    grabber.join()
    processor.join()


if __name__ == "__main__":
    main()
