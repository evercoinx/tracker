import argparse
import logging
import os
import sys
from multiprocessing import Event, Process, Queue

from tracker import __version__
from tracker.detector import ObjectDetector
from tracker.screen import Screen

STREAM_PATH = "./stream"
FRAME_FORMAT = "png"


def main():
    args = validate_args(parse_args())
    if args["replay"]:
        replay_session(args)
    else:
        play_session(args)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--replay",
        dest="replay",
        action="store_true",
        help="replay saved session",
    )
    ap.add_argument(
        "--display", type=str, default=":0.0", help="display number; defaults to :0.0"
    )
    ap.add_argument(
        "--windows",
        type=list,
        default=["0", "1", "2", "3"],
        help="windows to watch; defaults to 0123",
    )
    ap.add_argument(
        "--screen-width",
        type=int,
        default=1920,
        help="screen width in px; defaults to 1920",
    )
    ap.add_argument(
        "--screen-height",
        type=int,
        default=1080,
        help="screen height in px; defaults to 1080",
    )
    ap.add_argument(
        "--left-margin",
        type=int,
        default=0,
        help="left margin of screen in px; defaults to 0",
    )
    ap.add_argument(
        "--top-margin",
        type=int,
        default=0,
        help="top margin of screen in px; defaults to 0",
    )
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    return vars(ap.parse_args())


def validate_args(args):
    debug_env = os.environ.get("DEBUG", "0").strip()
    log_level = "DEBUG" if debug_env == "1" else "INFO"
    logging.basicConfig(
        level=logging.getLevelName(log_level),
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    windows = args["windows"]
    if len(windows) > 4:
        logging.critical(f"Too many windows to play: {len(windows)}")
        sys.exit(1)

    if args["replay"]:
        return args

    # the environment variable formatted as hostname:display.screen
    display_env = os.environ.get("DISPLAY", "").strip()
    if not display_env:
        logging.critical("Display is not set")
        sys.exit(1)

    parsed_display = display_env.split(":")
    display = f":{parsed_display[1]}"
    if display != args["display"]:
        logging.critical(f"Display mismatch: {display} != {args['display']}")
        sys.exit(1)

    return {
        **args,
        **{
            "log_level": log_level,
            "display": display,
        },
    }


def play_session(args):
    win_coords = Screen.calculate_window_coords(
        args["windows"],
        args["screen_width"],
        args["screen_height"],
        args["left_margin"],
        args["top_margin"],
    )
    win_count = len(win_coords)

    queue = Queue(win_count)
    events = [Event() for _ in range(win_count)]

    screen = Screen(queue, events, stream_path=STREAM_PATH, frame_format=FRAME_FORMAT)
    detector = ObjectDetector(
        queue, events, stream_path=STREAM_PATH, frame_format=FRAME_FORMAT
    )
    procs = []

    for (i, wc) in enumerate(win_coords):
        sp = Process(
            name=f"screen-{i}",
            target=screen.capture,
            args=(args["display"], wc, i),
        )
        procs.append(sp)

        dp = Process(name=f"detector-{i}", target=detector.play_live_stream, args=())
        procs.append(dp)

    for p in procs:
        p.start()

    for p in procs:
        p.join()


def replay_session(args):
    detector = ObjectDetector(
        None, [], stream_path=STREAM_PATH, frame_format=FRAME_FORMAT
    )
    detector.play_saved_stream(args["windows"])


if __name__ == "__main__":
    main()
