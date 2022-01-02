import argparse
import logging
import os
import sys
import traceback
from multiprocessing import Event, Process, Queue
from typing import Any, Dict, List

from tracker import __version__
from tracker.error import ValidationError
from tracker.object_recognition import ObjectRecognition
from tracker.region_detection import RegionDetection
from tracker.screen import Screen
from tracker.stream_player import StreamPlayer
from tracker.text_recognition import TextRecognition

TEMPLATE_PATH = "./template"
IMAGE_FORMAT = "png"


def main() -> None:
    try:
        args = validate_args(parse_args())
        if args["replay"]:
            replay_session(args)
            return

        play_session(args)
    except Exception as e:
        logging.critical(e)
        traceback.print_exc()
        sys.exit(1)


def parse_args() -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--replay",
        dest="replay",
        action="store_true",
        help="replay saved session",
    )
    ap.add_argument(
        "--stream-path",
        type=str,
        required=True,
        help="stream path",
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


def validate_args(args: Dict[str, Any]) -> Dict[str, Any]:
    debug_env = os.environ.get("DEBUG", "0").strip()
    log_level = "DEBUG" if debug_env == "1" else "INFO"
    logging.basicConfig(
        level=logging.getLevelName(log_level),
        format="%(asctime)s.%(msecs)03d - %(levelname)-8s - %(message)s",
        datefmt="%H:%M:%S",
    )

    windows_arg = args["windows"]
    if len(windows_arg) > 4:
        raise ValidationError(f"too many windows to play: {len(windows_arg)}")

    if args["replay"]:
        return args

    # the environment variable formatted as hostname:display.screen
    display_env = os.environ.get("DISPLAY", "").strip()
    if not display_env:
        raise ValidationError("display is not set")

    parsed_display = display_env.split(":")
    display = f":{parsed_display[1]}"
    if display != args["display"]:
        raise ValidationError(f"display is {display}; want {args['display']}")

    return {
        **args,
        **{
            "log_level": log_level,
            "display": display,
            "windows": [int(i) for i in windows_arg],
        },
    }


def replay_session(args: Dict[str, Any]) -> None:
    region_detection = RegionDetection(
        template_path=TEMPLATE_PATH, template_format=IMAGE_FORMAT
    )
    text_recognition = TextRecognition()
    object_recognition = ObjectRecognition()

    player = StreamPlayer(
        queue=None,
        events=[],
        stream_path=args["stream_path"],
        frame_format=IMAGE_FORMAT,
        region_detection=region_detection,
        text_recognition=text_recognition,
        object_recognition=object_recognition,
    )
    player.replay(args["windows"])


def play_session(args: Dict[str, Any]) -> None:
    win_screens = Screen.get_window_screens(
        args["windows"],
        args["screen_width"],
        args["screen_height"],
        args["left_margin"],
        args["top_margin"],
    )
    win_count = len(win_screens)

    queue = Queue(win_count)
    events = [Event() for _ in range(win_count)]

    region_detection = RegionDetection(
        template_path=TEMPLATE_PATH, template_format=IMAGE_FORMAT
    )
    text_recognition = TextRecognition()
    object_recognition = ObjectRecognition()

    player = StreamPlayer(
        queue=queue,
        events=events,
        stream_path=args["stream_path"],
        frame_format=IMAGE_FORMAT,
        region_detection=region_detection,
        text_recognition=text_recognition,
        object_recognition=object_recognition,
    )
    screen = Screen(
        queue=queue,
        events=events,
        stream_path=args["stream_path"],
        frame_format=IMAGE_FORMAT,
    )
    procs: List[Process] = []

    for (i, wc) in enumerate(win_screens):
        sp = Process(
            name=f"screen-{i}",
            target=screen.capture,
            args=(args["display"], wc, i),
        )
        procs.append(sp)

        pp = Process(name=f"player-{i}", target=player.play, args=())
        procs.append(pp)

    for p in procs:
        p.start()

    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
