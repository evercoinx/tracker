import argparse
import logging
import os
import sys
import traceback
from multiprocessing import Event, Process, Queue
from typing import Any, Dict, List

from tracker import __version__
from tracker.error import ValidationError
from tracker.image_classifier import ImageClassifier
from tracker.object_detection import ObjectDetection
from tracker.screen import Screen
from tracker.stream_player import GameMode, StreamPlayer
from tracker.text_recognition import TextRecognition

TEMPLATE_PATH = "./template"
DATASET_PATH = "./dataset"
IMAGE_FORMAT = "png"


def main() -> None:
    tr = TextRecognition()
    od = ObjectDetection(template_path=TEMPLATE_PATH, image_format=IMAGE_FORMAT)

    ic = ImageClassifier(dataset_path=DATASET_PATH, image_format=IMAGE_FORMAT)
    ic.train()

    try:
        args = validate_args(parse_args())
        if args["replay"]:
            replay_session(args, tr, od, ic)
            return

        play_session(args, tr, od, ic)
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

    if len(args["windows"]) > 4:
        raise ValidationError(f"too many windows specified: {len(args['windows'])}")

    # the environment variable formatted as hostname:display.screen
    display_env = os.environ.get("DISPLAY", "").strip()
    if not display_env:
        raise ValidationError("display is not set")

    parsed_display = display_env.split(":")
    display = f":{parsed_display[1]}"
    if not args["replay"] and display != args["display"]:
        raise ValidationError(f"display is {display}; want {args['display']}")

    save_regions = os.environ.get("SAVE_REGIONS", "").split(",")

    return {
        **args,
        **{
            "log_level": log_level,
            "display": display,
            "save_regions": save_regions,
        },
    }


def replay_session(
    args: Dict[str, Any],
    text_recognition: TextRecognition,
    object_detection: ObjectDetection,
    image_classifier: ImageClassifier,
) -> None:
    player = StreamPlayer(
        queue=None,
        events=[],
        stream_path=args["stream_path"],
        frame_format=IMAGE_FORMAT,
        game_mode=GameMode.REPLAY,
        save_regions=args["save_regions"],
        text_recognition=text_recognition,
        object_detection=object_detection,
        image_classifier=image_classifier,
    )
    player.replay(args["windows"])


def play_session(
    args: Dict[str, Any],
    text_recognition: TextRecognition,
    object_detection: ObjectDetection,
    image_classifier: ImageClassifier,
) -> None:
    win_screens = Screen.get_window_screens(
        args["windows"],
        args["left_margin"],
        args["top_margin"],
        args["screen_width"],
        args["screen_height"],
    )
    win_count = len(win_screens)

    queue = Queue(win_count)
    events = [Event() for _ in range(win_count)]

    player = StreamPlayer(
        queue=queue,
        events=events,
        stream_path=args["stream_path"],
        frame_format=IMAGE_FORMAT,
        game_mode=GameMode.PLAY,
        save_regions=args["save_regions"],
        text_recognition=text_recognition,
        object_detection=object_detection,
        image_classifier=image_classifier,
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
