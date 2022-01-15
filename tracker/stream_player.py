import logging
import os
import re
from collections import defaultdict
from datetime import datetime
from enum import Enum
from glob import glob
from multiprocessing import Queue, current_process
from multiprocessing.synchronize import Event
from pprint import pformat
from typing import Callable, DefaultDict, List, Optional, Tuple

import cv2
import numpy as np
from typing_extensions import TypedDict

from tracker.error import FrameError
from tracker.image_classifier import ImageClassifier
from tracker.object_detection import ObjectDetection, Region
from tracker.text_recognition import TextRecognition


class CardData(TypedDict):
    rank: str
    suit: str


class SeatData(TypedDict):
    number: int
    action: str
    stake: float
    balance: float
    playing: bool


class TextData(TypedDict):
    hand_number: int
    hand_time: datetime
    total_pot: float
    seats: List[SeatData]


class ObjectData(TypedDict):
    dealer_position: int
    playing_seats: List[bool]
    table_cards: List[CardData]


class SessionData(TypedDict):
    window_index: int
    frame_index: int
    hand_time: datetime
    total_pot: float
    dealer_position: int
    seats: List[SeatData]
    table_cards: List[CardData]


class GameMode(Enum):
    PLAY = 0
    REPLAY = 1


class StreamPlayer:
    """Plays a live stream or replays a saved one into the console"""

    game_mode: GameMode
    stream_path: str
    frame_format: str
    save_regions: List[str]
    text_recognition: TextRecognition
    object_detection: ObjectDetection
    image_classifier: ImageClassifier
    log_prefix: str
    session: DefaultDict[int, List[SessionData]]
    queue: Optional[Queue]
    events: List[Event]
    replay_windows: List[str]

    def __init__(
        self,
        game_mode: GameMode,
        stream_path: str,
        frame_format: str,
        save_regions: List[str],
        text_recognition: TextRecognition,
        object_detection: ObjectDetection,
        image_classifier: ImageClassifier,
        queue: Optional[Queue] = None,
        events: List[Event] = [],
        replay_windows: List[str] = [],
    ) -> None:
        self.game_mode = game_mode
        self.queue = queue
        self.events = events
        self.stream_path = stream_path
        self.frame_format = frame_format
        self.replay_windows = replay_windows
        self.save_regions = save_regions
        self.text_recognition = text_recognition
        self.object_detection = object_detection
        self.image_classifier = image_classifier
        self.log_prefix = ""
        self.session = defaultdict(list)

    def run(self) -> None:
        if self.game_mode == GameMode.PLAY:
            self._play()
        elif self.game_mode == GameMode.REPLAY:
            self._replay()
        else:
            raise ValueError(f"Unexpected game mode: {self.game_mode}")

    def _play(self) -> None:
        if self.queue is None:
            raise ValueError("Queue is not set")
        if not self.events:
            raise ValueError("Events are empty")

        frame_index = 0

        while True:
            try:
                window_index, frame = self.queue.get()
                self.log_prefix = self._get_log_prefix(window_index, frame_index)
                self._process_frame(frame, window_index, frame_index)

                frame_index += 1
                self.events[window_index].set()

            except (KeyboardInterrupt, SystemExit):
                self.queue.close()
                logging.warn(f"{self.log_prefix} interruption; exiting...")
                return

    def _replay(self) -> None:
        if not self.replay_windows:
            raise ValueError("Replay windows are empty")

        raw_frame_path_pattern = re.compile(
            r"window(["
            + re.escape(",".join(self.replay_windows))
            + r"])\/(\d+)_raw."
            + re.escape(self.frame_format)
            + r"$"
        )
        raw_frame_paths = glob(
            f"{self.stream_path}/window[{''.join(self.replay_windows)}]/"
            + f"*_raw.{self.frame_format}",
            recursive=True,
        )

        for p in sorted(raw_frame_paths, key=self._sort_path(raw_frame_path_pattern)):
            frame = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if frame is None:
                raise FrameError(f"unable to read raw frame path {p}")

            matches = re.findall(raw_frame_path_pattern, p)
            (window_index, frame_index) = matches[0]

            self.log_prefix = self._get_log_prefix(window_index, frame_index)
            self._process_frame(frame, int(window_index), int(frame_index))

    @staticmethod
    def _get_log_prefix(window_index: int, frame_index: int) -> str:
        proc_name = current_process().name
        if proc_name == "MainProcess":  # no multiprocessing
            proc_name = "player"
        return f"{proc_name}-w{window_index}-f{frame_index:<5} -"

    @staticmethod
    def _sort_path(pattern: re.Pattern) -> Callable[[str], Tuple[int, int]]:
        def match(path: str) -> Tuple[int, int]:
            matches = re.findall(pattern, path)
            if not matches:
                raise FrameError(f"unable to parse frame path {path}")
            return (
                int(matches[0][0]),
                int(matches[0][1]),
            )

        return match

    def _process_frame(
        self, frame: np.ndarray, window_index: int, frame_index: int
    ) -> None:
        inverted_frame = cv2.bitwise_not(frame)

        if self._should_save_frame("full"):
            full_frame = inverted_frame.copy()
            (h, w) = frame.shape[:2]
            for r in self.object_detection.get_seat_regions(w, h):
                full_frame = self._highlight_frame_region(full_frame, r)
            self._save_frame(full_frame, window_index, frame_index, "full")

        try:
            text_data = self._process_texts(inverted_frame, window_index, frame_index)
            object_data = self._process_objects(
                inverted_frame, window_index, frame_index, text_data["hand_number"]
            )
        except FrameError as err:
            logging.warn(f"{self.log_prefix} {err}\n")
            return

        if len(text_data["seats"]) != len(object_data["playing_seats"]):
            logging.warn(
                "invalid number of seats detected: "
                + f"{text_data['seats']} != {object_data['playing_seats']}"
            )
            return

        self.session[text_data["hand_number"]].append(
            {
                "window_index": window_index,
                "frame_index": frame_index,
                "hand_time": text_data["hand_time"],
                "total_pot": text_data["total_pot"],
                "dealer_position": object_data["dealer_position"],
                "seats": text_data["seats"],
                "table_cards": object_data["table_cards"],
            }
        )

        self._print_frame_info(text_data, object_data)

    def _process_texts(
        self, frame: np.ndarray, window_index: int, frame_index: int
    ) -> TextData:
        self.text_recognition.set_frame(frame)

        hand_number = self._get_hand_number(frame, window_index, frame_index)
        if not hand_number:
            raise FrameError("unable to recognize frame as game window")

        if self.game_mode == GameMode.PLAY:
            self._save_frame(frame, window_index, frame_index, "raw")
            logging.debug(f"{self.log_prefix} raw frame saved")

        hand_time = self._get_hand_time(frame, window_index, frame_index)
        total_pot = self._get_total_pot(frame, window_index, frame_index)
        seats = self._get_seats(frame, window_index, frame_index, hand_number)

        self.text_recognition.clear_frame_results()

        return {
            "hand_number": hand_number,
            "hand_time": hand_time,
            "total_pot": total_pot,
            "seats": seats,
        }

    def _process_objects(
        self, frame: np.ndarray, window_index: int, frame_index: int, hand_number: int
    ) -> ObjectData:
        dealer_position = self._get_dealer_position(
            frame, window_index, frame_index, hand_number
        )
        playing_seats = self._get_playing_seats(
            frame, window_index, frame_index, hand_number
        )
        table_cards = self._get_table_cards(
            frame, window_index, frame_index, hand_number
        )

        return {
            "dealer_position": dealer_position,
            "playing_seats": playing_seats,
            "table_cards": table_cards,
        }

    def _print_frame_info(self, text_data: TextData, object_data: ObjectData):
        seat_lines: List[str] = []
        for i, s in enumerate(text_data["seats"]):
            playing = "✔" if object_data["playing_seats"][i] else " "
            dealer = "●" if object_data["dealer_position"] == i else " "
            number = s["number"] if s["number"] != -1 else "⨯"
            balance = f"{s['balance']:.2f}" if s["balance"] > 0 else " "
            stake = f"{s['stake']:.2f}" if s["stake"] > 0 else " "
            action = s["action"] if s["action"] else " "

            seat_lines.append(
                f"{' ':<26}{self.log_prefix[:-2]} {i} {dealer} seat {number}  "
                + f"playing: {playing}  "
                + f"balance: {balance: <5} "
                + f"stake: {stake: <5} "
                + f"action: {action: <14}"
            )

        letter_to_suit = {
            "c": "♣",
            "d": "♦",
            "h": "♥",
            "s": "♠",
        }
        table_card_lines = ["—" for _ in range(self.object_detection.table_card_count)]
        for i, c in enumerate(object_data["table_cards"]):
            table_card_lines[i] = f"{c['rank']}{letter_to_suit[c['suit']]}"

        hand_time = text_data["hand_time"].strftime("%H:%M%z")
        if len(hand_time) > 5:
            hand_time = hand_time[:-2]

        logging.info(
            f"{self.log_prefix} hand number: {text_data['hand_number']} "
            + f"at {hand_time}\n"
            + f"{' ':<26}{self.log_prefix} table cards: "
            + " ".join(table_card_lines)
            + "\n"
            + f"{' ':<26}{self.log_prefix} total pot:   {text_data['total_pot']:.2f}\n"
            + "\n".join(seat_lines)
            + "\n"
        )

        logging.debug(
            f"{self.log_prefix} session data:\n" + f"{pformat(self.session, indent=4)}"
        )

    def _get_hand_number(
        self, frame: np.ndarray, window_index: int, frame_index: int
    ) -> int:
        region = self.object_detection.detect_hand_number(frame)
        if self._should_save_frame("hand_number"):
            self._save_frame(frame, window_index, frame_index, "hand_number", region)
        return self.text_recognition.recognize_hand_number(region)

    def _get_hand_time(
        self, frame: np.ndarray, window_index: int, frame_index: int
    ) -> datetime:
        region = self.object_detection.detect_hand_time(frame)
        if self._should_save_frame("hand_time"):
            self._save_frame(frame, window_index, frame_index, "hand_time", region)
        return self.text_recognition.recognize_hand_time(region)

    def _get_total_pot(
        self, frame: np.ndarray, window_index: int, frame_index: int
    ) -> float:
        region = self.object_detection.detect_total_pot(frame)
        if self._should_save_frame("total_pot"):
            self._save_frame(frame, window_index, frame_index, "total_pot", region)
        return self.text_recognition.recognize_total_pot(region)

    def _get_seats(
        self, frame: np.ndarray, window_index: int, frame_index: int, hand_number: int
    ) -> List[SeatData]:
        frame_data = self._get_latest_frame_data(hand_number)
        seats: List[SeatData] = []

        for i in range(self.object_detection.seat_count):
            seat_data = (
                frame_data["seats"][i]
                if frame_data and i in frame_data["seats"]
                else None
            )

            if seat_data and seat_data["number"] >= 0:
                number = seat_data["number"]
            else:
                region = self.object_detection.detect_seat_number(frame, i)
                number = self.text_recognition.recognize_seat_number(region)
                if self._should_save_frame("seat_numbers"):
                    self._save_frame(
                        frame, window_index, frame_index, f"seat_number_{i}", region
                    )

            if seat_data and not seat_data["playing"]:
                seats.append(
                    {
                        "number": number,
                        "action": "",
                        "stake": 0,
                        "balance": 0,
                        "playing": False,
                    }
                )
                continue

            region = self.object_detection.detect_seat_action(frame, i)
            action = self.text_recognition.recognize_seat_action(region)
            if self._should_save_frame("seat_actions"):
                self._save_frame(
                    frame, window_index, frame_index, f"seat_action_{i}", region
                )

            region = self.object_detection.detect_seat_stake(frame, i)
            stake = self.text_recognition.recognize_seat_money(region)
            if self._should_save_frame("seat_stakes"):
                self._save_frame(
                    frame, window_index, frame_index, f"seat_stake_{i}", region
                )

            region = self.object_detection.detect_seat_balance(frame, i)
            balance = self.text_recognition.recognize_seat_money(region)
            if self._should_save_frame("seat_balances"):
                self._save_frame(
                    frame, window_index, frame_index, f"seat_balance_{i}", region
                )

            seats.append(
                {
                    "number": number,
                    "action": action,
                    "stake": stake,
                    "balance": balance,
                    "playing": True,
                }
            )

        return seats

    def _get_dealer_position(
        self, frame: np.ndarray, window_index: int, frame_index: int, hand_number: int
    ) -> int:
        frame_data = self._get_latest_frame_data(hand_number)
        if frame_data:
            return frame_data["dealer_position"]

        (h, w) = frame.shape[:2]
        seat_regions = self.object_detection.get_seat_regions(w, h)

        for i, r in enumerate(seat_regions):
            cropped_frame = self._crop_frame(frame, r)
            region = self.object_detection.detect_dealer(cropped_frame)
            if region:
                if self._should_save_frame("dealer"):
                    dealer_frame = self._highlight_frame_region(frame.copy(), r)
                    self._save_frame(dealer_frame, window_index, frame_index, "dealer")
                return i

        return -1

    def _get_playing_seats(
        self, frame: np.ndarray, window_index: int, frame_index: int, hand_number: int
    ) -> List[bool]:
        frame_data = self._get_latest_frame_data(hand_number)

        (h, w) = frame.shape[:2]
        seat_regions = self.object_detection.get_seat_regions(w, h)

        playing_seats: List[bool] = []
        for i, r in enumerate(seat_regions):
            if (
                frame_data
                and i in frame_data["seats"]
                and not frame_data["seats"][i]["playing"]
            ):
                playing_seats.append(False)
                continue

            cropped_frame = self._crop_frame(frame, r)
            region = self.object_detection.detect_pocket_cards(cropped_frame, i)
            if region:
                if self._should_save_frame("hand_cards"):
                    hand_cards_frame = self._highlight_frame_region(frame.copy(), r)
                    self._save_frame(
                        hand_cards_frame,
                        window_index,
                        frame_index,
                        f"hand_cards_{i}",
                    )
                playing_seats.append(True)
            else:
                playing_seats.append(False)

        return playing_seats

    def _get_table_cards(
        self, frame: np.ndarray, window_index: int, frame_index: int, hand_number: int
    ) -> List[CardData]:
        frame_data = self._get_latest_frame_data(hand_number)
        table_cards: List[CardData] = []

        for i in range(self.object_detection.table_card_count):
            if frame_data and i in frame_data["table_cards"]:
                cards_data = frame_data["table_cards"][i]
                table_cards.append(
                    {
                        "rank": cards_data["rank"],
                        "suit": cards_data["suit"],
                    }
                )
                continue

            region = self.object_detection.detect_table_card(frame, i)
            if self._should_save_frame("table_cards"):
                self._save_frame(
                    frame, window_index, frame_index, f"table_card_{i}", region
                )

            cropped_frame = self._crop_frame(frame, region)
            cards_str = self.image_classifier.classify(cropped_frame)
            if cards_str:
                table_cards.append(
                    {
                        "rank": cards_str[0],
                        "suit": cards_str[1],
                    }
                )

        return table_cards

    def _get_latest_frame_data(self, hand_number: int) -> Optional[SessionData]:
        if hand_number in self.session and self.session[hand_number]:
            return self.session[hand_number][-1]
        return None

    def _should_save_frame(self, region_name: str) -> bool:
        return region_name in self.save_regions or "all" in self.save_regions

    def _save_frame(
        self,
        frame: np.ndarray,
        window_index: int,
        frame_index: int,
        name: str,
        region: Optional[Region] = None,
    ) -> None:
        cropped_frame = self._crop_frame(frame, region) if region else frame
        saved = cv2.imwrite(
            os.path.join(
                self.stream_path,
                f"window{window_index}",
                f"{frame_index}_{name}_processed.{self.frame_format}",
            ),
            cropped_frame,
        )
        if not saved:
            raise FrameError(f"unable to save {name} frame")

    @staticmethod
    def _crop_frame(frame: np.ndarray, region: Region) -> np.ndarray:
        x1, x2 = region.start.x, region.end.x
        y1, y2 = region.start.y, region.end.y
        return frame[y1:y2, x1:x2]

    @staticmethod
    def _highlight_frame_region(frame: np.ndarray, region: Region) -> np.ndarray:
        color = (255, 255, 255)
        return cv2.rectangle(
            frame,
            (region.start.x, region.start.y),
            (region.end.x, region.end.y),
            color,
            2,
        )
