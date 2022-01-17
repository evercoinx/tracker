import logging
import os
import re
import time
from collections import defaultdict
from datetime import datetime
from enum import Enum
from glob import glob
from multiprocessing import Queue, current_process
from multiprocessing.synchronize import Event
from typing import (
    Callable,
    ClassVar,
    DefaultDict,
    Dict,
    FrozenSet,
    List,
    Optional,
    Tuple,
)

import cv2
import grpc
import numpy as np
from typing_extensions import TypedDict

import tracker.proto.analyzer_pb2 as pbanalyzer
from tracker.image_classifier import ImageClassifier
from tracker.object_detection import ObjectDetection, Region
from tracker.proto.analyzer_pb2_grpc import AnalyzerStub
from tracker.text_recognition import Currency, Money, TextRecognition


class Card:
    ranks: ClassVar[FrozenSet[str]] = frozenset(
        ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
    )
    suit_to_symbol: ClassVar[Dict[str, str]] = {
        "c": "♣",
        "d": "♦",
        "h": "♥",
        "s": "♠",
    }

    rank: str
    suit: str

    def __init__(self, rank: str, suit: str):
        if rank not in type(self).ranks:
            raise ValueError(f"Unexpected rank: {rank}")
        if suit not in type(self).suit_to_symbol:
            raise ValueError(f"Unexpected suit: {suit}")

        self.rank = rank
        self.suit = suit

    def __str__(self) -> str:
        return f"{self.rank}{self.suit_to_symbol[self.suit]}"

    def __repr__(self) -> str:
        return f"Card('{self.rank}', '{self.suit}')"


class SeatData(TypedDict):
    number: int
    action: str
    stake: Money
    balance: Money
    playing: bool


class FrameData(TypedDict):
    window_index: int
    frame_index: int
    hand_time: datetime
    total_pot: Money
    dealer_position: int
    seats: List[SeatData]
    board: List[Card]


class GameMode(Enum):
    UNSET = 0
    PLAY = 1
    REPLAY = 2


def elapsed_time(name: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            ret = func(*args, **kwargs)
            elapsed_time = time.perf_counter() - start_time
            logging.debug(
                f"{'Performance':<14} - {name.capitalize() + ':': <16} "
                + f"{elapsed_time:.3f}s"
            )
            return ret

        return wrapper

    return decorator


class StreamPlayer:
    """Plays a live stream or replays a saved one into the console window"""

    game_mode: GameMode
    stream_path: str
    frame_format: str
    save_regions: List[str]
    text_recognition: TextRecognition
    object_detection: ObjectDetection
    image_classifier: ImageClassifier
    log_prefix: str
    session: DefaultDict[int, List[FrameData]]
    queue: Optional[Queue]
    events: List[Event]
    replay_windows: List[str]
    analyzer_address: str
    analyzer_stub: AnalyzerStub

    def __init__(
        self,
        game_mode: GameMode,
        stream_path: str,
        frame_format: str,
        save_regions: List[str],
        text_recognition: TextRecognition,
        object_detection: ObjectDetection,
        image_classifier: ImageClassifier,
        analyzer_address: str,
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

        chan = grpc.insecure_channel(analyzer_address)
        self.analyzer_stub = AnalyzerStub(chan)

    def run(self) -> None:
        if self.game_mode == GameMode.PLAY:
            self._play()
        elif self.game_mode == GameMode.REPLAY:
            self._replay()
        else:
            raise ValueError(f"Invalid game mode: {self.game_mode}")

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
                raise OSError(f"Unable to read frame at {p}")

            matches = re.findall(raw_frame_path_pattern, p)
            (window_index, frame_index) = matches[0]

            self.log_prefix = self._get_log_prefix(window_index, frame_index)
            self._process_frame(frame, int(window_index), int(frame_index))

    @staticmethod
    def _get_log_prefix(window_index: int, frame_index: int) -> str:
        proc_name = current_process().name
        if proc_name == "MainProcess":  # no multiprocessing
            proc_name = "Game"
        return f"{proc_name}-W{window_index}-F{frame_index:<5} -"

    @staticmethod
    def _sort_path(pattern: re.Pattern) -> Callable[[str], Tuple[int, int]]:
        def match(path: str) -> Tuple[int, int]:
            matches = re.findall(pattern, path)
            if not matches:
                raise Exception(f"Unable to parse frame at {path}")
            return (
                int(matches[0][0]),
                int(matches[0][1]),
            )

        return match

    def _process_frame(
        self, frame: np.ndarray, window_index: int, frame_index: int
    ) -> None:
        frame = cv2.bitwise_not(frame)

        if self._should_save_frame("full"):
            full_frame = frame.copy()
            (h, w) = frame.shape[:2]
            for r in self.object_detection.get_seat_regions(w, h):
                full_frame = self._highlight_frame_region(full_frame, r)
            self._save_frame(full_frame, window_index, frame_index, "full")

        self.text_recognition.set_image(frame)

        hand_number = self._get_hand_number(frame, window_index, frame_index)
        if not hand_number:
            logging.warn(
                f"{self.log_prefix} Unable to recognize frame as game window\n"
            )
            return

        if self.game_mode == GameMode.PLAY:
            self._save_frame(frame, window_index, frame_index, "raw")
            logging.debug(f"{self.log_prefix} raw frame saved")

        hand_time = self._get_hand_time(frame, window_index, frame_index)
        dealer_position = self._get_dealer_position(
            frame, window_index, frame_index, hand_number
        )

        board = self._get_board(frame, window_index, frame_index, hand_number)
        total_pot = self._get_total_pot(frame, window_index, frame_index)

        playing_seats = self._get_playing_seats(
            frame, window_index, frame_index, hand_number
        )
        seats = self._get_seats(
            frame, window_index, frame_index, hand_number, playing_seats
        )

        self.text_recognition.clear_image_results()

        frame_data: FrameData = {
            "window_index": window_index,
            "frame_index": frame_index,
            "hand_time": hand_time,
            "total_pot": total_pot,
            "dealer_position": dealer_position,
            "seats": seats,
            "board": board,
        }
        self.session[hand_number].append(frame_data)
        self._print_frame_data(frame_data, hand_number)

        req = pbanalyzer.FrameRequest(
            window_index=window_index,
            frame_index=frame_index,
            hand_number=hand_number,
            hand_time=hand_time.strftime("%H:%M%z"),
            total_pot=self._to_pb_money(total_pot),
            dealer_position=dealer_position,
            seats=[self._to_pb_seat(s) for s in seats],
            board=[f"{c.rank}{c.suit}" for c in board],
        )
        self.analyzer_stub.SendFrame(req)

    def _to_pb_seat(self, seat: SeatData) -> pbanalyzer.Seat:
        return pbanalyzer.Seat(
            number=seat["number"],
            action=seat["action"],
            stake=self._to_pb_money(seat["stake"]),
            balance=self._to_pb_money(seat["balance"]),
            playing=seat["playing"],
        )

    def _to_pb_money(self, money: Money) -> pbanalyzer.Money:
        if money.currency == Currency.EURO:
            currency = pbanalyzer.Money.Currency.EURO
        elif money.currency == Currency.DOLLAR:
            currency = pbanalyzer.Money.Currency.DOLLAR
        else:
            currency = pbanalyzer.Money.Currency.UNSET

        return pbanalyzer.Money(
            currency=currency,
            amount=money.amount,
        )

    def _print_frame_data(self, frame_data: FrameData, hand_number: int):
        seat_lines: List[str] = []
        for i, s in enumerate(frame_data["seats"]):
            playing = "✔" if s["playing"] else " "
            dealer = "●" if frame_data["dealer_position"] == i else " "
            number = str(s["number"]) if s["number"] else "⨯"
            balance = str(s["balance"]) if (s["balance"] or s["playing"]) else " " * 6
            stake = str(s["stake"]) if s["stake"] else " " * 6
            action = s["action"] if s["action"] else " "

            seat_lines.append(
                f"{' ':<26}{self.log_prefix[:-2]} {i} {dealer} Seat {number}:  "
                + f"playing {playing}  "
                + f"balance {balance} "
                + f"stake {stake} "
                + f"action {action: <14}"
            )

        board_lines = ["—" for _ in range(5)]
        for i, c in enumerate(frame_data["board"]):
            board_lines[i] = str(c)

        hand_time = frame_data["hand_time"].strftime("%H:%M%z")
        if len(hand_time) > 5:
            hand_time = hand_time[:-2]

        logging.info(
            f"{self.log_prefix} Hand number #{hand_number} "
            + f"at {hand_time}\n"
            + f"{' ':<26}{self.log_prefix} Total pot {frame_data['total_pot']}\n"
            + f"{' ':<26}{self.log_prefix} Board     "
            + " ".join(board_lines)
            + "\n"
            + "\n".join(seat_lines)
            + "\n"
        )

    @elapsed_time("hand number")
    def _get_hand_number(
        self, frame: np.ndarray, window_index: int, frame_index: int
    ) -> int:
        region = self.object_detection.detect_hand_number(frame)
        if self._should_save_frame("hand_number"):
            self._save_frame(frame, window_index, frame_index, "hand_number", region)
        return self.text_recognition.recognize_hand_number(region)

    @elapsed_time("hand time")
    def _get_hand_time(
        self, frame: np.ndarray, window_index: int, frame_index: int
    ) -> datetime:
        region = self.object_detection.detect_hand_time(frame)
        if self._should_save_frame("hand_time"):
            self._save_frame(frame, window_index, frame_index, "hand_time", region)
        return self.text_recognition.recognize_hand_time(region)

    @elapsed_time("total pot")
    def _get_total_pot(
        self, frame: np.ndarray, window_index: int, frame_index: int
    ) -> Money:
        region = self.object_detection.detect_total_pot(frame)
        if self._should_save_frame("total_pot"):
            self._save_frame(frame, window_index, frame_index, "total_pot", region)
        return self.text_recognition.recognize_total_pot(region)

    @elapsed_time("seats")
    def _get_seats(
        self,
        frame: np.ndarray,
        window_index: int,
        frame_index: int,
        hand_number: int,
        playing_seats: List[bool],
    ) -> List[SeatData]:
        frame_data = self._get_last_frame_data(hand_number)
        seats: List[SeatData] = []

        for i in range(6):
            seat_data = (
                frame_data["seats"][i]
                if frame_data and i < len(frame_data["seats"])
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

            if seat_data and not playing_seats[i]:
                seats.append(
                    {
                        "number": number,
                        "action": "",
                        "stake": Money(),
                        "balance": seat_data["balance"],
                        "playing": playing_seats[i],
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
                    "playing": playing_seats[i],
                }
            )

        return seats

    @elapsed_time("dealer position")
    def _get_dealer_position(
        self, frame: np.ndarray, window_index: int, frame_index: int, hand_number: int
    ) -> int:
        frame_data = self._get_last_frame_data(hand_number)
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

    @elapsed_time("playing seats")
    def _get_playing_seats(
        self, frame: np.ndarray, window_index: int, frame_index: int, hand_number: int
    ) -> List[bool]:
        frame_data = self._get_last_frame_data(hand_number)

        (h, w) = frame.shape[:2]
        seat_regions = self.object_detection.get_seat_regions(w, h)

        playing_seats: List[bool] = []
        for i, r in enumerate(seat_regions):
            if (
                frame_data
                and i < len(frame_data["seats"])
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

    @elapsed_time("board")
    def _get_board(
        self, frame: np.ndarray, window_index: int, frame_index: int, hand_number: int
    ) -> List[Card]:
        frame_data = self._get_last_frame_data(hand_number)
        board: List[Card] = []

        for i in range(5):
            if frame_data and i < len(frame_data["board"]):
                card = frame_data["board"][i]
                board.append(card)
                continue

            region = self.object_detection.detect_table_card(frame, i)
            if self._should_save_frame("board"):
                self._save_frame(frame, window_index, frame_index, f"board_{i}", region)

            cropped_frame = self._crop_frame(frame, region)
            card_str = self.image_classifier.classify(cropped_frame)
            if card_str:
                c = Card(card_str[0], card_str[1])
                board.append(c)

        return board

    def _get_last_frame_data(self, hand_number: int) -> Optional[FrameData]:
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
        path = os.path.join(
            self.stream_path,
            f"window{window_index}",
            f"{frame_index}_{name}_processed.{self.frame_format}",
        )
        cropped_frame = self._crop_frame(frame, region) if region else frame

        saved = cv2.imwrite(path, cropped_frame)
        if not saved:
            raise OSError(f"Unable to save frame at path {path}")

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
