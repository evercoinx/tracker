import logging
import os
import re
from collections import defaultdict
from datetime import datetime
from glob import glob
from multiprocessing import Queue, current_process, synchronize
from pprint import pformat
from typing import Any, DefaultDict, List, Optional

import cv2
import numpy as np
from typing_extensions import TypedDict  # pytype: disable=not-supported-yet

from tracker.error import FrameError
from tracker.object_detection import ObjectDetection, Region
from tracker.text_recognition import TextRecognition


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


class StreamPlayer:
    """Plays a live stream or replays a saved one"""

    TOTAL_SEATS = 6

    queue: Queue
    events: List[synchronize.Event]
    stream_path: str
    frame_format: str
    text_recognition: TextRecognition
    object_detection: ObjectDetection
    log_prefix: str
    session: DefaultDict[int, List[Any]]

    def __init__(
        self,
        queue: Queue,
        events: List[synchronize.Event],
        stream_path: str,
        frame_format: str,
        text_recognition: TextRecognition,
        object_detection: ObjectDetection,
    ):
        self.queue = queue
        self.events = events
        self.stream_path = stream_path
        self.frame_format = frame_format
        self.text_recognition = text_recognition
        self.object_detection = object_detection
        self.log_prefix = ""
        self.session = defaultdict(list)

    def play(self) -> None:
        frame_index = 0

        while True:
            try:
                window_index, frame = self.queue.get()
                self.log_prefix = self.get_log_prefix(window_index, frame_index)
                self.process_frame(frame, window_index, frame_index)

                frame_index += 1
                self.events[window_index].set()

            except (KeyboardInterrupt, SystemExit):
                self.queue.close()
                logging.warn(f"{self.log_prefix} interruption; exiting...")
                return

    def replay(self, windows: List[int]) -> None:
        raw_frame_path_pattern = re.compile(
            r"window(["
            + re.escape(",".join(windows))
            + r"])\/(\d+)_raw."
            + re.escape(self.frame_format)
            + r"$"
        )
        raw_frame_paths = glob(
            f"{self.stream_path}/window[{''.join(windows)}]/*_raw.{self.frame_format}",
            recursive=True,
        )

        for p in sorted(raw_frame_paths):
            frame = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if frame is None:
                raise FrameError(f"unable to read frame path {p}", -1, -1, "raw")

            matches = re.findall(raw_frame_path_pattern, p)
            if not matches:
                raise FrameError(f"unable to parse frame path {p}", -1, -1, "raw")

            (window_index, frame_index) = matches[0]
            self.log_prefix = self.get_log_prefix(window_index, frame_index)
            self.process_frame(frame, int(window_index), int(frame_index))

        logging.debug(
            f"{self.log_prefix} current session dump:\n{pformat(self.session)}"
        )

    def process_frame(
        self, frame: np.ndarray, window_index: int, frame_index: int
    ) -> None:
        inverted_frame = cv2.bitwise_not(frame)
        if self.is_debug():
            (h, w) = frame.shape[:2]
            for r in self.object_detection.get_player_regions(w, h):
                cv2.rectangle(
                    inverted_frame,
                    (r.start.x, r.start.y),
                    (r.end.x, r.end.y),
                    (255, 255, 255),
                    1,
                )
            self.save_frame(inverted_frame, window_index, frame_index, "full")

        # TODO Implement automatic text contour detection
        # self.save_text_contours(invereted_frame, window_index, frame_index)

        try:
            text_data = self.process_texts(inverted_frame, window_index, frame_index)
            object_data = self.process_objects(
                inverted_frame, window_index, frame_index
            )
        except FrameError as err:
            logging.warn(f"{self.log_prefix} {err}")
            return

        if len(text_data["seats"]) != len(object_data["playing_seats"]):
            logging.warn(
                "invalid number of seats detected: "
                + f"{text_data['seats']} != {object_data['playing_seats']}"
            )
            return

        indent = " " * 26
        seat_data: List[SeatData] = []

        for i, s in enumerate(text_data["seats"]):
            if s is None:
                seat_data.append(f"{indent}{self.log_prefix} {i}: seat is out")
                continue

            playing = "yes" if object_data["playing_seats"][i] else "no"
            seat_data.append(
                f"{indent}{self.log_prefix} {i}: seat {s['number']}, "
                + f"playing: {playing}, "
                + f"balance: {s['balance']:.2f}, "
                + f"stake: {s['stake']:.2f}, "
                + f"action: {s['action']}"
            )

        logging.info(
            f"{self.log_prefix} hand number: {text_data['hand_number']} "
            + f"at {text_data['hand_time'].strftime('%H:%M%z')}\n"
            + f"{indent}{self.log_prefix} total pot: {text_data['total_pot']:.2f}, "
            + f"dealer position: {object_data['dealer_position']}\n"
            + "\n".join(seat_data)
            + "\n"
        )

        self.session[text_data["hand_number"]].append(
            {
                "window_index": window_index,
                "frame_index": frame_index,
                "hand_time": text_data["hand_time"],
                "total_pot": text_data["total_pot"],
                "dealer_position": object_data["dealer_position"],
                "seats": text_data["seats"],
            }
        )

    def process_texts(
        self, frame: np.ndarray, window_index: int, frame_index: int
    ) -> TextData:
        self.text_recognition.set_frame(frame)

        hand_number = self.get_hand_number(frame, window_index, frame_index)
        if not hand_number:
            self.remove_frame(window_index, frame_index, "raw")
            raise FrameError(
                "unable to recognize frame", window_index, frame_index, "raw"
            )

        hand_time = self.get_hand_time(frame, window_index, frame_index)
        total_pot = self.get_total_pot(frame, window_index, frame_index)
        seats = self.get_seats(frame, window_index, frame_index)

        self.text_recognition.clear_frame_results()

        # pytype: disable=bad-return-type
        return {
            "hand_number": hand_number,
            "hand_time": hand_time,
            "total_pot": total_pot,
            "seats": seats,
        }
        # pytype: enable=bad-return-type

    def process_objects(
        self, frame: np.ndarray, window_index: int, frame_index: int
    ) -> ObjectData:
        dealer_position = self.get_dealer_position(frame, window_index, frame_index)
        playing_seats = self.get_playing_seats(frame, window_index, frame_index)

        # pytype: disable=bad-return-type
        return {
            "dealer_position": dealer_position,
            "playing_seats": playing_seats,
        }
        # pytype: enable=bad-return-type

    def get_hand_number(
        self, frame: np.ndarray, window_index: int, frame_index: int
    ) -> int:
        region = self.object_detection.detect_hand_number(frame)
        if self.is_debug():
            self.save_frame(frame, window_index, frame_index, "hand_number", region)

        return self.text_recognition.recognize_hand_number(region)

    def get_hand_time(
        self, frame: np.ndarray, window_index: int, frame_index: int
    ) -> datetime:
        region = self.object_detection.detect_hand_time(frame)
        if self.is_debug():
            self.save_frame(frame, window_index, frame_index, "hand_time", region)

        return self.text_recognition.recognize_hand_time(region)

    def get_total_pot(
        self, frame: np.ndarray, window_index: int, frame_index: int
    ) -> float:
        region = self.object_detection.detect_total_pot(frame)
        if self.is_debug():
            self.save_frame(frame, window_index, frame_index, "total_pot", region)

        return self.text_recognition.recognize_total_pot(region)

    def get_seats(
        self, frame: np.ndarray, window_index: int, frame_index: int
    ) -> List[Optional[SeatData]]:
        seats: List[Optional[SeatData]] = []

        for i in range(StreamPlayer.TOTAL_SEATS):
            region = self.object_detection.detect_seat_number(frame, i)
            number = self.text_recognition.recognize_seat_number(region)
            if self.is_debug():
                self.save_frame(
                    frame, window_index, frame_index, f"seat_number_{i}", region
                )

            if not number:
                seats.append(None)
                continue

            region = self.object_detection.detect_seat_action(frame, i)
            action = self.text_recognition.recognize_seat_action(region)
            if self.is_debug():
                self.save_frame(
                    frame, window_index, frame_index, f"seat_action_{i}", region
                )

            region = self.object_detection.detect_seat_stake(frame, i)
            stake = self.text_recognition.recognize_seat_money(region)
            if self.is_debug():
                self.save_frame(
                    frame, window_index, frame_index, f"seat_stake_{i}", region
                )

            region = self.object_detection.detect_seat_balance(frame, i)
            balance = self.text_recognition.recognize_seat_money(region)
            if self.is_debug():
                self.save_frame(
                    frame, window_index, frame_index, f"seat_balance_{i}", region
                )

            seats.append(
                {
                    "number": number,
                    "action": action,
                    "stake": stake,
                    "balance": balance,
                }
            )

        return seats

    def get_dealer_position(
        self, frame: np.ndarray, window_index: int, frame_index: int
    ) -> int:
        (h, w) = frame.shape[:2]
        player_regions = self.object_detection.get_player_regions(w, h)

        for i, r in enumerate(player_regions):
            roi = self.crop_frame(frame, r)
            region = self.object_detection.detect_dealer(roi)
            if region is not None:
                if self.is_debug():
                    dealer_frame = self.highlight_frame_region(frame.copy(), r)
                    self.save_frame(dealer_frame, window_index, frame_index, "dealer")
                return i

        return -1

    def get_playing_seats(
        self, frame: np.ndarray, window_index: int, frame_index: int
    ) -> List[bool]:
        (h, w) = frame.shape[:2]
        player_regions = self.object_detection.get_player_regions(w, h)

        playing_seats: List[bool] = []
        for i, r in enumerate(player_regions):
            roi = self.crop_frame(frame, r)
            region = self.object_detection.detect_hand_card(roi, i)
            if region is not None:
                if self.is_debug():
                    playing_seats_frame = self.highlight_frame_region(frame.copy(), r)
                    self.save_frame(
                        playing_seats_frame,
                        window_index,
                        frame_index,
                        f"hand_cards_{i}",
                    )
                playing_seats.append(True)
            else:
                playing_seats.append(False)

        return playing_seats

    def save_text_contours(
        self, frame: np.ndarray, window_index: int, frame_index: int
    ) -> None:
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
        sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        gray = cv2.GaussianBlur(frame, (3, 3), 0)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

        gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = np.min(gradX), np.max(gradX)
        gradX = (gradX - minVal) / (maxVal - minVal)
        gradX = (gradX * 255).astype("uint8")

        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
        thresh = cv2.threshold(gradX, 40, 255, cv2.THRESH_BINARY)[1]

        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
        thresh = cv2.erode(thresh, None, iterations=2)

        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        for i, c in enumerate(cnts):
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
            crWidth = w / float(gray.shape[1])

            if ar > 0.5 and crWidth > 0.01:
                pad_x = int((x + w) * 0.03)
                pad_y = int((y + h) * 0.03)

                (x, y) = (x - pad_x, y - pad_y)
                (w, h) = (w + (pad_x * 2), h + (pad_y * 2))

                y2 = y + h
                x2 = x + w
                roi = frame[y:y2, x:x2].copy()

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)
                if self.is_debug():
                    self.save_frame(roi, window_index, frame_index, f"contour_{i}")

        self.save_frame(frame, window_index, frame_index, "countours")

    def save_frame(
        self,
        frame: np.ndarray,
        window_index: int,
        frame_index: int,
        name: str,
        region: Optional[Region] = None,
    ) -> None:
        roi = self.crop_frame(frame, region) if region else frame
        saved = cv2.imwrite(
            os.path.join(
                self.stream_path,
                f"window{window_index}",
                f"{frame_index}_{name}_processed.{self.frame_format}",
            ),
            roi,
        )
        if not saved:
            raise FrameError(
                "unable to save processed frame", window_index, frame_index, name
            )

    def remove_frame(self, window_index: int, frame_index: int, name: str) -> None:
        try:
            os.remove(
                os.path.join(
                    self.stream_path,
                    f"window{window_index}",
                    f"{frame_index}_{name}.{self.frame_format}",
                ),
            )
        except OSError:
            raise FrameError("unable to remove frame", window_index, frame_index, name)

    @staticmethod
    def crop_frame(frame: np.ndarray, region: Region) -> np.ndarray:
        x1, x2 = region.start.x, region.end.x
        y1, y2 = region.start.y, region.end.y
        return frame[y1:y2, x1:x2]

    @staticmethod
    def highlight_frame_region(frame: np.ndarray, region: Region) -> np.ndarray:
        color = (255, 255, 255)
        return cv2.rectangle(
            frame,
            (region.start.x, region.start.y),
            (region.end.x, region.end.y),
            color,
            2,
        )

    @staticmethod
    def is_debug() -> bool:
        return logging.root.level == logging.DEBUG

    @staticmethod
    def get_log_prefix(window_index: int, frame_index: int) -> str:
        proc_name = current_process().name
        if proc_name == "MainProcess":  # no multiprocessing
            proc_name = "player"
        return f"{proc_name}-w{window_index}-f{frame_index:<5} -"
