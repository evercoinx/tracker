import logging
import os
import re
from collections import defaultdict
from functools import reduce
from glob import glob
from multiprocessing import current_process
from pprint import pformat

import cv2

from tracker.error import FrameError
from tracker.utils import Dimensions, Point


class StreamPlayer:
    """Plays a live stream or replays a saved one"""

    def __init__(
        self,
        queue,
        events,
        stream_path,
        frame_format,
        text_recognition,
        object_detection,
    ):
        self.queue = queue
        self.events = events
        self.stream_path = stream_path
        self.frame_format = frame_format
        self.text_recognition = text_recognition
        self.object_detection = object_detection
        self.log_prefix = ""
        self.session = defaultdict(list)

    def play(self):
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

    def replay(self, windows):
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

        for path in sorted(raw_frame_paths):
            frame = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if frame is None:
                raise FrameError(f"frame {path} is not found", -1, -1, "raw")

            matches = re.findall(raw_frame_path_pattern, path)
            if not matches:
                raise FrameError(f"unable to parse frame path {path}", -1, -1, "raw")

            (window_index, frame_index) = matches[0]
            self.log_prefix = self.get_log_prefix(window_index, frame_index)
            self.process_frame(frame, int(window_index), int(frame_index))

        logging.debug(
            f"{self.log_prefix} current session dump:\n{pformat(self.session)}"
        )

    def process_frame(self, frame, window_index, frame_index):
        inverted_frame = cv2.bitwise_not(frame)
        if self.is_debug():
            self.save_frame(inverted_frame, window_index, frame_index, "full")

        text_data = self.process_texts(inverted_frame, window_index, frame_index)
        if not text_data:
            logging.warn(f"{self.log_prefix} unable to process texts on frame")
            return

        object_data = self.process_objects(inverted_frame, window_index, frame_index)
        if not object_data:
            logging.warn(f"{self.log_prefix} unable to process objects on frame")
            return

        indent = " " * 26
        seat_data = []
        for seat in text_data["seats"]:
            seat_data.append(
                indent
                + f"{self.log_prefix} seat {seat['number']}, "
                + f"balance: {seat['balance']:.2f}, "
                + f"stake: {seat['stake']:.2f}, "
                + f"action: {seat['action']}"
            )

        logging.info(
            f"{self.log_prefix} hand number: {text_data['hand_number']} "
            + f"at {text_data['hand_time'].strftime('%H:%M%z')}\n"
            + indent
            + f"{self.log_prefix} total pot: {text_data['total_pot']:.2f}, "
            + f"total stakes: {text_data['total_stakes']:.2f}, "
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
                "total_stakes": text_data["total_stakes"],
                "dealer_position": object_data["dealer_position"],
                "seats": text_data["seats"],
            }
        )

    def process_texts(self, frame, window_index, frame_index):
        self.text_recognition.set_frame(frame)

        hand_number = self.recognize_hand_number(frame, window_index, frame_index)
        if not hand_number:
            self.remove_frame(window_index, frame_index, "raw")
            return

        hand_time = self.recognize_hand_time(frame, window_index, frame_index)

        total_pot = self.recognize_total_pot(frame, window_index, frame_index)
        seats = self.recognize_seats(frame, window_index, frame_index)
        total_stakes = reduce(lambda accum, seat: accum + seat["stake"], seats, 0)

        self.text_recognition.clear_current_frame()

        return {
            "hand_number": hand_number,
            "hand_time": hand_time,
            "seats": seats,
            "total_pot": total_pot,
            "total_stakes": total_stakes,
        }

    def process_objects(self, frame, window_index, frame_index):
        dealer_position = self.recognize_dealer(frame, window_index, frame_index)

        return {
            "dealer_position": dealer_position,
        }

    def recognize_hand_number(self, frame, window_index, frame_index):
        point = Point(73, 24)
        dims = Dimensions(101, 15)
        hand_number = self.text_recognition.get_hand_number(point, dims)

        if self.is_debug():
            self.save_frame(
                frame,
                window_index,
                frame_index,
                "hand_number",
                point=point,
                dimensions=dims,
            )

        return hand_number

    def recognize_hand_time(self, frame, window_index, frame_index):
        point = Point(857, 22)
        dims = Dimensions(55, 14)
        hand_time = self.text_recognition.get_hand_time(point, dims)

        if self.is_debug():
            self.save_frame(
                frame,
                window_index,
                frame_index,
                "hand_time",
                point=point,
                dimensions=dims,
            )

        return hand_time

    def recognize_total_pot(self, frame, window_index, frame_index):
        point = Point(462, 160)
        dims = Dimensions(91, 21)
        total_pot = self.text_recognition.get_total_pot(point, dims)

        if self.is_debug():
            self.save_frame(
                frame,
                window_index,
                frame_index,
                "total_pot",
                point=point,
                dimensions=dims,
            )

        return total_pot

    def recognize_seats(self, frame, window_index, frame_index):
        action_points = [
            Point(138, 321),
            Point(172, 100),
            Point(433, 68),
            Point(664, 100),
            Point(682, 321),
            Point(431, 328),
        ]
        action_dims = Dimensions(119, 14)

        number_points = [
            Point(138, 334),
            Point(172, 113),
            Point(433, 81),
            Point(664, 113),
            Point(682, 334),
            Point(431, 342),
        ]
        number_dims = Dimensions(119, 15)

        balance_points = [
            Point(138, 351),
            Point(172, 130),
            Point(433, 98),
            Point(664, 130),
            Point(682, 351),
            Point(431, 357),
        ]
        balance_dims = Dimensions(119, 16)

        stake_points = [
            Point(287, 288),
            Point(294, 154),
            Point(423, 131),
            Point(602, 153),
            Point(595, 290),
            Point(0, 0),
        ]
        stake_dims = Dimensions(56, 19)

        seats = []

        for i in range(len(number_points)):
            # last player is a hero
            if i == len(number_points) - 1:
                number = 9
            else:
                number = self.text_recognition.get_seat_number(
                    number_points[i], number_dims
                )

            if self.is_debug():
                self.save_frame(
                    frame,
                    window_index,
                    frame_index,
                    f"seat_number_{i}",
                    point=number_points[i],
                    dimensions=number_dims,
                )

            # if we failed to detect a seat number it is unreasonable to look for
            # a balance and an action of this seat
            if not number:
                continue

            action = self.text_recognition.get_seat_action(
                action_points[i], action_dims
            )
            if self.is_debug():
                self.save_frame(
                    frame,
                    window_index,
                    frame_index,
                    f"seat_action_{i}",
                    point=action_points[i],
                    dimensions=action_dims,
                )

            stake = self.text_recognition.get_seat_money(stake_points[i], stake_dims)
            if self.is_debug():
                self.save_frame(
                    frame,
                    window_index,
                    frame_index,
                    f"seat_stake_{i}",
                    point=stake_points[i],
                    dimensions=stake_dims,
                )

            balance = self.text_recognition.get_seat_money(
                balance_points[i], balance_dims
            )
            if self.is_debug():
                self.save_frame(
                    frame,
                    window_index,
                    frame_index,
                    f"seat_balance_{i}",
                    point=balance_points[i],
                    dimensions=balance_dims,
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

    def recognize_dealer(self, frame, window_index, frame_index):
        (start_x, start_y, end_x, end_y) = self.object_detection.get_dealer_coords(
            frame
        )

        if self.is_debug():
            dealer_frame = cv2.rectangle(
                frame.copy(),
                (start_x, start_y),
                (end_x, end_y),
                (255, 255, 255),
                2,
            )
            self.save_frame(dealer_frame, window_index, frame_index, "dealer")

        (h, w) = frame.shape[:2]
        return self.object_detection.point_in_region(
            Point(end_x, end_y),
            Dimensions(w, h),
            width_parts=3,
            height_parts=2,
        )

    def save_frame(
        self, frame, window_index, frame_index, name, *, point=None, dimensions=None
    ):
        roi = frame
        if point and dimensions:
            x1, x2 = point.x, point.x + dimensions.width
            y1, y2 = point.y, point.y + dimensions.height
            roi = frame[y1:y2, x1:x2]

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

    def remove_frame(self, window_index, frame_index, name):
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
    def is_debug():
        return logging.root.level == logging.DEBUG

    @staticmethod
    def get_log_prefix(window_index, frame_index):
        proc_name = current_process().name
        if proc_name == "MainProcess":  # no multiprocessing
            proc_name = "player"
        return f"{proc_name}-w{window_index}-f{frame_index:<5} -"
