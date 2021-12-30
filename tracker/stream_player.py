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

        logging.info(f"{self.log_prefix} {'-' * 60}")

        hand_time = self.recognize_hand_time(frame, window_index, frame_index)
        logging.info(
            f"{self.log_prefix} hand number: {hand_number} "
            + f"at {hand_time.strftime('%H:%M%z')}"
        )

        total_pot = self.recognize_total_pot(frame, window_index, frame_index)
        seats = self.recognize_seats(frame, window_index, frame_index)

        self.text_recognition.clear_current_frame()

        total_stakes = reduce(lambda accum, seat: accum + seat["stake"], seats, 0)
        logging.info(
            f"{self.log_prefix} total pot: {total_pot:.2f}, "
            + f"total stakes: {total_stakes:.2f}"
        )

        for seat in seats:
            logging.info(
                f"{self.log_prefix} seat {seat['number']}, "
                + f"balance: {seat['balance']:.2f}, "
                + f"stake: {seat['stake']:.2f}, "
                + f"action: {seat['action']}"
            )

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
        coords = (73, 24)
        dims = (101, 15)
        hand_number = self.text_recognition.get_hand_number(coords, dims)

        if self.is_debug():
            self.save_frame(
                frame,
                window_index,
                frame_index,
                "hand_number",
                coords=coords,
                dims=dims,
            )

        return hand_number

    def recognize_hand_time(self, frame, window_index, frame_index):
        coords = (857, 22)
        dims = (55, 14)
        hand_time = self.text_recognition.get_hand_time(coords, dims)

        if self.is_debug():
            self.save_frame(
                frame,
                window_index,
                frame_index,
                "hand_time",
                coords=coords,
                dims=dims,
            )

        return hand_time

    def recognize_total_pot(self, frame, window_index, frame_index):
        coords = (462, 160)
        dims = (91, 21)
        total_pot = self.text_recognition.get_total_pot(coords, dims)

        if self.is_debug():
            self.save_frame(
                frame,
                window_index,
                frame_index,
                "total_pot",
                coords=coords,
                dims=dims,
            )

        return total_pot

    def recognize_seats(self, frame, window_index, frame_index):
        action_coords_groups = [
            (138, 321),
            (172, 100),
            (433, 68),
            (664, 100),
            (682, 321),
            (431, 328),
        ]
        action_dims = (119, 14)

        number_coords_groups = [
            (138, 334),
            (172, 113),
            (433, 81),
            (664, 113),
            (682, 334),
            (431, 342),
        ]
        number_dims = (119, 15)

        balance_coords_groups = [
            (138, 351),
            (172, 130),
            (433, 98),
            (664, 130),
            (682, 351),
            (431, 357),
        ]
        balance_dims = (119, 16)

        stake_coords_groups = [
            (287, 288),
            (294, 154),
            (423, 131),
            (602, 153),
            (595, 290),
            (0, 0),
        ]
        stake_dims = (56, 19)

        seats = []

        for i in range(len(number_coords_groups)):
            # last player is a hero
            if i == len(number_coords_groups) - 1:
                number = 9
            else:
                number = self.text_recognition.get_seat_number(
                    number_coords_groups[i], number_dims
                )

            if self.is_debug():
                self.save_frame(
                    frame,
                    window_index,
                    frame_index,
                    f"seat_number_{i}",
                    coords=number_coords_groups[i],
                    dims=number_dims,
                )

            # if we failed to detect a seat number it is unreasonable to look for
            # a balance and an action of this seat
            if not number:
                continue

            action = self.text_recognition.get_seat_action(
                action_coords_groups[i], action_dims
            )
            if self.is_debug():
                self.save_frame(
                    frame,
                    window_index,
                    frame_index,
                    f"seat_action_{i}",
                    coords=action_coords_groups[i],
                    dims=action_dims,
                )

            stake = self.text_recognition.get_seat_money(
                stake_coords_groups[i], stake_dims
            )
            if self.is_debug():
                self.save_frame(
                    frame,
                    window_index,
                    frame_index,
                    f"seat_stake_{i}",
                    coords=stake_coords_groups[i],
                    dims=stake_dims,
                )

            balance = self.text_recognition.get_seat_money(
                balance_coords_groups[i], balance_dims
            )
            if self.is_debug():
                self.save_frame(
                    frame,
                    window_index,
                    frame_index,
                    f"seat_balance_{i}",
                    coords=balance_coords_groups[i],
                    dims=balance_dims,
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

        return 0

    def save_frame(self, frame, window_index, frame_index, name, *, coords=(), dims=()):
        if logging.root.level != logging.DEBUG:
            return

        roi = frame
        if coords and dims:
            x1, x2 = coords[0], coords[0] + dims[0]
            y1, y2 = coords[1], coords[1] + dims[1]
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
