import logging
import os
import re
from collections import defaultdict
from glob import glob
from multiprocessing import current_process
from pprint import pformat

import cv2


class StreamPlayer:
    """Plays a live or saved stream"""

    def __init__(self, queue, events, detector, stream_path, frame_format):
        self.queue = queue
        self.events = events
        self.detector = detector
        self.stream_path = stream_path
        self.frame_format = frame_format
        self.log_prefix = ""
        self.session = defaultdict(list)

    def play_live(self):
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

    def play_saved(self, windows):
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
            matches = re.findall(raw_frame_path_pattern, p)
            if matches:
                (window_index, frame_index) = matches[0]
                self.log_prefix = self.get_log_prefix(window_index, frame_index)
                self.process_frame(frame, window_index, frame_index)

        logging.debug(f"{self.log_prefix} session dump:\n{pformat(self.session)}")

    def process_frame(self, frame, window_index, frame_index):
        frame = cv2.bitwise_not(frame)

        if self.is_debug():
            cv2.imwrite(
                f"{self.stream_path}/window{window_index}/"
                + f"{frame_index}_processed.{self.frame_format}",
                frame,
            )
            logging.debug(f"{self.log_prefix} processed frame saved")

        self.detector.set_frame(frame)

        hand_number = self.get_hand_number(frame, window_index, frame_index)
        if not hand_number:
            os.remove(
                f"{self.stream_path}/window{window_index}/"
                + f"{frame_index}_raw.{self.frame_format}"
            )
            logging.warn(f"{self.log_prefix} raw frame removed as no data found")
            return

        hand_time = self.get_hand_time(frame, window_index, frame_index)
        logging.info(
            f"{self.log_prefix} hand number: {hand_number} "
            + f"at {hand_time.strftime('%H:%M%z')[:-2]}"
        )

        seats = self.get_seats(frame, window_index, frame_index)
        for s in seats:
            logging.info(
                f"{self.log_prefix} seat {s['number']}, "
                + f"balance: {s['balance']:.2f}, action: {s['action']}"
            )

        self.session[hand_number].append(
            {
                "window": window_index,
                "frame": frame_index,
                "time": hand_time,
                "seats": seats,
            }
        )

        self.detector.clear_current_frame()

    def get_hand_number(self, frame, window_index, frame_index):
        coords = (73, 24)
        dims = (101, 15)
        hand_number = self.detector.get_hand_number(coords, dims)

        if self.is_debug():
            self.save_frame_roi(
                frame,
                window_index,
                frame_index,
                coords=coords,
                dims=dims,
                name="hand_number",
            )

        return hand_number

    def get_hand_time(self, frame, window_index, frame_index):
        coords = (857, 22)
        dims = (55, 14)
        hand_time = self.detector.get_hand_time(coords, dims)

        if self.is_debug():
            self.save_frame_roi(
                frame,
                window_index,
                frame_index,
                coords=coords,
                dims=dims,
                name="hand_time",
            )

        return hand_time

    def get_seats(self, frame, window_index, frame_index):
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

        seats = []

        for i in range(len(number_coords_groups)):
            # last player is a hero
            if i == len(number_coords_groups) - 1:
                number = 9
            else:
                number = self.detector.get_seat_number(
                    number_coords_groups[i], number_dims
                )

            if self.is_debug():
                self.save_frame_roi(
                    frame,
                    window_index,
                    frame_index,
                    coords=number_coords_groups[i],
                    dims=number_dims,
                    name=f"seat_number_{i}",
                )

            # if we failed to detect a seat number it is unreasonable to look for
            # a balance and an action of this seat
            if not number:
                continue

            balance = self.detector.get_seat_balance(
                balance_coords_groups[i], balance_dims
            )
            if self.is_debug():
                self.save_frame_roi(
                    frame,
                    window_index,
                    frame_index,
                    coords=balance_coords_groups[i],
                    dims=balance_dims,
                    name=f"seat_balance_{i}",
                )

            action = self.detector.get_seat_action(action_coords_groups[i], action_dims)
            if self.is_debug():
                self.save_frame_roi(
                    frame,
                    window_index,
                    frame_index,
                    coords=action_coords_groups[i],
                    dims=action_dims,
                    name=f"seat_action_{i}",
                )

            seats.append(
                {
                    "number": number,
                    "balance": balance,
                    "action": action,
                }
            )

        return seats

    def save_frame_roi(self, frame, window_index, frame_index, *, coords, dims, name):
        x1, x2 = coords[0], coords[0] + dims[0]
        y1, y2 = coords[1], coords[1] + dims[1]
        frame_roi = frame[y1:y2, x1:x2]

        cv2.imwrite(
            f"{self.stream_path}/window{window_index}/"
            + f"{frame_index}_{name}_processed.{self.frame_format}",
            frame_roi,
        )

    @staticmethod
    def is_debug():
        return logging.root.level == logging.DEBUG

    @staticmethod
    def get_log_prefix(window_index, frame_index):
        proc_name = current_process().name
        if proc_name == "MainProcess":  # no multiprocessing
            proc_name = "player"
        return f"{proc_name}-w{window_index}-f{frame_index:<5} -"
