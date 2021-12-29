import logging
import os
import re
from collections import defaultdict
from functools import reduce
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
            matches = re.findall(raw_frame_path_pattern, path)
            if matches:
                (window_index, frame_index) = matches[0]
                self.log_prefix = self.get_log_prefix(window_index, frame_index)
                self.process_frame(frame, window_index, frame_index)

        logging.debug(f"{self.log_prefix} session dump:\n{pformat(self.session)}")

    def process_frame(self, frame, window_index, frame_index):
        processed_frame = cv2.bitwise_not(frame)

        if self.is_debug():
            cv2.imwrite(
                f"{self.stream_path}/window{window_index}/"
                + f"{frame_index}_processed.{self.frame_format}",
                processed_frame,
            )
            logging.debug(f"{self.log_prefix} processed frame saved")

        self.detector.set_frame(processed_frame)

        hand_number = self.get_hand_number(processed_frame, window_index, frame_index)
        if not hand_number:
            try:
                os.remove(
                    f"{self.stream_path}/window{window_index}/"
                    + f"{frame_index}_raw.{self.frame_format}"
                )
            except OSError:
                logging.error(f"{self.log_prefix} no data: raw frame was not removed")
            else:
                logging.warn(
                    f"{self.log_prefix} no data: raw frame removed successfully"
                )
            finally:
                return

        logging.info(f"{self.log_prefix} {'-' * 60}")

        hand_time = self.get_hand_time(processed_frame, window_index, frame_index)
        logging.info(
            f"{self.log_prefix} hand number: {hand_number} "
            + f"at {hand_time.strftime('%H:%M%z')[:-2]}"
        )

        total_pot = self.get_total_pot(processed_frame, window_index, frame_index)
        seats = self.get_seats(processed_frame, window_index, frame_index)
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

        self.session[hand_number].append(
            {
                "window": window_index,
                "frame": frame_index,
                "time": hand_time,
                "seats": seats,
                "total_pot": total_pot,
                "total_stakes": total_stakes,
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

    def get_total_pot(self, frame, window_index, frame_index):
        coords = (462, 160)
        dims = (91, 21)
        total_pot = self.detector.get_total_pot(coords, dims)

        if self.is_debug():
            self.save_frame_roi(
                frame,
                window_index,
                frame_index,
                coords=coords,
                dims=dims,
                name="total_pot",
            )

        return total_pot

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

            stake = self.detector.get_seat_money(stake_coords_groups[i], stake_dims)
            if self.is_debug():
                self.save_frame_roi(
                    frame,
                    window_index,
                    frame_index,
                    coords=stake_coords_groups[i],
                    dims=stake_dims,
                    name=f"seat_stake_{i}",
                )

            balance = self.detector.get_seat_money(
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

            seats.append(
                {
                    "number": number,
                    "action": action,
                    "stake": stake,
                    "balance": balance,
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
