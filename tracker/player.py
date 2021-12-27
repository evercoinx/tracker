import logging
import re
from glob import glob
from multiprocessing import current_process

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

    def play_live(self):
        frame_index = 0

        while True:
            try:
                window_index, frame = self.queue.get()
                self.set_log_prefix(window_index, frame_index)
                self.play_stream(window_index, frame_index, frame)

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
            + "$"
        )
        raw_frame_paths = glob(
            f"{self.stream_path}/window[{''.join(windows)}]/*_raw.{self.frame_format}",
            recursive=True,
        )

        for p in sorted(raw_frame_paths):
            frame = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            matches = re.findall(raw_frame_path_pattern, p)
            window_index = matches[0][0]
            frame_index = matches[0][1]

            self.set_log_prefix(window_index, frame_index)
            self.play_stream(window_index, frame_index, frame)

    def play_stream(self, window_index, frame_index, frame):
        frame = cv2.bitwise_not(frame)

        if self.is_debug():
            cv2.imwrite(
                f"{self.stream_path}/window{window_index}/"
                + f"{frame_index}_processed.{self.frame_format}",
                frame,
            )
            logging.debug(f"{self.log_prefix} processed frame saved")

        self.detector.set_frame(frame)

        self.print_hand_number(window_index, frame_index, frame)
        self.print_seats(window_index, frame_index, frame)

        self.detector.clear_current_frame()

    def print_hand_number(self, window_index, frame_index, frame):
        coords = (73, 24)
        dims = (101, 15)
        hand_number = self.detector.get_hand_number(coords, dims)
        if self.is_debug():
            self.save_frame_roi(
                window_index,
                frame_index,
                frame,
                coords=coords,
                dims=dims,
                name=f"hand_number{coords}",
            )
        logging.info(f"{self.log_prefix} hand number: {hand_number}")

    def print_seats(self, window_index, frame_index, frame):
        action_coords_groups = [
            (138, 321),
            (172, 100),
            (478, 68),
            (709, 100),
            (728, 338),
        ]
        action_dims = (74, 14)

        number_coords_groups = [
            (138, 334),
            (172, 113),
            (478, 81),
            (709, 113),
            (728, 334),
        ]
        number_dims = (74, 15)

        balance_coords_groups = [
            (138, 351),
            (172, 130),
            (478, 98),
            (709, 130),
            (728, 351),
        ]
        balance_dims = (74, 16)

        for i in range(len(number_coords_groups)):
            number = self.detector.get_seat_number(number_coords_groups[i], number_dims)
            in_play = "yes" if number else "no "
            if self.is_debug():
                self.save_frame_roi(
                    window_index,
                    frame_index,
                    frame,
                    coords=number_coords_groups[i],
                    dims=number_dims,
                    name=f"seat_number_{number_coords_groups[i]}",
                )

            balance = self.detector.get_seat_balance(
                balance_coords_groups[i], balance_dims
            )
            if self.is_debug():
                self.save_frame_roi(
                    window_index,
                    frame_index,
                    frame,
                    coords=balance_coords_groups[i],
                    dims=balance_dims,
                    name=f"seat_balance_{balance_coords_groups[i]}",
                )

            action = self.detector.get_seat_action(action_coords_groups[i], action_dims)
            if self.is_debug():
                self.save_frame_roi(
                    window_index,
                    frame_index,
                    frame,
                    coords=action_coords_groups[i],
                    dims=action_dims,
                    name=f"seat_action_{action_coords_groups[i]}",
                )
            logging.info(
                f"{self.log_prefix} seat {number}, in play: {in_play}, "
                + f"balance: {balance:.2f}, action: {action}"
            )

    def save_frame_roi(self, window_index, frame_index, frame, *, coords, dims, name):
        x1, x2 = coords[0], coords[0] + dims[0]
        y1, y2 = coords[1], coords[1] + dims[1]
        frame_roi = frame[y1:y2, x1:x2]

        frame_path = (
            f"{self.stream_path}/window{window_index}/"
            + f"{frame_index}_{name}_processed.{self.frame_format}"
        )
        cv2.imwrite(frame_path, frame_roi)

    def set_log_prefix(self, window_index, frame_index):
        proc_name = current_process().name
        if proc_name == "MainProcess":  # no multiprocessing
            proc_name = "player"
        self.log_prefix = f"{proc_name}-w{window_index}-f{frame_index:<5} -"

    @staticmethod
    def is_debug():
        logging.root.level == logging.DEBUG
