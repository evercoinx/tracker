import logging
import re
from glob import glob
from multiprocessing import current_process

import cv2
from PIL import Image
from tesserocr import OEM, PSM, PyTessBaseAPI


class ObjectDetector:
    """Detect objects on an window frame"""

    def __init__(self, queue, events, stream_path):
        self.queue = queue
        self.events = events
        self.stream_path = stream_path
        self.prefix = "detector"
        self.tesseract = PyTessBaseAPI(psm=PSM.SINGLE_LINE, oem=OEM.LSTM_ONLY)

    def __del__(self):
        self.tesseract.End()

    def play_live_stream(self):
        self.prefix = f"{current_process().name}:"
        frame_num = 1

        while True:
            try:
                window_index, frame = self.queue.get()
                self.detect_objects(frame, frame_num, window_index + 1)

                frame_num += 1
                self.events[window_index].set()

            except (KeyboardInterrupt, SystemExit):
                self.queue.close()
                logging.warn(f"{self.prefix} interruption; exiting...")
                return

    def replay_saved_stream(self, windows):
        path_pattern = re.compile(
            r"window([" + re.escape(",".join(windows)) + r"])\/(\d+)_raw.png$"
        )
        stream_paths = glob(
            f"{self.stream_path}/window[{''.join(windows)}]/*_raw.png", recursive=True
        )

        for p in stream_paths:
            frame = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            matches = re.findall(path_pattern, p)
            self.detect_objects(frame, matches[0][1], matches[0][0])

    def detect_objects(self, frame, frame_number, window_index):
        self.prefix = f"detector-{window_index}-{frame_number}:"
        frame = cv2.bitwise_not(frame)

        cv2.imwrite(
            f"{self.stream_path}/window{window_index}/{frame_number}_processed.png",
            frame,
        )

        # set image
        self.tesseract.SetImage(Image.fromarray(frame))

        # detect hand number
        hand_number = self.detect_hand_number((73, 24), (101, 15))
        logging.info(f"{self.prefix} hand number: {hand_number}")

        # detect seats
        action_coords = [(138, 321), (172, 100), (478, 68), (709, 100), (728, 338)]
        action_dims = (74, 14)

        number_coords = [(138, 334), (172, 113), (478, 81), (709, 113), (728, 334)]
        number_dims = (74, 15)

        balance_coords = [(138, 351), (172, 130), (478, 98), (709, 130), (728, 351)]
        balance_dims = (74, 16)

        for i in range(len(number_coords)):
            # self.save_frame(
            #     frame,
            #     frame_number,
            #     window_index,
            #     coords=number_coords[i],
            #     dims=number_dims,
            #     name=str(number_coords[i]),
            # )
            number = self.detect_seat_number(number_coords[i], number_dims)
            in_play = "yes" if number else "no "
            balance = self.detect_seat_balance(balance_coords[i], balance_dims)
            action = self.detect_seat_action(action_coords[i], action_dims)
            # self.save_frame(
            #     frame,
            #     frame_number,
            #     window_index,
            #     coords=action_coords[i],
            #     dims=action_dims,
            #     name=str(action_coords[i]),
            # )
            logging.info(
                f"{self.prefix} seat {i}, in play: {in_play} "
                + f"balance: {balance:.2f}, action: {action}"
            )

        # clear image
        self.tesseract.Clear()

    def detect_hand_number(self, coords, dims):
        self.tesseract.SetVariable("tessedit_char_whitelist", "Hand:#0123456789")
        self.tesseract.SetRectangle(coords[0], coords[1], dims[0], dims[1])

        line = self.tesseract.GetUTF8Text()
        matches = re.findall(r"(\d+)$", line.strip())
        if not len(matches):
            return "0"
        return matches[0]

    def detect_seat_number(self, coords, dims):
        self.tesseract.SetVariable("tessedit_char_whitelist", "Seat123456")
        self.tesseract.SetRectangle(coords[0], coords[1], dims[0], dims[1])

        line = self.tesseract.GetUTF8Text()
        matches = re.findall(r"(\d)$", line.strip())
        if not len(matches):
            return 0
        return int(matches[0])

    def detect_seat_balance(self, coords, dims):
        self.tesseract.SetVariable("tessedit_char_whitelist", "€.0123456789")
        self.tesseract.SetRectangle(coords[0], coords[1], dims[0], dims[1])

        line = self.tesseract.GetUTF8Text()
        matches = re.findall(r"([.\d]+)$", line.strip())
        if not len(matches):
            return 0.0
        return float(matches[0])

    def detect_seat_action(self, coords, dims):
        self.tesseract.SetVariable(
            "tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        )
        self.tesseract.SetRectangle(coords[0], coords[1], dims[0], dims[1])

        line = self.tesseract.GetUTF8Text()
        matches = re.findall(r"(\w{3,})$", line.strip())
        if not len(matches):
            return "-"
        return matches[0].lower()

    def save_frame(self, frame, frame_number, window_index, *, coords, dims, name):
        x1, x2 = coords[0], coords[0] + dims[0]
        y1, y2 = coords[1], coords[1] + dims[1]
        roi = frame[y1:y2, x1:x2]

        frame_path = (
            f"{self.stream_path}/window{window_index}/"
            + f"{frame_number}_{name}_processed.png"
        )
        cv2.imwrite(frame_path, roi)
