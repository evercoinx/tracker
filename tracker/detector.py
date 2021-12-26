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
        self.prefix = "detector:"

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
        logging.info(f"{self.prefix} window: {window_index}, frame: {frame_number}")

        with PyTessBaseAPI(psm=PSM.SINGLE_LINE, oem=OEM.LSTM_ONLY) as api:
            hand_number = self.detect_hand_number(
                api, frame, frame_number, window_index
            )
            logging.info(f"{self.prefix} hand number: {hand_number}")

            for num in range(1, 7):
                seat = self.detect_seat(api, frame, frame_number, window_index, num)
                logging.info(
                    f"{self.prefix} seat {seat['number']} action: {seat['action']}"
                )
                logging.info(
                    f"{self.prefix} seat {seat['number']} balance: {seat['balance']}"
                )
            return

    def detect_hand_number(self, api, frame, frame_number, window_index):
        roi = cv2.bitwise_not(frame[25:38, 73:174])
        frame_path = (
            f"{self.stream_path}/window{window_index}/"
            + f"{frame_number}_hand_number_processed.png"
        )
        cv2.imwrite(frame_path, roi)

        api.SetVariable("tessedit_char_whitelist", "Hand:#0123456789")
        api.SetImageFile(frame_path)
        line = api.GetUTF8Text()

        matches = re.findall(r"(\d+)$", line.strip())
        if not len(matches):
            return "0"
        return matches[0]

    def detect_seat(self, api, frame, frame_number, window_index, seat_number):
        roi = cv2.bitwise_not(frame[67:118, 478:553])
        frame_path = (
            f"{self.stream_path}/window{window_index}/"
            + f"{frame_number}_seat_{seat_number}_processed.png"
        )
        cv2.imwrite(frame_path, roi)
        return {
            "action": self.detect_seat_action(api, roi),
            "number": self.detect_seat_number(api, roi),
            "balance": self.detect_seat_balance(api, roi),
        }

    def detect_seat_action(self, api, frame):
        roi = frame[:15]
        api.SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        api.SetImage(Image.fromarray(roi))

        line = api.GetUTF8Text()
        matches = re.findall(r"(\w+)$", line.strip())
        if not len(matches):
            return ""
        return matches[0].lower()

    def detect_seat_number(self, api, frame):
        roi = frame[14:31]
        api.SetVariable("tessedit_char_whitelist", "Seat123456")
        api.SetImage(Image.fromarray(roi))

        line = api.GetUTF8Text()
        matches = re.findall(r"(\d)$", line.strip())
        if not len(matches):
            return 0
        return int(matches[0])

    def detect_seat_balance(self, api, frame):
        h = frame.shape[0]
        roi = frame[31:h]
        api.SetVariable("tessedit_char_whitelist", "€0123456789.")
        api.SetImage(Image.fromarray(roi))

        line = api.GetUTF8Text()
        matches = re.findall(r"([.\d]+)$", line.strip())
        if not len(matches):
            return 0.0
        return float(matches[0])
