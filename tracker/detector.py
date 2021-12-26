import logging
import re
from glob import glob
from multiprocessing import current_process

import cv2
import pytesseract


class ObjectDetector:
    """Detect objects on an window frame"""

    TESSERACT_LANGUAGE = "eng"
    TESSERACT_NICENESS = -10
    TESSERACT_TIMEOUT = 2

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
        logging.info(f"{self.prefix} window #{window_index}, frame #{frame_number}")

        hand_number = self.detect_hand_number(frame, frame_number, window_index)
        logging.info(f"{self.prefix} hand number #{hand_number}")
        return

    def detect_hand_number(self, frame, frame_number, window_index):
        roi = cv2.bitwise_not(frame[21:42, 69:178])
        cv2.imwrite(
            f"{self.stream_path}/window{window_index}/"
            + f"{frame_number}_hand_number_processed.png",
            roi,
        )

        line = pytesseract.image_to_string(
            roi,
            config="--oem 1 --psm 7 -c tessedit_char_whitelist=Hand:#0123456789",
            lang=ObjectDetector.TESSERACT_LANGUAGE,
            nice=ObjectDetector.TESSERACT_NICENESS,
            timeout=ObjectDetector.TESSERACT_TIMEOUT,
        )

        matches = re.findall(r"(\d+)$", line.strip())
        if not len(matches):
            return "0"
        return matches[0]
