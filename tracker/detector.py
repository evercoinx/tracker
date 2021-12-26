import logging
import re
from glob import glob
from multiprocessing import current_process

import cv2


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
                self.process_frame(frame, frame_num, window_index + 1)

                frame_num += 1
                self.events[window_index].set()

            except (KeyboardInterrupt, SystemExit):
                self.queue.close()
                logging.warn(f"{self.prefix} interruption; exiting...")
                return

    def replay_saved_stream(self):
        path_pattern = re.compile(r"window(\d+)\/(\d+)_raw.png$")
        stream_paths = glob(f"{self.stream_path}/**/*_raw.png", recursive=True)

        for p in stream_paths:
            frame = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            matches = re.findall(path_pattern, p)
            self.process_frame(frame, matches[0][1], matches[0][0])

    def process_frame(self, frame, frame_number, window_index):
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

        cv2.imwrite(
            f"{self.stream_path}/window{window_index}/{frame_number}_processed.png",
            thresh,
        )
        logging.info(f"{self.prefix} table {window_index}: frame of {len(frame)}B")
        return
