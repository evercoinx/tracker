import logging
from multiprocessing import current_process

import cv2
import numpy as np


class ObjectDetector:
    """Detect objects on an window frame"""

    def __init__(self, queue, events, stream_path):
        self.queue = queue
        self.events = events
        self.stream_path = stream_path

    def run(self):
        prefix = f"{current_process().name}:"
        frame_num = 1

        while True:
            try:
                win_idx, win_frame = self.queue.get()
                win_arr = np.asarray(win_frame, dtype=np.uint8)

                gray = cv2.cvtColor(win_arr, cv2.COLOR_BGRA2GRAY)
                cv2.imwrite(
                    f"{self.stream_path}/table{win_idx+1}/{frame_num}.png",
                    gray,
                )
                logging.info(f"{prefix} table {win_idx+1}: frame {frame_num}.png saved")

                frame_num += 1
                self.events[win_idx].set()

            except (KeyboardInterrupt, SystemExit):
                self.queue.close()
                logging.warn(f"{prefix} interruption; exiting...")
                return
