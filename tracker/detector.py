import logging
from multiprocessing import current_process

import cv2


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

                blurred = cv2.GaussianBlur(win_frame, (5, 5), 0)
                thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

                cv2.imwrite(
                    f"{self.stream_path}/window{win_idx+1}/{frame_num}_processed.png",
                    thresh,
                )
                logging.info(f"{prefix} table {win_idx+1}: frame of {len(win_frame)}B")

                frame_num += 1
                self.events[win_idx].set()

            except (KeyboardInterrupt, SystemExit):
                self.queue.close()
                logging.warn(f"{prefix} interruption; exiting...")
                return
