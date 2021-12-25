import logging
from multiprocessing import current_process

import pytesseract


class ObjectDetector:
    """Detect objects on an window frame"""

    def __init__(self, queue, events):
        self.queue = queue
        self.events = events

    def run(self):
        prefix = f"{current_process().name}:"
        config = r"-l eng --oem 3 --psm 6"

        while True:
            try:
                win_idx, win_frame = self.queue.get()
                text = pytesseract.image_to_string(win_frame, config=config)
                logging.info(f"{prefix} table {win_idx+1}: '{text}'")

                self.events[win_idx].set()

            except (KeyboardInterrupt, SystemExit):
                self.queue.close()
                logging.warn(f"{prefix} interruption; exiting...")
                return
