import logging
from multiprocessing import current_process


class ObjectDetector:
    """Detect objects on an window frame"""

    def __init__(self, queue, events):
        self.queue = queue
        self.events = events

    def run(self):
        prefix = f"{current_process().name}:"

        while True:
            try:
                win_idx, win_frame = self.queue.get()
                logging.info(f"{prefix} table {win_idx+1}: frame of {len(win_frame)}B")

                self.events[win_idx].set()

            except (KeyboardInterrupt, SystemExit):
                self.queue.close()
                logging.warn(f"{prefix} interruption; exiting...")
                return
