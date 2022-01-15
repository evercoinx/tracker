import logging
from multiprocessing import current_process
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event
from typing import List

import cv2
import numpy as np
from mss.linux import MSS as mss
from mss.models import Monitor


class Screen:
    """Capture a window of a screen"""

    queue: Queue
    events: List[Event]
    stream_path: str
    frame_format: str
    log_prefix: str

    def __init__(
        self,
        queue: Queue,
        events: List[Event],
        stream_path: str,
        frame_format: str,
    ) -> None:
        self.queue = queue
        self.events = events
        self.stream_path = stream_path
        self.frame_format = frame_format
        self.log_prefix = ""

    def capture(self, display: str, window_coords: Monitor, window_index: int) -> None:
        frame_index = 0
        self.log_prefix = self._get_log_prefix(window_index, frame_index)

        with mss(display) as screen:
            while True:
                try:
                    frame = screen.grab(window_coords)
                    frame_arr = np.asarray(frame, dtype=np.uint8)
                    gray_frame = cv2.cvtColor(frame_arr, cv2.COLOR_BGRA2GRAY)

                    frame_index += 1
                    self.log_prefix = self._get_log_prefix(window_index, frame_index)

                    self.queue.put((window_index, gray_frame))
                    self.events[window_index].wait()

                except (KeyboardInterrupt, SystemExit):
                    logging.warn(f"{self.log_prefix} interruption; exiting...")
                    return

    @staticmethod
    def _get_log_prefix(window_index: int, frame_index: int):
        return f"{current_process().name}-w{window_index}-f{frame_index:<5} -"

    @staticmethod
    def get_window_screens(
        windows: List[str],
        left_margin: int,
        top_margin: int,
        screen_width: int,
        screen_height: int,
    ) -> List[Monitor]:
        window_width = screen_width // 2
        window_height = (screen_height - top_margin) // 2

        window_coords: List[Monitor] = [
            # top left window, index 0
            {
                "left": left_margin,
                "top": top_margin,
                "width": window_width,
                "height": window_height,
            },
            # top right window, index 1
            {
                "left": left_margin + window_width,
                "top": top_margin,
                "width": window_width,
                "height": window_height,
            },
            # bottom left window, index 2
            {
                "left": left_margin,
                "top": top_margin + window_height,
                "width": window_width,
                "height": window_height,
            },
            # bottom right window, index 3
            {
                "left": left_margin + window_width,
                "top": top_margin + window_height,
                "width": window_width,
                "height": window_height,
            },
        ]

        return [window_coords[int(i)] for i in windows]
