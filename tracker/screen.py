import logging
from multiprocessing import current_process

import cv2
import numpy as np
from mss.linux import MSS as mss


class Screen:
    """Capture a window of a screen"""

    def __init__(self, queue, events, stream_path, frame_format):
        self.queue = queue
        self.events = events
        self.stream_path = stream_path
        self.frame_format = frame_format
        self.log_prefix = ""

    def capture(self, display, window_coords, window_index):
        frame_index = 0
        self.log_prefix = self.get_log_prefix(window_index, frame_index)

        with mss(display) as screen:
            while True:
                try:
                    color_frame = screen.grab(window_coords)
                    win_arr = np.asarray(color_frame, dtype=np.uint8)
                    gray_frame = cv2.cvtColor(win_arr, cv2.COLOR_BGRA2GRAY)

                    cv2.imwrite(
                        f"{self.stream_path}/window{window_index}/"
                        + f"{frame_index}_raw.{self.frame_format}",
                        gray_frame,
                    )
                    logging.info(f"{self.log_prefix} raw frame saved")

                    frame_index += 1
                    self.log_prefix = self.get_log_prefix(window_index, frame_index)

                    self.queue.put((window_index, gray_frame))
                    self.events[window_index].wait()

                except (KeyboardInterrupt, SystemExit):
                    logging.warn(f"{self.log_prefix} interruption; exiting...")
                    return

    @staticmethod
    def calculate_window_coords(
        window_indexes, screen_width, screen_height, left_margin, top_margin
    ):
        window_width = screen_width // 2
        window_height = (screen_height - top_margin) // 2

        windows = (
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
        )

        return [windows[i] for i in window_indexes]

    @staticmethod
    def get_log_prefix(window_index, frame_index):
        return f"{current_process().name}-w{window_index}-f{frame_index:<5} -"
