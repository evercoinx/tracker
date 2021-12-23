import logging
from multiprocessing import current_process

from mss.linux import MSS as mss


class Screen:
    """Capture a screen window by coordinates"""

    def __init__(self, queue, events):
        self.queue = queue
        self.events = events

    def capture(self, display, window_coords, window_index):
        prefix = f"{current_process().name}{display}:"

        with mss(display) as screen:
            while True:
                try:
                    window_image = screen.grab(window_coords)
                    self.queue.put((window_index, window_image))
                    logging.debug(f"{prefix} window {window_index+1} captured")

                    self.events[window_index].wait()

                except (KeyboardInterrupt, SystemExit):
                    logging.warn(f"{prefix} interruption; exiting...")
                    return

    @staticmethod
    def calculate_window_coords(screen_width, screen_height, left_margin, top_margin):
        window_width = screen_width // 2
        window_height = (screen_height - top_margin) // 2

        return (
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
