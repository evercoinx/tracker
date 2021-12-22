import logging
from multiprocessing import current_process

from mss.linux import MSS as mss


class Screen:
    """Grabs a screen part identified by a region of interest"""

    def __init__(self, queue, events):
        self.queue = queue
        self.events = events

    def capture(self, display, roi, screen_index):
        prefix = f"{current_process().name}{display}:"

        with mss(display) as viewport:
            while True:
                try:
                    screen = viewport.grab(roi)
                    self.queue.put((screen_index, screen))
                    logging.debug(f"{prefix} screen part captured")

                    self.events[screen_index].wait()

                except (KeyboardInterrupt, SystemExit):
                    logging.warn(f"{prefix} interruption; exiting...")
                    return

    @staticmethod
    def get_rois(screen_width, screen_height, left_margin, top_margin):
        roi_width = screen_width // 2
        roi_height = (screen_height - top_margin) // 2

        return (
            # Top Left: 0
            {
                "left": left_margin,
                "top": top_margin,
                "width": roi_width,
                "height": roi_height,
            },
            # Top Right: 1
            {
                "left": left_margin + roi_width,
                "top": top_margin,
                "width": roi_width,
                "height": roi_height,
            },
            # Bottom Left: 2
            {
                "left": left_margin,
                "top": top_margin + roi_height,
                "width": roi_width,
                "height": roi_height,
            },
            # Bottom Right: 3
            {
                "left": left_margin + roi_width,
                "top": top_margin + roi_height,
                "width": roi_width,
                "height": roi_height,
            },
        )
