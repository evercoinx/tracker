import logging
from multiprocessing import current_process

from mss.linux import MSS as mss


class Grabber:
    """Provides API to grab a screen with a given region of interest"""

    def __init__(self, queue, events):
        self.queue = queue
        self.events = events

    def capture(self, display, roi, index):
        prefix = f"{current_process().name}{display}:"

        with mss(display) as screen:
            while True:
                try:
                    img = screen.grab(roi)
                    self.queue.put((index, img))
                    logging.debug(f"{prefix} image part grabbed")

                    self.events[index].wait()
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
