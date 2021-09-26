import logging
from multiprocessing import current_process

from mss.linux import MSS as mss


def get_rois(width, height):
    c_x = width // 2
    c_h = height // 2

    return (
        {
            "top": 0,
            "left": 0,
            "width": c_x,
            "height": c_h,
        },
        {
            "top": 0,
            "left": c_x,
            "width": c_x,
            "height": c_h,
        },
        {
            "top": c_h,
            "left": 0,
            "width": c_x,
            "height": c_h,
        },
        {
            "top": c_h,
            "left": c_x,
            "width": c_x,
            "height": c_h,
        },
    )


class Screen:
    """Provides API to grab a screen with a given region of interest"""

    def __init__(self, queue, events):
        self.queue = queue
        self.events = events

    def grab(self, display, roi, index):
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
