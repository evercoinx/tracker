import multiprocessing as mp

from mss.linux import MSS as mss


def get_rois(width, height):
    c_w = width // 2
    c_h = height // 2

    return (
        {
            "top": 0,
            "left": 0,
            "width": c_w,
            "height": c_h,
        },
        {
            "top": 0,
            "left": c_w,
            "width": c_w,
            "height": c_h,
        },
        {
            "top": c_h,
            "left": 0,
            "width": c_w,
            "height": c_h,
        },
        {
            "top": c_h,
            "left": c_w,
            "width": c_w,
            "height": c_h,
        },
    )


class Screen:
    """Provides API to grab a screen with a given region of interest"""

    def __init__(self, log, queue, events):
        self.log = log
        self.queue = queue
        self.events = events

    def grab(self, display, roi, index):
        prefix = f"{mp.current_process().name}{display}:"

        with mss(display) as screen:
            while True:
                try:
                    img = screen.grab(roi)
                    self.queue.put((index, img))
                    self.log.debug(f"{prefix} image part grabbed")

                    self.events[index].wait()
                except (KeyboardInterrupt, SystemExit):
                    self.log.warn(f"{prefix} interruption; exiting...")
                    return
