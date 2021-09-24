import multiprocessing as mp

from mss.linux import MSS as mss

rois = (
    {
        "top": 0,
        "left": 0,
        "width": 960,
        "height": 540,
    },
    {
        "top": 0,
        "left": 960,
        "width": 960,
        "height": 540,
    },
    {
        "top": 540,
        "left": 0,
        "width": 960,
        "height": 540,
    },
    {
        "top": 540,
        "left": 960,
        "width": 960,
        "height": 540,
    },
)


class Screen:
    """Provides API to grab a screen with a given region of interest"""

    def __init__(self, log, queue, events):
        self.log = log
        self.queue = queue
        self.events = events

    def grab(self, display, roi, event_index):
        prefix = f"{mp.current_process().name}{display}:"

        with mss(display) as screen:
            while True:
                try:
                    img = screen.grab(roi)
                    self.queue.put((event_index, img))
                    self.log.debug(f"{prefix} image part grabbed")

                    self.events[event_index].wait()
                except (KeyboardInterrupt, SystemExit):
                    self.log.warn(f"{prefix} interruption; exiting...")
                    return
