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

    def __init__(self, log, data_queue, event_queues):
        self.log = log
        self.data_queue = data_queue
        self.event_queues = event_queues

    def grab(self, display, roi, index):
        prefix = f"{mp.current_process().name}{display}:"

        with mss(display) as screen:
            while True:
                try:
                    img = screen.grab(roi)
                    self.data_queue.put((index, img))
                    self.log.debug(f"{prefix} image grabbed")

                    self.event_queues[index].get()
                except (KeyboardInterrupt, SystemExit):
                    self.event_queues[index].close()
                    self.log.warn(f"{prefix} interruption; exiting...")
                    return
