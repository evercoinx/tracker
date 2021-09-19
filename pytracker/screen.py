from mss.linux import MSS as mss


class Screen:
    """Provides API to grab a screen with a given region of interest"""

    def __init__(self, name, *, log, queue):
        self.name = name
        self.log = log
        self.queue = queue

    def grab(self, display, roi):
        prefix = f"{self.name}{display}:"
        i = 0

        with mss(display) as screen:
            while True:
                try:
                    img = screen.grab(roi)
                    self.queue.put(img)
                    self.log.debug(f"{prefix} image {i} grabbed")
                    i += 1
                except (KeyboardInterrupt, SystemExit):
                    self.queue.put(None)
                    self.log.warn(f"{prefix} interruption; exiting...")
                    return
