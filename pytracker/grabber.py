from mss.linux import MSS as mss


def grab(log, queue, display, area):
    with mss(display) as screen:
        for i in range(10):
            screenshot = screen.grab(area)
            queue.put(screenshot)
            log.debug(f"screenshot {i} grabbed")

    queue.put(None)
