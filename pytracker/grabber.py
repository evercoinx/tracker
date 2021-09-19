from mss.linux import MSS as mss


def grab(log, queue, display, roi):
    name = "grabber"
    i = 0

    with mss(display) as screen:
        while True:
            try:
                img = screen.grab(roi)
                queue.put(img)
                log.debug(f"{name}: image {i} grabbed")
                i += 1
            except (KeyboardInterrupt, SystemExit):
                queue.put(None)
                log.warn(f"{name}: interruption; exiting...")
                return
