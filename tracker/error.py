class ValidationError(Exception):
    """Raises an error when user input is invalid or the environment variable has
    an unexpected value"""


class FrameError(Exception):
    """Raises an error when a window frame is unable to be read or written"""

    message: str
    window_index: int
    frame_index: int
    frame_name: str

    def __init__(
        self, message: str, window_index: int, frame_index: int, frame_name: str
    ) -> None:
        super().__init__(message)
        self.message = message
        self.window_index = window_index
        self.frame_index = frame_index
        self.frame_name = frame_name

    def __str__(self) -> str:
        return (
            f"{self.frame_name}-w{self.window_index}-f{self.frame_index:<5} "
            + f"- {self.message}"
        )


class ImageError(Exception):
    """Raises an error when an image is unable to be read or parsed"""

    message: str
    path: str

    def __init__(self, message: str, path: str) -> None:
        super().__init__(message)
        self.message = message
        self.path = path

    def __str__(self) -> str:
        return f"{self.message} at {self.path}"
