class ValidationError(Exception):
    """Raises an error when user input is invalid or the environment variable has
    an unexpected value"""


class FrameError(Exception):
    """Raises an error when an unexpected behaviour during processing of a frame
    is occurred"""

    def __init__(self, message, window_index, frame_index, name):
        super().__init__(message)
        self.window_index = window_index
        self.frame_index = frame_index
        self.name = name

    def __str__(self):
        return f"w{self.window_index}-f{self.frame_index:<5} - {self.message}"
