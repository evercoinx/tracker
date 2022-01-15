class ImageError(Exception):
    """Raises an error when an image cannot be read, parsed or recognized"""

    message: str
    path: str

    def __init__(self, message: str, path: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.path = path

    def __str__(self) -> str:
        path = f" at path: {self.path}" if self.path else ""
        return f"{self.message} {path}"
