import re

from PIL import Image
from tesserocr import OEM, PSM, PyTessBaseAPI


class ObjectDetector:
    """Detect objects on an window frame"""

    def __init__(self):
        self.tess_api = PyTessBaseAPI(psm=PSM.SINGLE_LINE, oem=OEM.LSTM_ONLY)

    def __del__(self):
        self.tess_api.End()

    def set_frame(self, frame):
        self.tess_api.SetImage(Image.fromarray(frame))

    def clear_current_frame(self):
        self.tess_api.Clear()

    def get_hand_number(self, coords, dims):
        self.tess_api.SetVariable("tessedit_char_whitelist", "Hand:#0123456789")
        self.tess_api.SetRectangle(coords[0], coords[1], dims[0], dims[1])

        line = self.tess_api.GetUTF8Text()
        matches = re.findall(r"(\d+)$", line.strip())
        if not len(matches):
            return 0
        return int(matches[0])

    def get_seat_number(self, coords, dims):
        self.tess_api.SetVariable("tessedit_char_whitelist", "Seat123456")
        self.tess_api.SetRectangle(coords[0], coords[1], dims[0], dims[1])

        line = self.tess_api.GetUTF8Text()
        matches = re.findall(r"(\d)$", line.strip())
        if not len(matches):
            return 0
        return int(matches[0])

    def get_seat_balance(self, coords, dims):
        self.tess_api.SetVariable("tessedit_char_whitelist", "€.0123456789")
        self.tess_api.SetRectangle(coords[0], coords[1], dims[0], dims[1])

        line = self.tess_api.GetUTF8Text()
        matches = re.findall(r"([.\d]+)$", line.strip())
        if not len(matches):
            return 0.0
        return float(matches[0])

    def get_seat_action(self, coords, dims):
        self.tess_api.SetVariable(
            "tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        )
        self.tess_api.SetRectangle(coords[0], coords[1], dims[0], dims[1])

        line = self.tess_api.GetUTF8Text()
        matches = re.findall(r"(\w{3,})$", line.strip())
        if not len(matches):
            return "none"
        return matches[0].lower()
