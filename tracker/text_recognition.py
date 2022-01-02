import re
from datetime import datetime

import numpy as np
from dateutil import parser as dateparser
from PIL import Image
from tesserocr import OEM, PSM, PyTessBaseAPI

from tracker.utils import Region


class TextRecognition:
    """Recognizes texts on a window frame"""

    REGEX_ACTION = re.compile(
        r"(bet|call|check|fold|raise|sittingin|waitingforbb)", flags=re.IGNORECASE
    )
    REGEX_MONEY = re.compile(r"[$€]([.\d]+)")
    REGEX_MULTIPLE_DIGITS = re.compile(r"(\d+)")
    REGEX_SINGLE_DIGIT = re.compile(r"(\d)")
    REGEX_TIME_WITH_ZONE = re.compile(r"\d{2}:\d{2}\+\d{2}")

    def __init__(self) -> None:
        self.tess_api = PyTessBaseAPI(psm=PSM.SINGLE_LINE, oem=OEM.LSTM_ONLY)

    def __del__(self) -> None:
        self.tess_api.End()

    def set_frame(self, frame: np.ndarray) -> None:
        self.tess_api.SetImage(Image.fromarray(frame))

    def clear_current_frame(self) -> None:
        self.tess_api.Clear()

    def get_hand_number(self, region: Region) -> int:
        self.tess_api.SetVariable("tessedit_char_whitelist", "Hand:#0123456789")
        self.tess_api.SetRectangle(
            region.start.x,
            region.start.y,
            region.end.x - region.start.x,
            region.end.y - region.start.y,
        )

        line = self.tess_api.GetUTF8Text()
        matches = re.findall(TextRecognition.REGEX_MULTIPLE_DIGITS, line.strip())
        if not matches:
            return 0
        return int(matches[0])

    def get_hand_time(self, region: Region) -> datetime:
        self.tess_api.SetVariable("tessedit_char_whitelist", ":+0123456789")
        self.tess_api.SetRectangle(
            region.start.x,
            region.start.y,
            region.end.x - region.start.x,
            region.end.y - region.start.y,
        )

        line = self.tess_api.GetUTF8Text()
        matches = re.findall(TextRecognition.REGEX_TIME_WITH_ZONE, line.strip())
        if not matches:
            return datetime.min
        return dateparser.parse(matches[0])

    def get_total_pot(self, region: Region) -> float:
        self.tess_api.SetVariable("tessedit_char_whitelist", "pot:€.0123456789")
        self.tess_api.SetRectangle(
            region.start.x,
            region.start.y,
            region.end.x - region.start.x,
            region.end.y - region.start.y,
        )

        line = self.tess_api.GetUTF8Text()
        matches = re.findall(TextRecognition.REGEX_MONEY, line.strip())
        if not matches:
            return 0.0
        return float(matches[0])

    def get_seat_number(self, region: Region) -> int:
        self.tess_api.SetVariable("tessedit_char_whitelist", "Seat123456")
        self.tess_api.SetRectangle(
            region.start.x,
            region.start.y,
            region.end.x - region.start.x,
            region.end.y - region.start.y,
        )

        line = self.tess_api.GetUTF8Text()
        matches = re.findall(TextRecognition.REGEX_SINGLE_DIGIT, line.strip())
        if not matches:
            return 0
        return int(matches[0])

    def get_seat_money(self, region: Region) -> float:
        self.tess_api.SetVariable("tessedit_char_whitelist", "€.0123456789")
        self.tess_api.SetRectangle(
            region.start.x,
            region.start.y,
            region.end.x - region.start.x,
            region.end.y - region.start.y,
        )

        line = self.tess_api.GetUTF8Text()
        matches = re.findall(TextRecognition.REGEX_MONEY, line.strip())
        if not matches:
            return 0.0
        return float(matches[0])

    def get_seat_action(self, region: Region) -> str:
        self.tess_api.SetVariable(
            "tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        )
        self.tess_api.SetRectangle(
            region.start.x,
            region.start.y,
            region.end.x - region.start.x,
            region.end.y - region.start.y,
        )

        line = self.tess_api.GetUTF8Text()
        matches = re.findall(TextRecognition.REGEX_ACTION, line.strip())
        if not matches:
            return "none"

        match = matches[0].lower()
        if match == "sittingin":
            return "sitting in"
        elif match == "waitingforbb":
            return "waiting for bb"
        return match
