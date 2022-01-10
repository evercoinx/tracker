import re
from datetime import datetime
from typing import Tuple

import numpy as np
from dateutil import parser as dateparser
from PIL import Image
from tesserocr import OEM, PSM, PyTessBaseAPI  # pytype: disable=import-error

from tracker.object_detection import Region


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

    def clear_frame_results(self) -> None:
        self.tess_api.Clear()

    def recognize_hand_number(self, region: Region) -> int:
        self.tess_api.SetVariable("tessedit_char_whitelist", "Hand:#0123456789")

        dims = self.calculate_rectangle_dimensions(region)
        self.tess_api.SetRectangle(*dims)

        line = self.tess_api.GetUTF8Text()
        matches = re.findall(TextRecognition.REGEX_MULTIPLE_DIGITS, line.strip())
        if not matches:
            return 0
        return int(matches[0])

    def recognize_hand_time(self, region: Region) -> datetime:
        self.tess_api.SetVariable("tessedit_char_whitelist", ":+0123456789")

        dims = self.calculate_rectangle_dimensions(region)
        self.tess_api.SetRectangle(*dims)

        line = self.tess_api.GetUTF8Text()
        matches = re.findall(TextRecognition.REGEX_TIME_WITH_ZONE, line.strip())
        if not matches:
            return datetime.min

        try:
            return dateparser.parse(matches[0])
        except Exception:
            return datetime.min

    def recognize_total_pot(self, region: Region) -> float:
        self.tess_api.SetVariable("tessedit_char_whitelist", "pot:€.0123456789")

        dims = self.calculate_rectangle_dimensions(region)
        self.tess_api.SetRectangle(*dims)

        line = self.tess_api.GetUTF8Text()
        matches = re.findall(TextRecognition.REGEX_MONEY, line.strip())
        if not matches:
            return 0.0
        return float(matches[0])

    def recognize_seat_number(self, region: Region) -> int:
        self.tess_api.SetVariable("tessedit_char_whitelist", "Seat123456")

        dims = self.calculate_rectangle_dimensions(region)
        self.tess_api.SetRectangle(*dims)

        line = self.tess_api.GetUTF8Text()
        matches = re.findall(TextRecognition.REGEX_SINGLE_DIGIT, line.strip())
        if not matches:
            return -1
        return int(matches[0])

    def recognize_seat_money(self, region: Region) -> float:
        self.tess_api.SetVariable("tessedit_char_whitelist", "€.0123456789")

        dims = self.calculate_rectangle_dimensions(region)
        self.tess_api.SetRectangle(*dims)

        line = self.tess_api.GetUTF8Text()
        matches = re.findall(TextRecognition.REGEX_MONEY, line.strip())
        if not matches:
            return 0.0
        return float(matches[0])

    def recognize_seat_action(self, region: Region) -> str:
        self.tess_api.SetVariable(
            "tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        )

        dims = self.calculate_rectangle_dimensions(region)
        self.tess_api.SetRectangle(*dims)

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

    @staticmethod
    def calculate_rectangle_dimensions(region: Region) -> Tuple[int]:
        return (
            region.start.x,
            region.start.y,
            region.end.x - region.start.x,
            region.end.y - region.start.y,
        )
