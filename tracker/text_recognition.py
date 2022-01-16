import re
from datetime import datetime
from enum import Enum
from typing import ClassVar, Dict, Tuple

import numpy as np
from dateutil import parser as dateparser
from PIL import Image
from tesserocr import OEM, PSM, PyTessBaseAPI

from tracker.object_detection import Region


class Currency(Enum):
    UNSET = 0
    EURO = 1
    DOLLAR = 2


class Money:
    currency_to_symbol: ClassVar[Dict[Currency, str]] = {
        Currency.UNSET: "?",
        Currency.EURO: "€",
        Currency.DOLLAR: "$",
    }

    currency: Currency
    amount: float

    def __init__(self, currency: Currency = Currency.UNSET, amount: float = 0) -> None:
        self.currency = currency
        self.amount = amount

    def __bool__(self) -> bool:
        return self.amount != 0

    def __str__(self) -> str:
        return f"{type(self).currency_to_symbol[self.currency]}{self.amount: <5.2f}"

    def __repr__(self) -> str:
        return f"Money({self.currency}, {self.amount})"


class TextRecognition:
    """Recognizes texts on a window frame"""

    regex_action: ClassVar[re.Pattern] = re.compile(
        r"(bet|call|check|fold|raise|allin|sittingin|waitingforbb)",
        flags=re.IGNORECASE,
    )
    regex_money: ClassVar[re.Pattern] = re.compile(r"([$€])([.\d]+)")
    regex_hand_number: ClassVar[re.Pattern] = re.compile(r"(\d{10,})")
    regex_seat_number: ClassVar[re.Pattern] = re.compile(r"([123456])")
    regex_time_with_zone: ClassVar[re.Pattern] = re.compile(r"\d{2}:\d{2}\+\d{2}")

    tess_api: PyTessBaseAPI

    def __init__(self) -> None:
        self.tess_api = PyTessBaseAPI(psm=PSM.SINGLE_LINE, oem=OEM.LSTM_ONLY)

    def __del__(self) -> None:
        self.tess_api.End()

    def set_image(self, image: np.ndarray) -> None:
        self.tess_api.SetImage(Image.fromarray(image))

    def clear_image_results(self) -> None:
        self.tess_api.Clear()

    def recognize_hand_number(self, region: Region) -> int:
        self.tess_api.SetVariable("tessedit_char_whitelist", "Hand:#0123456789")

        dims = self._calculate_rectangle_dimensions(region)
        self.tess_api.SetRectangle(*dims)

        line = self.tess_api.GetUTF8Text()
        matches = re.findall(type(self).regex_hand_number, line.strip())
        if not matches:
            return 0
        return int(matches[0])

    def recognize_hand_time(self, region: Region) -> datetime:
        self.tess_api.SetVariable("tessedit_char_whitelist", ":+0123456789")

        dims = self._calculate_rectangle_dimensions(region)
        self.tess_api.SetRectangle(*dims)

        line = self.tess_api.GetUTF8Text()
        matches = re.findall(type(self).regex_time_with_zone, line.strip())
        if not matches:
            return datetime.min

        try:
            return dateparser.parse(matches[0])
        except Exception:
            return datetime.min

    def recognize_total_pot(self, region: Region) -> Money:
        self.tess_api.SetVariable("tessedit_char_whitelist", "pot:$€0123456789")

        dims = self._calculate_rectangle_dimensions(region)
        self.tess_api.SetRectangle(*dims)

        line = self.tess_api.GetUTF8Text()
        matches = re.findall(type(self).regex_money, line.strip())
        if not matches:
            return Money()
        return self._to_money(
            currency_symbol=matches[0][0], amount_digits=matches[0][1]
        )

    def recognize_seat_number(self, region: Region) -> int:
        self.tess_api.SetVariable("tessedit_char_whitelist", "Seat123456")

        dims = self._calculate_rectangle_dimensions(region)
        self.tess_api.SetRectangle(*dims)

        line = self.tess_api.GetUTF8Text()
        matches = re.findall(type(self).regex_seat_number, line.strip())
        if not matches:
            return 0
        return int(matches[0])

    def recognize_seat_money(self, region: Region) -> Money:
        self.tess_api.SetVariable("tessedit_char_whitelist", "$€0123456789")

        dims = self._calculate_rectangle_dimensions(region)
        self.tess_api.SetRectangle(*dims)

        line = self.tess_api.GetUTF8Text()
        matches = re.findall(type(self).regex_money, line.strip())
        if not matches:
            return Money()
        return self._to_money(
            currency_symbol=matches[0][0], amount_digits=matches[0][1]
        )

    def recognize_seat_action(self, region: Region) -> str:
        self.tess_api.SetVariable(
            "tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        )

        dims = self._calculate_rectangle_dimensions(region)
        self.tess_api.SetRectangle(*dims)

        line = self.tess_api.GetUTF8Text()
        matches = re.findall(type(self).regex_action, line.strip())
        if not matches:
            return ""

        match = matches[0].lower()
        replacements = {
            "allin": "all-in",
            "sittingin": "sitting in",
            "waitingforbb": "waiting for bb",
        }
        return replacements.get(match, match)

    @staticmethod
    def _to_money(currency_symbol: str, amount_digits: str) -> Money:
        currency = Currency.DOLLAR if currency_symbol == "$" else Currency.EURO
        cents = f"{amount_digits:0<3}"
        amount = round(float(cents) / 100, 2)
        return Money(currency, amount)

    @staticmethod
    def _calculate_rectangle_dimensions(region: Region) -> Tuple[int, int, int, int]:
        return (
            region.start.x,
            region.start.y,
            region.end.x - region.start.x,
            region.end.y - region.start.y,
        )
