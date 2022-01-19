from datetime import datetime
from typing import List

import grpc
from google.protobuf.timestamp_pb2 import Timestamp
from typing_extensions import TypedDict

import tracker.api.proto.analyzer_pb2 as pbanalyzer
from tracker.api.proto.analyzer_pb2_grpc import AnalyzerStub
from tracker.card import Card
from tracker.vision.text_recognition import Action, Currency, Money


class SeatData(TypedDict):
    name: str
    action: Action
    stake: Money
    balance: Money
    playing: bool


class FrameData(TypedDict):
    window_index: int
    frame_index: int
    hand_number: int
    hand_time: datetime
    total_pot: Money
    dealer_position: int
    seats: List[SeatData]
    board: List[Card]


class AnalyzerClient:
    client: AnalyzerStub

    def __init__(self, api_address: str) -> None:
        chan = grpc.insecure_channel(api_address)
        self.client = AnalyzerStub(chan)

    def send_frame(self, frame_data: FrameData):
        hand_number = frame_data["hand_number"]
        req = pbanalyzer.FrameRequest(
            window_index=frame_data["window_index"],
            frame_index=frame_data["frame_index"],
            hand_number=hand_number,
            hand_time=self._to_pb_timestamp(frame_data["hand_time"]),
            total_pot=self._to_pb_money(frame_data["total_pot"]),
            dealer_position=frame_data["dealer_position"],
            seats=[self._to_pb_seat(s) for s in frame_data["seats"]],
            board=[self._to_pb_card(c) for c in frame_data["board"]],
        )
        self.client.SendFrame(req)

    def _to_pb_timestamp(self, dt: datetime) -> Timestamp:
        ts = dt.timestamp()
        return Timestamp(
            seconds=int(ts),
            nanos=int(ts % 1 * 1e9),
        )

    def _to_pb_money(self, money: Money) -> pbanalyzer.Money:
        mappings = {
            Currency.UNSET: pbanalyzer.Money.Currency.UNSET,
            Currency.EURO: pbanalyzer.Money.Currency.EURO,
            Currency.DOLLAR: pbanalyzer.Money.Currency.DOLLAR,
        }
        return pbanalyzer.Money(
            currency=mappings.get(money.currency, pbanalyzer.Money.Currency.UNSET),
            amount=money.amount,
        )

    def _to_pb_seat(self, seat: SeatData) -> pbanalyzer.Seat:
        return pbanalyzer.Seat(
            name=seat["name"],
            action=self._to_pb_action(seat["action"]),
            stake=self._to_pb_money(seat["stake"]),
            balance=self._to_pb_money(seat["balance"]),
            playing=seat["playing"],
        )

    def _to_pb_action(self, action: Action) -> pbanalyzer.Seat.Action:
        mappings = {
            Action.UNSET: pbanalyzer.Seat.Action.UNSET,
            Action.BET: pbanalyzer.Seat.Action.BET,
            Action.RAISE: pbanalyzer.Seat.Action.RAISE,
            Action.CALL: pbanalyzer.Seat.Action.CALL,
            Action.FOLD: pbanalyzer.Seat.Action.FOLD,
            Action.CHECK: pbanalyzer.Seat.Action.CHECK,
            Action.ALL_IN: pbanalyzer.Seat.Action.ALL_IN,
            Action.SITTING_IN: pbanalyzer.Seat.Action.SITTING_IN,
            Action.WAITING_FOR_BB: pbanalyzer.Seat.Action.WAITING_FOR_BB,
            Action.ANTE: pbanalyzer.Seat.Action.ANTE,
        }
        return mappings.get(action, pbanalyzer.Seat.Action.UNSET)

    def _to_pb_card(self, card: Card) -> str:
        return f"{card.rank}{card.suit}"
