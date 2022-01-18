from typing import ClassVar, Dict, FrozenSet


class Card:
    ranks: ClassVar[FrozenSet[str]] = frozenset(
        ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
    )
    suit_to_symbol: ClassVar[Dict[str, str]] = {
        "c": "♣",
        "d": "♦",
        "h": "♥",
        "s": "♠",
    }

    rank: str
    suit: str

    def __init__(self, rank: str, suit: str):
        if rank not in type(self).ranks:
            raise ValueError(f"Unexpected rank: {rank}")
        if suit not in type(self).suit_to_symbol:
            raise ValueError(f"Unexpected suit: {suit}")

        self.rank = rank
        self.suit = suit

    def __str__(self) -> str:
        return f"{self.rank}{self.suit_to_symbol[self.suit]}"

    def __repr__(self) -> str:
        return f"Card('{self.rank}', '{self.suit}')"
