from typing import NamedTuple


class Dimensions(NamedTuple):
    width: int
    height: int


class Point(NamedTuple):
    x: int
    y: int


class Region(NamedTuple):
    start: Point
    end: Point
