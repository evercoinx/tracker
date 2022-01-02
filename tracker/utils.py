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


def calculate_end_point(start_point: Point, dimensions: Dimensions) -> Point:
    end_x = start_point.x + dimensions.width
    end_y = start_point.y + dimensions.height
    return Point(end_x, end_y)
