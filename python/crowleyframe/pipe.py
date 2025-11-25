from __future__ import annotations

from typing import Any, Callable

from .api import Frame, GroupedFrame


class PipeNamespace:
    """
    pipe.group_by("id") returns a function that takes a Frame
    and returns a GroupedFrame, so you can write:

        cf >> pipe.group_by("id") >> pipe.summarise(...)
        cf | pipe.group_by("id") | pipe.summarise(...)

    Also:
        cf >> pipe.filter("score > 10")
        cf >> pipe.arrange("-score")
    """

    def group_by(self, *cols: str) -> Callable[[Frame], GroupedFrame]:
        def _op(frame: Frame) -> GroupedFrame:
            return frame.group_by(*cols)

        return _op

    def summarise(self, **agg_specs: tuple[str, str]) -> Callable[[GroupedFrame], Frame]:
        def _op(obj: GroupedFrame) -> Frame:
            return obj.summarise(**agg_specs)

        return _op

    def filter(self, expr: str) -> Callable[[Frame], Frame]:
        def _op(frame: Frame) -> Frame:
            return frame.filter(expr=expr)

        return _op

    def arrange(self, *cols: str) -> Callable[[Frame], Frame]:
        def _op(frame: Frame) -> Frame:
            return frame.arrange(*cols)

        return _op


pipe = PipeNamespace()
