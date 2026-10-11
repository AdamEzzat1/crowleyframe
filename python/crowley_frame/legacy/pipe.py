from __future__ import annotations

from typing import Any, Callable, Union

from .api import Frame, GroupedFrame


class PipeNamespace:
    """
    Small helper namespace for building pipeable operations.

    Each method returns a callable that takes a Frame (or GroupedFrame)
    and returns a new Frame/GroupedFrame, so you can write:

        cf >> pipe.group_by("id") >> pipe.summarise(mean_x=("x", "mean"))
        cf >> pipe.filter("x > 0") >> pipe.arrange("x")
    """

    # -----------------------------
    # Grouped operations
    # -----------------------------
    def group_by(self, *cols: str) -> Callable[[Frame], GroupedFrame]:
        def _op(frame: Frame) -> GroupedFrame:
            return frame.group_by(*cols)

        return _op

    def summarise(
        self,
        **agg_specs: tuple[str, Union[str, Callable[..., Any]]],
    ) -> Callable[[GroupedFrame], Frame]:
        def _op(grouped: GroupedFrame) -> Frame:
            return grouped.summarise(**agg_specs)

        return _op

    # -----------------------------
    # Frame-level operations
    # -----------------------------
    def filter(self, expr: str) -> Callable[[Frame], Frame]:
        def _op(frame: Frame) -> Frame:
            return frame.filter(expr)

        return _op

    def arrange(self, *cols: str) -> Callable[[Frame], Frame]:
        def _op(frame: Frame) -> Frame:
            return frame.arrange(*cols)

        return _op

    def select(self, *selectors: Any) -> Callable[[Frame], Frame]:
        def _op(frame: Frame) -> Frame:
            return frame.select(*selectors)

        return _op

    def mutate(self, **kwargs: Any) -> Callable[[Frame], Frame]:
        def _op(frame: Frame) -> Frame:
            return frame.mutate(**kwargs)

        return _op

    def rename(self, **mapping: str) -> Callable[[Frame], Frame]:
        def _op(frame: Frame) -> Frame:
            return frame.rename(**mapping)

        return _op

    def relocate(
        self,
        *cols: str,
        before: str | None = None,
        after: str | None = None,
    ) -> Callable[[Frame], Frame]:
        def _op(frame: Frame) -> Frame:
            return frame.relocate(*cols, before=before, after=after)

        return _op

    def distinct(self, *cols: str, keep: str = "first") -> Callable[[Frame], Frame]:
        def _op(frame: Frame) -> Frame:
            return frame.distinct(*cols, keep=keep)

        return _op

    # -----------------------------
    # Joins
    # -----------------------------
    def left_join(
        self,
        other: Frame,
        on: str | list[str] | None = None,
        left_on: str | list[str] | None = None,
        right_on: str | list[str] | None = None,
        suffixes: tuple[str, str] = ("_x", "_y"),
        validate: str | None = None,
    ) -> Callable[[Frame], Frame]:
        def _op(frame: Frame) -> Frame:
            return frame.left_join(
                other,
                on=on,
                left_on=left_on,
                right_on=right_on,
                suffixes=suffixes,
                validate=validate,
            )

        return _op

    def inner_join(
        self,
        other: Frame,
        on: str | list[str] | None = None,
        left_on: str | list[str] | None = None,
        right_on: str | list[str] | None = None,
        suffixes: tuple[str, str] = ("_x", "_y"),
        validate: str | None = None,
    ) -> Callable[[Frame], Frame]:
        def _op(frame: Frame) -> Frame:
            return frame.inner_join(
                other,
                on=on,
                left_on=left_on,
                right_on=right_on,
                suffixes=suffixes,
                validate=validate,
            )

        return _op


pipe = PipeNamespace()
