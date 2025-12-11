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

    This keeps the core verbs defined on Frame / GroupedFrame, while
    still allowing a tidy, pipe-friendly style.
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
        """
        Pipe-friendly wrapper around GroupedFrame.summarise.

        Example
        -------
        cf >> pipe.group_by("grp") >> pipe.summarise(
            mean_x=("x", "mean"),
            n=("x", "count"),
        )
        """
        def _op(grouped: GroupedFrame) -> Frame:
            return grouped.summarise(**agg_specs)

        return _op

    # -----------------------------
    # Frame-level operations
    # -----------------------------
    def filter(self, expr: str) -> Callable[[Frame], Frame]:
        def _op(frame: Frame) -> Frame:
            return frame.filter(expr=expr)

        return _op

    def arrange(self, *cols: str) -> Callable[[Frame], Frame]:
        def _op(frame: Frame) -> Frame:
            return frame.arrange(*cols)

        return _op

    def select(self, *selectors: Any) -> Callable[[Frame], Frame]:
        """
        Pipe-friendly wrapper around Frame.select.

        Example
        -------
        cf >> pipe.select("id", col.starts_with("user_"))
        """
        def _op(frame: Frame) -> Frame:
            return frame.select(*selectors)

        return _op

    def mutate(self, **kwargs: Any) -> Callable[[Frame], Frame]:
        """
        Pipe-friendly wrapper around Frame.mutate.

        Example
        -------
        cf >> pipe.mutate(z="(x - x.mean()) / x.std()")
        """
        def _op(frame: Frame) -> Frame:
            return frame.mutate(**kwargs)

        return _op


pipe = PipeNamespace()
