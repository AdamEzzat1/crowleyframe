from __future__ import annotations

from typing import Any, Callable, List

from .api import Frame

_NOT_VERBS = {"from_pandas", "from_polars", "from_arrow"}


def _is_verb(name: str) -> bool:
    return not name.startswith("_") and name not in _NOT_VERBS and callable(getattr(Frame, name, None))


class PipeNamespace:
    """
    Pipeable versions of every Frame verb.

    `pipe.verb(...)` returns a callable that applies `frame.verb(...)`,
    so you can write:

        cf >> pipe.group_by("id") >> pipe.summarise(mean_x=("x", "mean"))
        cf >> pipe.filter("x > 0") >> pipe.arrange("x") >> pipe.collect()
    """

    def __getattr__(self, name: str) -> Callable[..., Callable[[Frame], Any]]:
        if not _is_verb(name):
            raise AttributeError(f"pipe has no verb {name!r}")

        def verb(*args: Any, **kwargs: Any) -> Callable[[Frame], Any]:
            def _op(frame: Frame) -> Any:
                return getattr(frame, name)(*args, **kwargs)

            _op.__name__ = name
            return _op

        verb.__name__ = name
        return verb

    def __dir__(self) -> List[str]:
        return sorted(name for name in dir(Frame) if _is_verb(name))


pipe = PipeNamespace()
