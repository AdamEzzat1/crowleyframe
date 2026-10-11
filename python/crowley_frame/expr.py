from __future__ import annotations

import difflib
import operator
from typing import Any, Callable, Iterable, Optional, Set, Tuple

import polars as pl


class CrowleyTranslationError(ValueError):
    """Raised when an expression or verb cannot be translated to a lazy Polars plan."""


TranslationError = CrowleyTranslationError  # earlier name, kept as an alias


class CrowleyValidationError(ValueError):
    """Raised when data fails a validation, expectation or key-integrity check."""

    def __init__(self, message: str, report: Any = None):
        super().__init__(message)
        self.report = report


class ColumnNotFoundError(KeyError):
    """Unknown column name. Subclasses KeyError for backwards compatibility."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


def column_not_found(name: Any, columns: Iterable[str]) -> ColumnNotFoundError:
    msg = f"Column {name!r} not found in DataFrame"
    close = difflib.get_close_matches(str(name), [str(c) for c in columns], n=1)
    if close:
        msg += f"; did you mean {close[0]!r}?"
    return ColumnNotFoundError(msg)


class Ctx:
    """
    What a verb tells an expression when compiling it.

    columns : known column names (None disables validation)
    schema  : column name -> Polars dtype of the input frame (for selectors)
    groups  : grouping columns of the frame
    window  : True in mutate/filter, where aggregates and window functions
              are evaluated per group via .over(); False in summarise.
    """

    __slots__ = ("columns", "schema", "groups", "window")

    def __init__(
        self,
        columns: Optional[Set[str]],
        groups: Tuple[str, ...] = (),
        window: bool = True,
        schema: Any = None,
    ):
        self.columns = columns
        self.schema = schema
        self.groups = groups
        self.window = window


def over(e: pl.Expr, ctx: Ctx) -> pl.Expr:
    if ctx.window and ctx.groups:
        return e.over(list(ctx.groups))
    return e


class Expr:
    """
    A deferred column expression. Compiles to a `pl.Expr` once the verb
    supplies its context, so nothing here ever touches row data.
    """

    __slots__ = ("_build", "_text", "_is_desc")

    def __init__(self, build: Callable[[Ctx], pl.Expr], text: str, is_desc: bool = False):
        self._build = build
        self._text = text
        self._is_desc = is_desc

    def _compile(self, ctx: Ctx) -> pl.Expr:
        return self._build(ctx)

    def __repr__(self) -> str:
        return f"<crowley expr: {self._text}>"

    def __bool__(self) -> bool:
        raise TypeError(
            f"The truth value of the expression {self._text} is ambiguous. "
            "Combine conditions with & | ~ (each side in parentheses), "
            "or use and/or/not inside a string expression."
        )

    def __iter__(self):
        raise TypeError(
            f"{self._text} is a column expression, not an iterable. "
            "Python builtins such as sum()/min()/max() do not work on it; "
            "use crowley_frame.functions (e.g. cr.sum) instead."
        )

    __hash__ = None  # type: ignore[assignment]

    # -- methods mirroring the free functions ---------------------------
    def is_in(self, values: Iterable[Any]) -> "Expr":
        vals = list(values)
        return Expr(lambda ctx: self._compile(ctx).is_in(vals), f"is_in({self._text}, {vals!r})")

    def is_na(self) -> "Expr":
        return Expr(lambda ctx: self._compile(ctx).is_null(), f"is_na({self._text})")

    def __neg__(self) -> "Expr":
        return Expr(lambda ctx: -self._compile(ctx), f"(-{self._text})")

    def __pos__(self) -> "Expr":
        return self

    def __invert__(self) -> "Expr":
        return Expr(lambda ctx: ~self._compile(ctx), f"(not {self._text})")


def lit(value: Any) -> Expr:
    """A literal value. Needed for strings, which otherwise name columns in function arguments."""
    return Expr(lambda ctx: pl.lit(value), repr(value))


def as_expr(value: Any) -> Expr:
    """Operand of an operator: anything that is not an expression is a literal."""
    if isinstance(value, Expr):
        return value
    return lit(value)


def _binop(op: Callable[[Any, Any], Any], sym: str, reverse: bool = False):
    def method(self: Expr, other: Any) -> Expr:
        o = as_expr(other)
        a, b = (o, self) if reverse else (self, o)
        return Expr(lambda ctx: op(a._compile(ctx), b._compile(ctx)), f"({a._text} {sym} {b._text})")

    return method


for _name, _op, _sym in [
    ("add", operator.add, "+"),
    ("sub", operator.sub, "-"),
    ("mul", operator.mul, "*"),
    ("truediv", operator.truediv, "/"),
    ("floordiv", operator.floordiv, "//"),
    ("mod", operator.mod, "%"),
    ("pow", operator.pow, "**"),
    ("and", operator.and_, "&"),
    ("or", operator.or_, "|"),
]:
    setattr(Expr, f"__{_name}__", _binop(_op, _sym))
    setattr(Expr, f"__r{_name}__", _binop(_op, _sym, reverse=True))

for _name, _op, _sym in [
    ("eq", operator.eq, "=="),
    ("ne", operator.ne, "!="),
    ("lt", operator.lt, "<"),
    ("le", operator.le, "<="),
    ("gt", operator.gt, ">"),
    ("ge", operator.ge, ">="),
]:
    setattr(Expr, f"__{_name}__", _binop(_op, _sym))


class ColumnExpr(Expr):
    """A single column. Usable both as an expression and as a selector."""

    __slots__ = ("kind", "value")

    def __init__(self, name: str):
        self.kind = "name"
        self.value = name

        def build(ctx: Ctx) -> pl.Expr:
            if ctx.columns is not None and name not in ctx.columns:
                raise column_not_found(name, ctx.columns)
            return pl.col(name)

        super().__init__(build, str(name))

    def _resolve(self, schema: Any) -> list:
        if self.value not in schema:
            raise column_not_found(self.value, schema)
        return [self.value]


def col_arg(value: Any) -> Expr:
    """Function argument where a column is expected: a string is a column name."""
    if isinstance(value, Expr):
        return value
    if isinstance(value, str):
        return ColumnExpr(value)
    return lit(value)
