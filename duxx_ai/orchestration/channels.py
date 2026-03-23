"""Channel implementations for state management in StateGraph.

Channels define how state values are stored, updated, and merged during
parallel execution. Advanced channel system for typed state management.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Generic, TypeVar, get_type_hints

T = TypeVar("T")


class BaseChannel(Generic[T]):
    """Abstract base for all channels."""

    def __init__(self, typ: type[T] | None = None, default: T | None = None):
        self.typ = typ
        self.default = default
        self._value: T | None = default
        self._updated = False

    def get(self) -> T:
        if self._value is None and self.default is not None:
            return self.default
        return self._value  # type: ignore

    def update(self, value: T) -> None:
        self._value = value
        self._updated = True

    def checkpoint(self) -> Any:
        return copy.deepcopy(self._value)

    def from_checkpoint(self, data: Any) -> None:
        self._value = data
        self._updated = False

    def is_updated(self) -> bool:
        return self._updated

    def reset_updated(self) -> None:
        self._updated = False

    def copy(self) -> BaseChannel[T]:
        ch = self.__class__.__new__(self.__class__)
        ch.typ = self.typ
        ch.default = self.default
        ch._value = copy.deepcopy(self._value)
        ch._updated = self._updated
        return ch


class LastValue(BaseChannel[T]):
    """Stores only the last value written. Default channel type.

    Usage:
        class State(TypedDict):
            name: str  # Implicitly uses LastValue
    """

    def update(self, value: T) -> None:
        self._value = value
        self._updated = True


class Topic(BaseChannel[list[T]]):
    """Accumulates values into a list (like append_reducer).

    Usage:
        class State(TypedDict):
            messages: Annotated[list[str], Topic]
    """

    def __init__(self, typ: type[T] | None = None, default: list[T] | None = None):
        super().__init__(typ, default or [])
        if self._value is None:
            self._value = []

    def get(self) -> list[T]:
        return self._value or []

    def update(self, value: T | list[T]) -> None:
        if self._value is None:
            self._value = []
        if isinstance(value, list):
            self._value.extend(value)
        else:
            self._value.append(value)
        self._updated = True


class BinaryOperatorAggregate(BaseChannel[T]):
    """Applies a binary operator to combine old and new values.

    Usage:
        class State(TypedDict):
            total: Annotated[int, BinaryOperatorAggregate(operator.add)]
    """

    def __init__(self, operator: Callable[[T, T], T], typ: type[T] | None = None, default: T | None = None):
        super().__init__(typ, default)
        self.operator = operator

    def update(self, value: T) -> None:
        if self._value is not None:
            self._value = self.operator(self._value, value)
        else:
            self._value = value
        self._updated = True

    def copy(self) -> BinaryOperatorAggregate[T]:
        ch = BinaryOperatorAggregate(self.operator, self.typ, self.default)
        ch._value = copy.deepcopy(self._value)
        ch._updated = self._updated
        return ch


class EphemeralValue(BaseChannel[T]):
    """Stores a value from the preceding step, then clears it.
    Only available during the step that wrote it and the next step.

    Usage:
        class State(TypedDict):
            temp_result: Annotated[str, EphemeralValue]
    """

    def __init__(self, typ: type[T] | None = None):
        super().__init__(typ, None)
        self._step_written: int = -1
        self._current_step: int = 0

    def get(self) -> T | None:
        if self._current_step - self._step_written <= 1:
            return self._value
        return None

    def update(self, value: T) -> None:
        self._value = value
        self._step_written = self._current_step
        self._updated = True

    def advance_step(self) -> None:
        self._current_step += 1
        if self._current_step - self._step_written > 1:
            self._value = None


class AnyValue(BaseChannel[T]):
    """Stores last value, assumes all concurrent writes are identical.

    Usage:
        class State(TypedDict):
            shared_config: Annotated[dict, AnyValue]
    """
    pass  # Same as LastValue but semantically different


class NamedBarrierValue(BaseChannel[dict[str, T]]):
    """Waits for all named values before becoming available.

    Usage:
        barrier = NamedBarrierValue(names=["a", "b", "c"])
    """

    def __init__(self, names: list[str], typ: type[T] | None = None):
        super().__init__(typ, None)
        self.names = set(names)
        self._received: dict[str, T] = {}

    def update_named(self, name: str, value: T) -> None:
        self._received[name] = value
        if set(self._received.keys()) >= self.names:
            self._value = dict(self._received)
            self._updated = True

    def is_complete(self) -> bool:
        return set(self._received.keys()) >= self.names

    def get(self) -> dict[str, T] | None:
        if self.is_complete():
            return dict(self._received)
        return None


# ── Annotated reducer support ──

def merge_messages(left: list, right: list | Any, *, format: str | None = None) -> list:
    """Merge message lists intelligently 

    - If right message has same ID as left message, it replaces it
    - Otherwise right messages are appended
    - Handles single messages (wraps in list)
    """
    if not isinstance(right, list):
        right = [right]
    if not left:
        return list(right)

    # Build ID index of left messages
    result = list(left)
    left_ids = {}
    for i, msg in enumerate(result):
        msg_id = msg.get("id") if isinstance(msg, dict) else getattr(msg, "id", None)
        if msg_id:
            left_ids[msg_id] = i

    for msg in right:
        msg_id = msg.get("id") if isinstance(msg, dict) else getattr(msg, "id", None)
        if msg_id and msg_id in left_ids:
            result[left_ids[msg_id]] = msg  # Replace
        else:
            result.append(msg)  # Append

    return result


# ── Channel factory from type annotations ──

def channel_from_annotation(annotation: Any) -> BaseChannel:
    """Create a channel from a type annotation, supporting Annotated[type, Channel]."""
    import typing
    origin = getattr(annotation, "__origin__", None)

    # Handle Annotated[type, ChannelSpec]
    if origin is not None and str(origin) == "typing.Annotated":
        args = annotation.__args__
        base_type = args[0]
        for meta in args[1:]:
            if isinstance(meta, BaseChannel):
                return meta
            if callable(meta) and meta is merge_messages:
                return BinaryOperatorAggregate(add_messages, list, [])
            if isinstance(meta, type) and issubclass(meta, BaseChannel):
                return meta()
        return LastValue(base_type)

    # Plain type → LastValue
    if isinstance(annotation, type) and annotation is list:
        return LastValue(list, [])
    return LastValue()
