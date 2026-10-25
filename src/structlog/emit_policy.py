# SPDX-License-Identifier: MIT OR Apache-2.0
# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the MIT License.  See the LICENSE file in the root of this
# repository for complete details.

"""
A stack of *emit-time policies*.

The stack is independent from both the process-wide defaults configured via
`structlog.configure` and the per-call key-values carried by
`structlog.BoundLogger.bind` / `structlog.contextvars.bound_contextvars`.

Each frame of the stack expresses up to three independent policy dimensions.
Only the dimensions a frame explicitly sets take effect; the others are
inherited from the frame below (and ultimately from the *pin* defaults, which
make every exit behave exactly as without a stack):

- ``context_merge`` decides how the structlog contextvars are merged into an
  ``event_dict`` at the context-merge point (i.e. the
  `structlog.contextvars.merge_contextvars` processor or an equivalent
  processor):

  - ``setdefault`` -- the *pin*: context-local keys only fill keys that are
    not already present in the ``event_dict``.
  - ``context_wins`` -- context-local keys override same-named event and
    bound keys.
  - ``bound_only`` -- context-local keys are not merged at all for this
    frame.

- ``level_gate`` decides what happens at the method-level gate of a
  `structlog.typing.FilteringBoundLogger` (in other words, the point where
  calls below the configured minimum level become no-ops):

  - ``pin`` -- the wrapper class' compiled threshold is used unchanged.
  - ``raise_min`` -- the effective minimum level is the stricter of the
    frame's *min_level* floor and the wrapper class' threshold.
  - ``force_emit`` -- the method-level no-op is bypassed; the event still
    runs the complete processor chain and processors raising
    `structlog.DropEvent` still drop the event.

- ``stdlib_route`` decides how `structlog.stdlib.ProcessorFormatter` routes
  records.  The two branches are evaluated independently and a frame can
  only influence the branch it names:

  - ``pin`` -- the existing fork between structlog and foreign
    `logging.LogRecord` objects is kept.
  - ``foreign_skip_context`` -- foreign records do not pass through a step
    that merges structlog contextvars on the formatter side.  Structlog
    records are not affected by this dimension.
  - ``structlog_passthrough`` -- ``_from_structlog`` records keep passing
    through the formatter without running the ``foreign_pre_chain`` while
    keeping the *pin* handling of wrapped records; foreign records keep
    their *pin* routing.

The stack is stored in a `contextvars.ContextVar`, so concurrently running
asyncio tasks and threads never share frames.  Contexts copied via
`contextvars.copy_context` (like the ones the async log methods hand to
``run_in_executor``) keep observing the stack top that was active when the
context was copied.

.. versionadded:: 26.2.0
"""

from __future__ import annotations

import contextlib
import contextvars

from collections.abc import Iterator
from enum import Enum
from typing import Any, NamedTuple

from ._log_levels import NOTSET


__all__ = [
    "CONTEXT_MERGE_MARKER",
    "ContextMerge",
    "EmitPolicyFrame",
    "LevelGate",
    "StdlibRoute",
    "apply_context_merge",
    "emit_policy",
    "foreign_context_merge_skipped",
    "get_context_merge",
    "get_effective_policy",
    "get_level_gate",
    "get_stdlib_route",
    "is_context_merge_processor",
    "mark_context_merge_processor",
    "pop_emit_policy",
    "push_emit_policy",
    "structlog_records_passthrough",
]


class ContextMerge(str, Enum):
    """
    How the context-local context is merged into an ``event_dict``.
    """

    SETDEFAULT = "setdefault"
    CONTEXT_WINS = "context_wins"
    BOUND_ONLY = "bound_only"


class LevelGate(str, Enum):
    """
    How the method-level gate of a filtering bound logger behaves.
    """

    PIN = "pin"
    RAISE_MIN = "raise_min"
    FORCE_EMIT = "force_emit"


class StdlibRoute(str, Enum):
    """
    How `structlog.stdlib.ProcessorFormatter` routes records.
    """

    PIN = "pin"
    FOREIGN_SKIP_CONTEXT = "foreign_skip_context"
    STRUCTLOG_PASSTHROUGH = "structlog_passthrough"


class EmitPolicyFrame(NamedTuple):
    """
    One immutable frame of the emit-time policy stack.

    Dimensions that are `None` are inherited from the frame below.

    Attributes:
        context_merge: The context merge policy of the frame.

        level_gate: The method-level level gate policy of the frame.

        min_level: The floor used with `LevelGate.RAISE_MIN`.  It uses the
            numeric `logging` levels (for example ``logging.WARNING``).

        stdlib_route: The `structlog.stdlib.ProcessorFormatter` routing
            policy of the frame.
    """

    context_merge: ContextMerge | None = None
    level_gate: LevelGate | None = None
    min_level: int | None = None
    stdlib_route: StdlibRoute | None = None


# A tuple gives us copy-on-write semantics: pushing sets a new tuple in this
# context only, so asyncio tasks -- and contexts copied for thread pool
# executors -- observe exactly the frames that were active when the context
# was copied, without ever mutating frames visible to another context.
#
# The name deliberately does NOT start with the ``structlog_`` prefix: that
# namespace is scanned by get_contextvars()/merge_contextvars as user
# context keys, so the stack itself would leak into event dicts.
_EMIT_POLICY_STACK: contextvars.ContextVar[
    tuple[EmitPolicyFrame, ...] | None
] = contextvars.ContextVar("_structlog_emit_policy_stack", default=None)


# Push-time projections of the stack, one ContextVar per policy dimension.
# These make the *read* side, which runs on every gated log call, a single
# ContextVar lookup: each variable always already holds the topmost frame
# setting that dimension (or the pin default).  All four variables are
# mutated together, synchronously, inside push/pop, so any copied context
# sees a consistent snapshot.  The names share the underscore prefix for the
# same reason as _EMIT_POLICY_STACK above.
_CONTEXT_MERGE_VAR: contextvars.ContextVar[ContextMerge] = (
    contextvars.ContextVar(
        "_structlog_emit_policy_context_merge",
        default=ContextMerge.SETDEFAULT,
    )
)
_LEVEL_GATE_VAR: contextvars.ContextVar[tuple[LevelGate, int] | None] = (
    contextvars.ContextVar("_structlog_emit_policy_level_gate", default=None)
)
_STDLIB_ROUTE_VAR: contextvars.ContextVar[StdlibRoute] = (
    contextvars.ContextVar(
        "_structlog_emit_policy_stdlib_route",
        default=StdlibRoute.PIN,
    )
)


class _EmitPolicyTokens(NamedTuple):
    # One token per ContextVar touched by a single push, in reset order.
    stack: contextvars.Token[Any]
    context_merge: contextvars.Token[Any]
    level_gate: contextvars.Token[Any]
    stdlib_route: contextvars.Token[Any]


# Attribute marking a processor as an equivalent context-merge point.
CONTEXT_MERGE_MARKER = "__structlog_merges_contextvars__"


def _coerce_context_merge(value: ContextMerge | str | None) -> ContextMerge:
    return value if isinstance(value, ContextMerge) else ContextMerge(value)


def _coerce_level_gate(value: LevelGate | str) -> LevelGate:
    return value if isinstance(value, LevelGate) else LevelGate(value)


def _coerce_stdlib_route(value: StdlibRoute | str) -> StdlibRoute:
    return value if isinstance(value, StdlibRoute) else StdlibRoute(value)


def push_emit_policy(
    *,
    context_merge: ContextMerge | str | None = None,
    level_gate: LevelGate | str | None = None,
    min_level: int | None = None,
    stdlib_route: StdlibRoute | str | None = None,
) -> _EmitPolicyTokens:
    """
    Push one frame onto the emit-time policy stack of the current context.

    Every argument is keyword-only and optional; dimensions that are not
    given are inherited from the frame below.

    Args:
        context_merge: One of the `ContextMerge` values (or its string
            value).

        level_gate: One of the `LevelGate` values (or its string value).

        min_level: Numeric `logging` level used as the floor when
            *level_gate* is `LevelGate.RAISE_MIN`.

        stdlib_route: One of the `StdlibRoute` values (or its string
            value).

    Returns:
        An opaque token bag that must be passed to `pop_emit_policy` to
        remove the frame.  Prefer the `emit_policy` context manager which
        guarantees that the frame is popped even if an exception is raised.

    .. versionadded:: 26.2.0
    """
    frame = EmitPolicyFrame(
        context_merge=(
            None
            if context_merge is None
            else _coerce_context_merge(context_merge)
        ),
        level_gate=None
        if level_gate is None
        else _coerce_level_gate(level_gate),
        min_level=min_level,
        stdlib_route=None
        if stdlib_route is None
        else _coerce_stdlib_route(stdlib_route),
    )

    # Compute the per-dimension projections of the new stack from the
    # projections of the current one: either this frame sets a dimension or
    # the effective value is inherited from below.
    gate = _LEVEL_GATE_VAR.get()
    if frame.level_gate is not None:
        gate = (
            frame.level_gate,
            NOTSET if frame.min_level is None else frame.min_level,
        )

    stack = _EMIT_POLICY_STACK.get()

    # Set the history first and the projections afterwards; the four sets
    # are synchronous so no observer can see a half-updated state.
    stack_token = _EMIT_POLICY_STACK.set((stack or ()) + (frame,))
    merge_token = _CONTEXT_MERGE_VAR.set(
        frame.context_merge
        if frame.context_merge is not None
        else _CONTEXT_MERGE_VAR.get()
    )
    gate_token = _LEVEL_GATE_VAR.set(gate)
    route_token = _STDLIB_ROUTE_VAR.set(
        frame.stdlib_route
        if frame.stdlib_route is not None
        else _STDLIB_ROUTE_VAR.get()
    )

    return _EmitPolicyTokens(
        stack=stack_token,
        context_merge=merge_token,
        level_gate=gate_token,
        stdlib_route=route_token,
    )


def pop_emit_policy(token: _EmitPolicyTokens) -> None:
    """
    Remove the frame that was pushed with *token*.

    Raises:
        ValueError: If *token* was created in another context or the stack
            has been modified since the token was created.  The
            `emit_policy` context manager handles this for you.

    .. versionadded:: 26.2.0
    """
    # Reset in the reverse order of push.
    _STDLIB_ROUTE_VAR.reset(token.stdlib_route)
    _LEVEL_GATE_VAR.reset(token.level_gate)
    _CONTEXT_MERGE_VAR.reset(token.context_merge)
    _EMIT_POLICY_STACK.reset(token.stack)


@contextlib.contextmanager
def emit_policy(
    *,
    context_merge: ContextMerge | str | None = None,
    level_gate: LevelGate | str | None = None,
    min_level: int | None = None,
    stdlib_route: StdlibRoute | str | None = None,
) -> Iterator[None]:
    """
    Push an `EmitPolicyFrame` for the duration of the ``with`` block.

    The frame is popped again when the block is left, including when an
    exception propagates through it.  Only the dimensions passed as keyword
    arguments are set on the frame; all others are inherited from the frame
    below.

    .. versionadded:: 26.2.0
    """
    token = push_emit_policy(
        context_merge=context_merge,
        level_gate=level_gate,
        min_level=min_level,
        stdlib_route=stdlib_route,
    )
    try:
        yield
    finally:
        pop_emit_policy(token)


def get_context_merge() -> ContextMerge:
    """
    Return the effective `ContextMerge` of the current context.

    This is the projection read by the context-merge point
    (`structlog.contextvars.merge_contextvars` and processors marked as
    equivalent merge points).  With an empty stack it returns
    `ContextMerge.SETDEFAULT`, which is the behavior of
    `structlog.contextvars.merge_contextvars` without a stack.

    .. versionadded:: 26.2.0
    """
    return _CONTEXT_MERGE_VAR.get()


def get_level_gate() -> tuple[LevelGate, int] | None:
    """
    Return the effective method-level level gate and its floor.

    This is the projection read by the method-level gate of filtering bound
    loggers.  It returns `None` when no frame sets a gate, letting the
    wrapper class' compiled threshold be the only check.  Otherwise it
    returns the topmost frame's gate together with that frame's
    ``min_level`` floor (`NOTSET` if none was given).

    .. versionadded:: 26.2.0
    """
    return _LEVEL_GATE_VAR.get()


def get_stdlib_route() -> StdlibRoute:
    """
    Return the effective `StdlibRoute` of the current context.

    With an empty stack it returns `StdlibRoute.PIN`.

    .. versionadded:: 26.2.0
    """
    return _STDLIB_ROUTE_VAR.get()


def get_effective_policy() -> EmitPolicyFrame:
    """
    Return the merged view of the stack as one `EmitPolicyFrame`.

    Every dimension shows the value from the topmost frame that sets it, or
    the *pin* default when no frame sets it.  This is a read-only view for
    introspection; the individual emission exits read their own projection
    directly.

    .. versionadded:: 26.2.0
    """
    gate = get_level_gate()

    return EmitPolicyFrame(
        context_merge=get_context_merge(),
        level_gate=LevelGate.PIN if gate is None else gate[0],
        min_level=NOTSET if gate is None else gate[1],
        stdlib_route=get_stdlib_route(),
    )


def foreign_context_merge_skipped() -> bool:
    """
    Return whether *foreign* records skip the context-merge step.

    This projection is read exclusively by the foreign-record branch of
    `structlog.stdlib.ProcessorFormatter.format`.

    .. versionadded:: 26.2.0
    """
    return get_stdlib_route() is StdlibRoute.FOREIGN_SKIP_CONTEXT


def structlog_records_passthrough() -> bool:
    """
    Return whether ``_from_structlog`` records keep passing through.

    This projection is read exclusively by the structlog-record branch of
    `structlog.stdlib.ProcessorFormatter.format`.  It is independent from
    `foreign_context_merge_skipped`: a frame that changes the treatment of
    foreign records never changes the treatment of structlog records and
    vice versa.

    .. versionadded:: 26.2.0
    """
    route = get_stdlib_route()

    return (
        route is StdlibRoute.PIN or route is StdlibRoute.STRUCTLOG_PASSTHROUGH
    )


def mark_context_merge_processor(processor: Any) -> Any:
    """
    Mark *processor* as an equivalent structlog-contextvars merge point.

    Processors marked this way are skipped on foreign records when
    `StdlibRoute.FOREIGN_SKIP_CONTEXT` is active, just like
    `structlog.contextvars.merge_contextvars` itself.  Can be used as a
    decorator.

    .. versionadded:: 26.2.0
    """
    setattr(processor, CONTEXT_MERGE_MARKER, True)

    return processor


def is_context_merge_processor(processor: Any) -> bool:
    """
    Return whether *processor* merges the structlog contextvars.

    This is true for `structlog.contextvars.merge_contextvars` and for
    processors marked with `mark_context_merge_processor`.

    .. versionadded:: 26.2.0
    """
    if getattr(processor, CONTEXT_MERGE_MARKER, False):
        return True

    # Imported lazily to keep this module free of import cycles.
    from .contextvars import merge_contextvars

    return processor is merge_contextvars


def apply_context_merge(event_dict: Any) -> Any:
    """
    Merge the structlog contextvars into *event_dict* using the stack.

    This is the shared body of the context-merge exit:
    `structlog.contextvars.merge_contextvars` delegates here and custom
    equivalent merge points can do the same.  With an empty stack the
    behavior is identical to calling
    `structlog.contextvars.merge_contextvars` directly.

    .. versionadded:: 26.2.0
    """
    # Imported lazily to keep this module free of import cycles.
    from .contextvars import get_contextvars

    mode = get_context_merge()
    if mode is ContextMerge.BOUND_ONLY:
        return event_dict

    context = get_contextvars()
    if mode is ContextMerge.CONTEXT_WINS:
        event_dict.update(context)
    else:
        for key, value in context.items():
            event_dict.setdefault(key, value)

    return event_dict
