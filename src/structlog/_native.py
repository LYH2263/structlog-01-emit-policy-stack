# SPDX-License-Identifier: MIT OR Apache-2.0
# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the MIT License.  See the LICENSE file in the root of this
# repository for complete details.

"""
structlog's native high-performance loggers.
"""

from __future__ import annotations

import asyncio
import collections
import contextvars
import sys
import threading

from collections.abc import Callable
from typing import Any

from ._base import BoundLoggerBase
from ._log_levels import (
    CRITICAL,
    DEBUG,
    ERROR,
    INFO,
    LEVEL_TO_NAME,
    NAME_TO_LEVEL,
    NOTSET,
    WARNING,
)
from .contextvars import _ASYNC_CALLING_STACK, _ASYNC_CALLING_THREAD

# The method-level gate reads the gate projection ContextVar directly: this
# is the hottest path in structlog (every suppressed log call ends here), so
# it must be one ContextVar lookup with no extra function call on the
# empty-stack fast path.
from .emit_policy import _LEVEL_GATE_VAR, LevelGate
from .typing import FilteringBoundLogger


def _gate_allows(
    gate: tuple[LevelGate, int], level: int, configured_min_level: int
) -> bool:
    """
    Apply a non-empty level-gate projection at the method layer.
    """
    mode, floor = gate
    if mode is LevelGate.FORCE_EMIT:
        # Bypass the method-layer no-op only; the event still runs the
        # full processor chain and DropEvent is unchanged.
        return True
    if mode is LevelGate.RAISE_MIN:
        return level >= configured_min_level and level >= floor

    # ``pin``: the wrapper class' compiled threshold.
    return level >= configured_min_level


def _level_allows(level: int, configured_min_level: int) -> bool:
    """
    Decide at the *method-level gate* whether *level* is emitted.

    This projection belongs to the filtering bound logger's method layer
    only; it does not bypass the processor chain and does not change
    `structlog.DropEvent` semantics.

    With no pushed frame (or a ``pin`` frame), the wrapper class' compiled
    threshold is the only check.  ``raise_min`` takes the stricter of the
    frame floor and the configured threshold, ``force_emit`` bypasses the
    method-level no-op.
    """
    gate = _LEVEL_GATE_VAR.get()
    if gate is None:
        return level >= configured_min_level

    return _gate_allows(gate, level, configured_min_level)


def _effective_min_level(configured_min_level: int) -> int:
    """
    Return the effective threshold exposed via ``get_effective_level()``.
    """
    gate = _LEVEL_GATE_VAR.get()
    if gate is None:
        return configured_min_level

    mode, floor = gate
    if mode is LevelGate.FORCE_EMIT:
        return NOTSET
    if mode is LevelGate.RAISE_MIN:
        return max(configured_min_level, floor)

    return configured_min_level


async def _run_in_executor_context(
    self: Any,
    emit: Callable[[], Any],
    executor: Any = None,
) -> Any:
    """
    Run the synchronous *emit* callable in an executor.

    The whole current context -- including the emit-time policy stack -- is
    copied before handing off, so the emission observes the stack top that
    was active when the async log method was called, even if an outer frame
    is popped before the executor runs.

    The caller frame is captured at two calls deep: this helper is invoked
    from the generated ``a*`` method (or ``aexception``), whose own caller
    is the user frame.
    """
    # Capture thread-specific info before handing off to the executor.
    thread_token = _ASYNC_CALLING_THREAD.set(
        (threading.get_ident(), threading.current_thread().name)
    )
    # This helper is awaited from the generated method (one frame up), so
    # the caller's frame is one step further out.
    method_frame = sys._getframe().f_back
    caller_frame = method_frame.f_back if method_frame is not None else None
    scs_token = _ASYNC_CALLING_STACK.set(caller_frame)  # type: ignore[arg-type]
    ctx = contextvars.copy_context()

    try:
        runner = await asyncio.get_running_loop().run_in_executor(
            executor, lambda: ctx.run(emit)
        )
    finally:
        _ASYNC_CALLING_STACK.reset(scs_token)
        _ASYNC_CALLING_THREAD.reset(thread_token)

    return runner


def exception(
    self: FilteringBoundLogger, event: str, *args: Any, **kw: Any
) -> Any:
    kw.setdefault("exc_info", True)

    return self.error(event, *args, **kw)


async def aexception(
    self: FilteringBoundLogger, event: str, *args: Any, **kw: Any
) -> Any:
    """
    .. versionchanged:: 23.3.0
       Callsite parameters are now also collected under asyncio.
    """
    # Exception info has to be extracted this early, because it is no longer
    # available once control is passed to the executor.
    if kw.get("exc_info", True) is True:
        kw["exc_info"] = sys.exc_info()

    # The level gate is consulted by the error() method inside the copied
    # context, so the executor observes the same stack top as the caller.
    return await _run_in_executor_context(
        self, lambda: self.error(event, *args, **kw)
    )


def make_filtering_bound_logger(
    min_level: int | str,
) -> type[FilteringBoundLogger]:
    """
    Create a new `FilteringBoundLogger` that only logs *min_level* or higher.

    The logger is optimized such that log levels below *min_level* only consist
    of a ``return None``.

    All familiar log methods are present, with async variants of each that are
    prefixed by an ``a``. Therefore, the async version of ``log.info("hello")``
    is ``await log.ainfo("hello")``.

    Additionally it has a ``log(self, level: int, **kw: Any)`` method to mirror
    `logging.Logger.log` and `structlog.stdlib.BoundLogger.log`.

    Compared to using *structlog*'s standard library integration and the
    `structlog.stdlib.filter_by_level` processor:

    - It's faster because once the logger is built at program start; it's a
      static class.
    - For the same reason you can't change the log level once configured. Use
      the dynamic approach of `standard-library` instead, if you need this
      feature.
    - You *can* have (much) more fine-grained filtering by :ref:`writing a
      simple processor <finer-filtering>`.

    Args:
        min_level:
            The log level as an integer. You can use the constants from
            `logging` like ``logging.INFO`` or pass the values directly. See
            `this table from the logging docs
            <https://docs.python.org/3/library/logging.html#levels>`_ for
            possible values.

            If you pass a string, it must be one of: ``critical``, ``error``,
            ``warning``, ``info``, ``debug``, ``notset`` (upper/lower case
            doesn't matter).

    .. versionadded:: 20.2.0
    .. versionchanged:: 21.1.0 The returned loggers are now pickleable.
    .. versionadded:: 20.1.0 The ``log()`` method.
    .. versionadded:: 22.2.0
       Async variants ``alog()``, ``adebug()``, ``ainfo()``, and so forth.
    .. versionchanged:: 25.1.0 *min_level* can now be a string.
    """
    if isinstance(min_level, str):
        min_level = NAME_TO_LEVEL[min_level.lower()]

    return LEVEL_TO_FILTERING_LOGGER[min_level]


def _maybe_interpolate(event: str, args: tuple[Any, ...]) -> str:
    """
    Interpolate the event string with the given arguments.

    If there's exactly one argument and it's a mapping, use it for dict-based
    interpolation. Otherwise, use the arguments for positional interpolation.
    """
    if not args:
        return event

    if (
        len(args) == 1
        and isinstance(args[0], collections.abc.Mapping)
        and args[0]
    ):
        return event % args[0]

    return event % args


def _make_filtering_bound_logger(min_level: int) -> type[FilteringBoundLogger]:
    """
    Create a new `FilteringBoundLogger` that only logs *min_level* or higher.

    The logger is optimized such that log levels below *min_level* only consist
    of a ``return None``.
    """

    # Bind the projection lookup as a closure cell to skip module globals on
    # the hot path.
    gate_get = _LEVEL_GATE_VAR.get

    def make_method(
        level: int,
    ) -> tuple[Callable[..., Any], Callable[..., Any]]:
        name = LEVEL_TO_NAME[level]

        def meth(self: Any, event: str, *args: Any, **kw: Any) -> Any:
            # The method-level gate reads the emit-time policy stack.  The
            # empty-stack fast path stays a single ContextVar lookup plus
            # the static comparison the class was compiled with; only when
            # a frame is active do we pay for the policy resolution.
            gate = gate_get()
            if gate is None:
                if level < min_level:
                    return None
            elif not _gate_allows(gate, level, min_level):
                return None

            return self._proxy_to_logger(
                name, _maybe_interpolate(event, args), **kw
            )

        async def ameth(self: Any, event: str, *args: Any, **kw: Any) -> Any:
            """
            .. versionchanged:: 23.3.0
                Callsite parameters are now also collected under asyncio.
            .. versionchanged:: 26.2.0
                The emit-time policy stack is honored, including from the
                context copied for the executor.
            """
            gate = gate_get()
            if gate is None:
                if level < min_level:
                    return None
            elif not _gate_allows(gate, level, min_level):
                return None

            event = _maybe_interpolate(event, args)

            await _run_in_executor_context(
                self, lambda: self._proxy_to_logger(name, event, **kw)
            )

        meth.__name__ = name
        ameth.__name__ = f"a{name}"

        return meth, ameth

    def log(self: Any, level: int, event: str, *args: Any, **kw: Any) -> Any:
        if not _level_allows(level, min_level):
            return None
        name = LEVEL_TO_NAME[level]

        return self._proxy_to_logger(
            name, _maybe_interpolate(event, args), **kw
        )

    async def alog(
        self: Any, level: int, event: str, *args: Any, **kw: Any
    ) -> Any:
        """
        .. versionchanged:: 23.3.0
           Callsite parameters are now also collected under asyncio.
        .. versionchanged:: 26.2.0
           The emit-time policy stack is honored, including from the
           context copied for the executor.
        """
        if not _level_allows(level, min_level):
            return None
        name = LEVEL_TO_NAME[level]
        event = _maybe_interpolate(event, args)

        return await _run_in_executor_context(
            self, lambda: self._proxy_to_logger(name, event, **kw)
        )

    meths: dict[str, Callable[..., Any]] = {"log": log, "alog": alog}
    for lvl, name in LEVEL_TO_NAME.items():
        meths[name], meths[f"a{name}"] = make_method(lvl)

    meths["exception"] = exception
    meths["aexception"] = aexception
    meths["fatal"] = meths["critical"]
    meths["afatal"] = meths["acritical"]
    meths["warn"] = meths["warning"]
    meths["awarn"] = meths["awarning"]
    meths["msg"] = meths["info"]
    meths["amsg"] = meths["ainfo"]

    # Introspection
    meths["is_enabled_for"] = lambda self, level: _level_allows(
        level, min_level
    )
    meths["get_effective_level"] = lambda self: _effective_min_level(min_level)

    return type(
        f"BoundLoggerFilteringAt{LEVEL_TO_NAME.get(min_level, 'Notset').capitalize()}",
        (BoundLoggerBase,),
        meths,
    )


# Pre-create all possible filters to make them pickleable.
BoundLoggerFilteringAtNotset = _make_filtering_bound_logger(NOTSET)
BoundLoggerFilteringAtDebug = _make_filtering_bound_logger(DEBUG)
BoundLoggerFilteringAtInfo = _make_filtering_bound_logger(INFO)
BoundLoggerFilteringAtWarning = _make_filtering_bound_logger(WARNING)
BoundLoggerFilteringAtError = _make_filtering_bound_logger(ERROR)
BoundLoggerFilteringAtCritical = _make_filtering_bound_logger(CRITICAL)

LEVEL_TO_FILTERING_LOGGER = {
    CRITICAL: BoundLoggerFilteringAtCritical,
    ERROR: BoundLoggerFilteringAtError,
    WARNING: BoundLoggerFilteringAtWarning,
    INFO: BoundLoggerFilteringAtInfo,
    DEBUG: BoundLoggerFilteringAtDebug,
    NOTSET: BoundLoggerFilteringAtNotset,
}
