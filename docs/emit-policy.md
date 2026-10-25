(emit_policy)=

# Emit-time Policy Stack

```{testsetup}
import logging

import structlog
from structlog.contextvars import (
    bind_contextvars,
    clear_contextvars,
    merge_contextvars,
)

structlog.configure(processors=[structlog.processors.KeyValueRenderer()])
```

```{testcleanup}
import structlog

structlog.reset_defaults()
```

Sometimes you need to change *how a log event is shaped* for one section of your program -- middleware, a single request, or a noisy subsystem -- without changing process-wide configuration and without touching bound key-values.

Reconfiguring via {func}`structlog.configure` is global and hard to restore reliably; {func}`structlog.contextvars.bound_contextvars` only changes *which key-values* are merged, not *how* they are merged or how levels and foreign records are treated.

The {mod}`structlog.emit_policy` module adds an **emit-time policy stack** that is independent from both.
Each frame you push lives on the current {mod}`contextvars` context -- so it is scoped to the current asyncio task or thread, is copied along when log methods hand work to an executor, and never leaks into a concurrent task or thread.

```{versionadded} 26.2.0
```

You push frames with the {func}`~structlog.emit_policy.emit_policy` context manager (or the lower-level {func}`~structlog.emit_policy.push_emit_policy` / {func}`~structlog.emit_policy.pop_emit_policy` pair).
Frames are popped when the block exits, **including when an exception propagates through it**; nested frames inherit the dimensions they do not set, and popping an inner frame restores the outer one.
With no frame pushed, everything behaves exactly as before.

Every argument is keyword-only and every dimension is independent -- a frame may set any combination of them.

## Level gate

`level_gate` changes the method-level threshold of `structlog.typing.FilteringBoundLogger`:

- `"pin"` (default): the threshold the logger class was compiled with is the only check.
- `"raise_min"`: the effective minimum level is the *stricter* of the frame's `min_level` floor and the configured threshold.
- `"force_emit"`: bypass the method-layer no-op, while the event still runs the **full processor chain** -- `structlog.DropEvent` and processor-level filtering are unchanged.

```{doctest}
>>> from structlog.emit_policy import emit_policy

>>> cl = structlog.testing.CapturingLogger()
>>> log = structlog.make_filtering_bound_logger(logging.INFO)(cl, [], {})
>>> log.debug("suppressed")
>>> with emit_policy(level_gate="force_emit"):
...     log.debug("forced through the chain")
>>> [call.method_name for call in cl.calls]
['debug']
```

## Context merge

`context_merge` changes how {func}`structlog.contextvars.merge_contextvars` (and [processors marked as equivalent merge points](#equivalent-merge-points)) combines context-local keys with the event dict:

- `"setdefault"` (default): context keys fill in keys that are not already present, matching the long-standing behavior.
- `"context_wins"`: same-named context-local keys override event and bound values.
- `"bound_only"`: context-local keys are not merged into events for this frame.

```{doctest}
>>> clear_contextvars()
>>> bind_contextvars(user="anne")
>>> log = structlog.make_filtering_bound_logger(logging.INFO)(
...     cl, [merge_contextvars], {}
... )
>>> log.info("default: the event wins", user="bob")
>>> with emit_policy(context_merge="context_wins"):
...     log.info("scoped: the context wins", user="bob")
>>> cl.calls[-2].kwargs["user"]
'bob'
>>> cl.calls[-1].kwargs["user"]
'anne'
```

## Standard-library routing

`stdlib_route` changes how {class}`structlog.stdlib.ProcessorFormatter` routes records, and -- unlike the other dimensions -- acts **per branch** inside the formatter:

- `"pin"` (default): the existing fork between structlog records (wrapped via `ProcessorFormatter.wrap_for_formatter`) and foreign {class}`logging.LogRecord`s is kept.
- `"foreign_skip_context"`: *foreign* records must not pass through steps that would merge structlog contextvars into them on the formatter side.
  Structlog-source records are unaffected -- they already merged their context at emission time.
- `"structlog_passthrough"`: `_from_structlog` records keep skipping the foreign pre-chain inside the formatter, exactly as in the pinned behavior.

This dimension is read in {meth}`structlog.stdlib.ProcessorFormatter.format`; it does not change anything at emission time.

## Isolation and threads

The stack is plain {mod}`contextvars` state, so:

- Two concurrent tasks or threads that each push a frame do not see each other's frames.
- Async log methods (`await log.ainfo(...)`, including `aexception`) copy the context when they hand off to the executor; the emission observes the frame that was active **at call time**, even if it is popped before the worker runs.
- {func}`structlog.testing.capture_logs` keeps working while frames are active; the stack continues to constrain levels and merging while processors are temporarily replaced.

The stack itself never appears in your event dicts: its backing variables live outside the `structlog_` key namespace that {func}`structlog.contextvars.get_contextvars` scans.

(equivalent-merge-points)=

## Equivalent merge points

If your own processor plays the role of {func}`structlog.contextvars.merge_contextvars`, mark it with {func}`~structlog.emit_policy.mark_context_merge_processor` so that `bound_only`, `context_wins`, and `foreign_skip_context` apply to it as well, or delegate to {func}`~structlog.emit_policy.apply_context_merge` directly.

For introspection, {func}`~structlog.emit_policy.get_effective_policy` returns the merged view of the stack; the individual emission exits read their own projections.
