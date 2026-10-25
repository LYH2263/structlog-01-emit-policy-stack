# SPDX-License-Identifier: MIT OR Apache-2.0
# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the MIT License.  See the LICENSE file in the root of this
# repository for complete details.

import asyncio
import contextvars
import logging
import threading

from concurrent.futures import ThreadPoolExecutor

import pytest

import structlog

from structlog import emit_policy as ep
from structlog.contextvars import (
    bind_contextvars,
    clear_contextvars,
    get_contextvars,
    merge_contextvars,
)
from structlog.exceptions import DropEvent
from structlog.processors import KeyValueRenderer
from structlog.stdlib import (
    AsyncBoundLogger,
    ProcessorFormatter,
)
from structlog.testing import CapturingLogger, capture_logs

from .helpers import call_recorder


@pytest.fixture(autouse=True)
def _clear_contextvars():
    clear_contextvars()


@pytest.fixture(name="cl")
def _cl():
    return CapturingLogger()


@pytest.fixture(name="info_logger")
def _info_logger(cl):
    return structlog.make_filtering_bound_logger(logging.INFO)(
        cl, [merge_contextvars], {}
    )


# ---------------------------------------------------------------------------
# Stack mechanics
# ---------------------------------------------------------------------------


class TestStack:
    def test_empty_stack_is_pin(self):
        """
        Without pushed frames the merged view is all-pin and reads like the
        legacy defaults.
        """
        assert ep.get_level_gate() is None
        assert ep.ContextMerge.SETDEFAULT == ep.get_context_merge()
        assert ep.StdlibRoute.PIN == ep.get_stdlib_route()

        view = ep.get_effective_policy()
        assert ep.ContextMerge.SETDEFAULT == view.context_merge
        assert ep.LevelGate.PIN == view.level_gate
        assert logging.NOTSET == view.min_level
        assert ep.StdlibRoute.PIN == view.stdlib_route

    def test_push_pop_token_restores(self):
        """
        Imperative push/pop with a token restores the previous state.
        """
        token = ep.push_emit_policy(context_merge="bound_only")

        assert ep.ContextMerge.BOUND_ONLY == ep.get_context_merge()

        ep.pop_emit_policy(token)

        assert ep.ContextMerge.SETDEFAULT == ep.get_context_merge()
        assert ep.get_level_gate() is None

    def test_context_manager_unwinds_on_exception(self):
        """
        A frame is popped even when the with-block raises.
        """
        assert ep.get_level_gate() is None
        with (
            pytest.raises(RuntimeError),
            ep.emit_policy(level_gate="force_emit"),
        ):
            raise RuntimeError("boom")

        assert ep.get_level_gate() is None

    def test_nested_partial_frames_inherit_and_restore(self):
        """
        Unspecified dimensions inherit from the frame below; popping an
        inner frame restores the outer frame; an explicitly pin-valued
        inner frame overrides the outer one.
        """
        with ep.emit_policy(
            context_merge="bound_only",
            level_gate="raise_min",
            min_level=logging.ERROR,
        ):
            with ep.emit_policy(context_merge="context_wins"):
                view = ep.get_effective_policy()
                assert ep.ContextMerge.CONTEXT_WINS == view.context_merge
                # Inherited from the outer frame.
                assert ep.LevelGate.RAISE_MIN == view.level_gate
                assert logging.ERROR == view.min_level

            view = ep.get_effective_policy()
            assert ep.ContextMerge.BOUND_ONLY == view.context_merge
            assert ep.LevelGate.RAISE_MIN == view.level_gate

            with ep.emit_policy(level_gate="pin"):
                assert ep.LevelGate.PIN == ep.get_level_gate()[0]

            # Outer gate is back after the inner pin frame.
            assert ep.LevelGate.RAISE_MIN == ep.get_level_gate()[0]

        assert ep.get_effective_policy().level_gate is ep.LevelGate.PIN

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"context_merge": "nope"},
            {"level_gate": "nope"},
            {"stdlib_route": "nope"},
        ],
    )
    def test_invalid_value_rejected(self, kwargs):
        with (
            pytest.raises(ValueError, match="is not a valid"),
            ep.emit_policy(**kwargs),
        ):
            pass

    def test_stack_does_not_leak_into_contextvars(self, info_logger, cl):
        """
        The backing ContextVars live outside the structlog_ user-key
        namespace and must never appear in merged event dicts.
        """
        bind_contextvars(user_key=42)
        with ep.emit_policy(
            context_merge="context_wins",
            level_gate="raise_min",
            min_level=logging.ERROR,
            stdlib_route="foreign_skip_context",
        ):
            # Exactly the user's own keys are visible; none of the internal
            # stack or projection variables show up.
            assert {"user_key": 42} == get_contextvars()
            # ERROR passes both the wrapper threshold (INFO) and the frame
            # floor; INFO would now be suppressed by the raise_min gate.
            info_logger.error("hi")

        assert {"event": "hi", "user_key": 42} == cl.calls[0].kwargs

    def test_dimension_projections_are_independent(self):
        """
        Frames touching different dimensions do not shadow each other.
        """
        with ep.emit_policy(context_merge="bound_only"):
            with ep.emit_policy(stdlib_route="structlog_passthrough"):
                assert ep.ContextMerge.BOUND_ONLY is ep.get_context_merge()
                assert (
                    ep.StdlibRoute.STRUCTLOG_PASSTHROUGH
                    is ep.get_stdlib_route()
                )
                assert ep.get_level_gate() is None

            # The inner frame only owned the route dimension.
            assert ep.StdlibRoute.PIN is ep.get_stdlib_route()
            assert ep.ContextMerge.BOUND_ONLY is ep.get_context_merge()

        assert ep.ContextMerge.SETDEFAULT is ep.get_context_merge()
        assert ep.StdlibRoute.PIN is ep.get_stdlib_route()

    def test_copied_context_carries_all_three_projections(self):
        """
        copy_context() snapshots the stack history and every per-dimension
        projection together; the executor handoff therefore observes all
        three dimensions at push time, not just the level gate.
        """
        with ep.emit_policy(
            context_merge="bound_only",
            level_gate="raise_min",
            min_level=logging.ERROR,
            stdlib_route="foreign_skip_context",
        ):
            ctx = contextvars.copy_context()

        # The outer context is fully back to pin after pop.
        assert ep.ContextMerge.SETDEFAULT is ep.get_context_merge()
        assert ep.get_level_gate() is None
        assert ep.StdlibRoute.PIN is ep.get_stdlib_route()

        def assert_snapshot():
            assert ep.ContextMerge.BOUND_ONLY is ep.get_context_merge()
            assert (
                ep.LevelGate.RAISE_MIN,
                logging.ERROR,
            ) == ep.get_level_gate()
            assert ep.StdlibRoute.FOREIGN_SKIP_CONTEXT is ep.get_stdlib_route()
            view = ep.get_effective_policy()
            assert ep.ContextMerge.BOUND_ONLY is view.context_merge
            assert ep.LevelGate.RAISE_MIN is view.level_gate
            assert logging.ERROR == view.min_level
            assert ep.StdlibRoute.FOREIGN_SKIP_CONTEXT is view.stdlib_route

        ctx.run(assert_snapshot)


# ---------------------------------------------------------------------------
# Exit 1: the context-merge point (merge_contextvars / equivalents)
# ---------------------------------------------------------------------------


class TestContextMergeExit:
    def test_pin_is_setdefault(self):
        bind_contextvars(a=1, b=2)

        assert {"a": 1, "b": 99, "c": 3} == merge_contextvars(
            None, "info", {"b": 99, "c": 3}
        )

    def test_context_wins_overrides_event_and_bound(self):
        bind_contextvars(a=1, b=2)

        with ep.emit_policy(context_merge="context_wins"):
            assert {"a": 1, "b": 2, "c": 3} == merge_contextvars(
                None, "info", {"b": 99, "c": 3}
            )

    def test_bound_only_skips_merge(self):
        bind_contextvars(a=1, b=2)

        with ep.emit_policy(context_merge="bound_only"):
            assert {"b": 99, "c": 3} == merge_contextvars(
                None, "info", {"b": 99, "c": 3}
            )

    def test_pop_restores_setdefault(self):
        bind_contextvars(a=1)

        with ep.emit_policy(context_merge="bound_only"):
            assert {} == merge_contextvars(None, "info", {})

        assert {"a": 1} == merge_contextvars(None, "info", {})

    def test_end_to_end_through_processor_chain(self, info_logger, cl):
        bind_contextvars(a=1)

        with ep.emit_policy(context_merge="bound_only"):
            info_logger.info("inside")
        info_logger.info("outside")

        assert "a" not in cl.calls[0].kwargs
        assert 1 == cl.calls[1].kwargs["a"]

    def test_equivalent_merge_point_uses_same_policy(self):
        @ep.mark_context_merge_processor
        def custom_merge(logger, name, event_dict):
            return ep.apply_context_merge(event_dict)

        assert ep.is_context_merge_processor(custom_merge)
        assert ep.is_context_merge_processor(merge_contextvars)

        bind_contextvars(a=1)
        with ep.emit_policy(context_merge="context_wins"):
            assert {"a": 1} == custom_merge(None, "info", {"a": 99})
        with ep.emit_policy(context_merge="bound_only"):
            assert {"a": 99} == custom_merge(None, "info", {"a": 99})


# ---------------------------------------------------------------------------
# Exit 2: the FilteringBoundLogger method-level gate
# ---------------------------------------------------------------------------


class TestLevelGateExit:
    def test_pin_frame_is_identical_to_empty_stack(self, cl):
        bl = structlog.make_filtering_bound_logger(logging.INFO)(cl, [], {})

        with ep.emit_policy(level_gate="pin"):
            bl.debug("nope")
            bl.info("yep")

        assert [("info", (), {"event": "yep"})] == cl.calls

    def test_raise_min_takes_stricter_floor(self, cl):
        bl = structlog.make_filtering_bound_logger(logging.INFO)(cl, [], {})

        with ep.emit_policy(level_gate="raise_min", min_level=logging.WARNING):
            bl.debug("d")
            bl.info("i")
            bl.warning("w")
            bl.error("e")
            assert not bl.is_enabled_for(logging.INFO)
            assert bl.is_enabled_for(logging.WARNING)
            assert logging.WARNING == bl.get_effective_level()

        assert ["warning", "error"] == [c.method_name for c in cl.calls]
        assert logging.INFO == bl.get_effective_level()

    def test_raise_min_below_configured_threshold_is_noop(self, cl):
        bl = structlog.make_filtering_bound_logger(logging.WARNING)(cl, [], {})

        with ep.emit_policy(level_gate="raise_min", min_level=logging.DEBUG):
            bl.info("still-nope")
            bl.warning("yep")

        assert ["warning"] == [c.method_name for c in cl.calls]

    def test_force_emit_bypasses_nop_but_keeps_chain(self, cl):
        seen = []

        def recorder(logger, name, event_dict):
            seen.append(event_dict["event"])
            return event_dict

        bl = structlog.make_filtering_bound_logger(logging.INFO)(
            cl, [recorder], {}
        )

        with ep.emit_policy(level_gate="force_emit"):
            bl.debug("forced")
            assert bl.is_enabled_for(logging.DEBUG)
            assert logging.NOTSET == bl.get_effective_level()

        bl.debug("nope-again")

        assert ["forced"] == seen
        assert [("debug", (), {"event": "forced"})] == cl.calls

    def test_force_emit_does_not_change_drop_event(self, cl):
        """
        Bypassing the method-layer nop must not bypass DropEvent raised
        from within the processor chain.
        """

        def dropper(logger, name, event_dict):
            raise DropEvent

        bl = structlog.make_filtering_bound_logger(logging.INFO)(
            cl, [dropper], {}
        )

        with ep.emit_policy(level_gate="force_emit"):
            assert bl.debug("forced-but-dropped") is None

        assert cl.calls == []

    def test_nested_force_emit_overrides_outer_raise_min(self, cl):
        bl = structlog.make_filtering_bound_logger(logging.INFO)(cl, [], {})

        with ep.emit_policy(level_gate="raise_min", min_level=logging.ERROR):
            bl.info("outer-suppressed")
            with ep.emit_policy(level_gate="force_emit"):
                bl.debug("inner-forced")
            bl.info("outer-suppressed-again")

        assert [("debug", (), {"event": "inner-forced"})] == cl.calls

    def test_log_and_alog_honor_gate(self, cl):
        bl = structlog.make_filtering_bound_logger(logging.INFO)(cl, [], {})

        with ep.emit_policy(level_gate="raise_min", min_level=logging.WARNING):
            bl.log(logging.INFO, "nope")
            bl.log(logging.WARNING, "yep")

        assert ["warning"] == [c.method_name for c in cl.calls]

    async def test_alog_honors_gate(self, cl):
        bl = structlog.make_filtering_bound_logger(logging.INFO)(cl, [], {})

        with ep.emit_policy(level_gate="raise_min", min_level=logging.ERROR):
            await bl.ainfo("nope")
            await bl.alog(logging.ERROR, "yep")

        assert ["error"] == [c.method_name for c in cl.calls]

    async def test_async_methods_honor_raise_min(self, cl):
        bl = structlog.make_filtering_bound_logger(logging.INFO)(cl, [], {})

        with ep.emit_policy(level_gate="raise_min", min_level=logging.ERROR):
            await bl.ainfo("suppressed")
            await bl.aerror("emitted")

        assert [("error", (), {"event": "emitted"})] == cl.calls

    async def test_aexception_force_emit_through_executor(self, cl):
        """
        aexception copies the context for run_in_executor; the gate is
        read by error() inside the copied context.
        """
        bl = structlog.make_filtering_bound_logger(logging.CRITICAL)(
            cl, [], {}
        )

        with ep.emit_policy(level_gate="force_emit"):
            try:
                raise ValueError("boom")
            except ValueError:
                await bl.aexception("late-error")

        assert 1 == len(cl.calls)
        assert "error" == cl.calls[0].method_name
        assert isinstance(cl.calls[0].kwargs["exc_info"], tuple)

    def test_copied_context_pins_push_time_stack_top(self, cl):
        """
        A context copied while a frame is active keeps observing it after
        the frame has been popped -- this is what makes the executor path
        obey the push-time stack top.
        """
        bl = structlog.make_filtering_bound_logger(logging.INFO)(cl, [], {})
        token = ep.push_emit_policy(level_gate="force_emit")
        ctx = contextvars.copy_context()
        ep.pop_emit_policy(token)

        bl.debug("outside")
        ctx.run(lambda: bl.debug("inside-snapshot"))

        assert [("debug", (), {"event": "inside-snapshot"})] == cl.calls

    async def test_aexception_uses_snapshot_after_frame_popped(self, cl):
        """
        Even when the frame is popped before the executor worker runs,
        aexception's copied context still carries the push-time top.
        """
        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=1)
        loop.set_default_executor(executor)
        go = threading.Event()
        # Occupy the single worker so the aexception job is queued until we
        # have popped the frame.
        executor.submit(go.wait)

        bl = structlog.make_filtering_bound_logger(logging.CRITICAL)(
            cl, [], {}
        )
        token = ep.push_emit_policy(level_gate="force_emit")
        try:
            task = asyncio.create_task(bl.aexception("queued"))
            await asyncio.sleep(0.2)
        finally:
            ep.pop_emit_policy(token)

        go.set()
        await task

        assert 1 == len(cl.calls)
        assert "error" == cl.calls[0].method_name
        assert () == cl.calls[0].args
        assert "queued" == cl.calls[0].kwargs["event"]
        assert isinstance(cl.calls[0].kwargs["exc_info"], tuple)

    async def test_deprecated_async_bound_logger_executor_path(self, cl):
        """
        structlog.stdlib.AsyncBoundLogger dispatches via copy_context too;
        the context-merge policy must reach the worker thread.
        """

        seen = []

        def spy(logger, name, event_dict):
            seen.append(dict(event_dict))
            return (), event_dict

        async_bl = AsyncBoundLogger(cl, [merge_contextvars, spy], {})
        bind_contextvars(frag="context")

        with ep.emit_policy(context_merge="context_wins"):
            await async_bl.warning("async", frag="event")

        assert {"frag": "context", "event": "async"} == seen[0]
        assert 1 == len(cl.calls)
        # context_wins: the context-local value overrides the call's kwarg
        # all the way into the executor worker.
        assert "context" == cl.calls[0].kwargs["frag"]


# ---------------------------------------------------------------------------
# Exit 3: ProcessorFormatter.format foreign-record branch
# ---------------------------------------------------------------------------


def _foreign_record(msg="foreign message"):
    return logging.LogRecord(
        "foreign.logger", logging.WARNING, __file__, 0, msg, (), None
    )


def _structlog_record(event_dict):
    record = logging.LogRecord(
        "structlog.logger", logging.INFO, __file__, 0, event_dict, (), None
    )
    # Both attached by wrap_for_formatter.
    record._logger = logging.getLogger("structlog.logger")
    record._name = "info"
    return record


def _formatter(foreign_pre_chain, *, merge_in_processors=False):
    processors = []
    if merge_in_processors:
        processors.append(merge_contextvars)
    processors.extend(
        [ProcessorFormatter.remove_processors_meta, KeyValueRenderer()]
    )
    return ProcessorFormatter(
        processors=processors, foreign_pre_chain=foreign_pre_chain
    )


class TestStdlibRouteExit:
    def test_foreign_pin_merges_context(self):
        bind_contextvars(ctx_key="cv")
        formatter = _formatter([merge_contextvars])

        assert "ctx_key=" in formatter.format(_foreign_record())

    def test_foreign_skip_context_foreign_record(self):
        bind_contextvars(ctx_key="cv")
        formatter = _formatter([merge_contextvars])

        with ep.emit_policy(stdlib_route="foreign_skip_context"):
            rendered = formatter.format(_foreign_record())

        assert "ctx_key=" not in rendered
        assert "event='foreign message'" in rendered

    def test_foreign_skip_context_also_covers_shared_processors(self):
        bind_contextvars(ctx_key="cv")
        formatter = _formatter(None, merge_in_processors=True)

        pin = formatter.format(_foreign_record())
        with ep.emit_policy(stdlib_route="foreign_skip_context"):
            skipped = formatter.format(_foreign_record())

        assert "ctx_key=" in pin
        assert "ctx_key=" not in skipped

    def test_foreign_skip_context_leaves_structlog_record_alone(self):
        """
        The foreign-branch dimension must not kidnap structlog records:
        their emission-side merge results stay untouched.
        """
        bind_contextvars(ctx_key="cv")
        # Emulate the emission-side chain (frame not active at emission).
        event_dict = merge_contextvars(
            None, "info", {"event": "from structlog"}
        )
        formatter = _formatter([merge_contextvars])

        with ep.emit_policy(stdlib_route="foreign_skip_context"):
            rendered = formatter.format(_structlog_record(event_dict))
            foreign = formatter.format(_foreign_record())

        assert "ctx_key='cv'" in rendered
        assert "ctx_key=" not in foreign

    def test_structlog_passthrough_keeps_foreign_branch_pin(self):
        """
        structlog_passthrough names the structlog branch only; foreign
        records keep their pin routing (foreign_pre_chain incl. the merge
        still runs).
        """
        bind_contextvars(ctx_key="cv")
        recorder = call_recorder(lambda logger, name, event_dict: event_dict)
        formatter = _formatter([recorder, merge_contextvars])

        event_dict = {"event": "from structlog"}
        with ep.emit_policy(stdlib_route="structlog_passthrough"):
            structlog_out = formatter.format(_structlog_record(event_dict))
            assert 0 == recorder.call_count
            foreign_out = formatter.format(_foreign_record())

        assert 1 == recorder.call_count
        assert "ctx_key=" not in structlog_out
        assert "ctx_key='cv'" in foreign_out

    def test_marked_equivalent_processor_is_skipped_for_foreign(self):
        bind_contextvars(ctx_key="cv")

        @ep.mark_context_merge_processor
        def custom_merge(logger, name, event_dict):
            return ep.apply_context_merge(event_dict)

        formatter = _formatter([custom_merge])

        assert "ctx_key=" in formatter.format(_foreign_record())
        with ep.emit_policy(stdlib_route="foreign_skip_context"):
            rendered = formatter.format(_foreign_record())
        assert "ctx_key=" not in rendered

    def test_route_restored_after_pop(self):
        bind_contextvars(ctx_key="cv")
        formatter = _formatter([merge_contextvars])

        with ep.emit_policy(stdlib_route="foreign_skip_context"):
            assert "ctx_key=" not in formatter.format(_foreign_record())

        assert "ctx_key=" in formatter.format(_foreign_record())


# ---------------------------------------------------------------------------
# Cross-cutting guarantees
# ---------------------------------------------------------------------------


class TestCrossCutting:
    def test_capture_logs_respects_level_gate(self, cl):
        """
        capture_logs swaps processors only; the stack keeps constraining
        the method-level gate without having to be turned off.
        """
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(
                logging.WARNING
            )
        )
        log = structlog.get_logger()

        with (
            ep.emit_policy(level_gate="raise_min", min_level=logging.ERROR),
            capture_logs() as entries,
        ):
            log.warning("suppressed-by-stack")
            log.error("captured")

        assert ["captured"] == [e["event"] for e in entries]

    def test_capture_logs_respects_context_merge(self):
        bind_contextvars(ctx_key="cv")

        with (
            ep.emit_policy(context_merge="bound_only"),
            capture_logs(processors=[merge_contextvars]) as entries,
        ):
            structlog.get_logger().info("hi")

        assert "ctx_key" not in entries[0]

        with capture_logs(processors=[merge_contextvars]) as entries:
            structlog.get_logger().info("hi")
        assert "cv" == entries[0]["ctx_key"]

    def test_capture_logs_with_force_emit(self):
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(
                logging.WARNING
            )
        )

        with (
            ep.emit_policy(level_gate="force_emit"),
            capture_logs() as entries,
        ):
            structlog.get_logger().debug("forced-into-capture")

        assert ["forced-into-capture"] == [e["event"] for e in entries]

    def test_two_threads_do_not_share_stack(self):
        clear_contextvars()
        bind_contextvars(frag="main")
        results = {}
        errors = []
        barrier = threading.Barrier(2, timeout=5)

        def worker(mode, value):
            try:
                with ep.emit_policy(context_merge=mode):
                    bind_contextvars(frag=value)
                    barrier.wait()
                    merged = merge_contextvars(None, "info", {"frag": "bound"})
                    results[value] = merged["frag"]
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=worker, args=("context_wins", "thread-1"))
        t2 = threading.Thread(target=worker, args=("bound_only", "thread-2"))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors
        assert {"thread-1": "thread-1", "thread-2": "bound"} == results
        # The main thread's context is untouched by both pushes.
        assert ep.get_context_merge() is ep.ContextMerge.SETDEFAULT
        assert {"frag": "main"} == merge_contextvars(None, "info", {})

    async def test_two_tasks_do_not_share_stack(self):
        clear_contextvars()

        async def worker(mode, value):
            with ep.emit_policy(context_merge=mode):
                bind_contextvars(frag=value)
                # Interleave while both frames are active.
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                return merge_contextvars(None, "info", {"frag": "bound"})[
                    "frag"
                ]

        r1, r2 = await asyncio.gather(
            worker("context_wins", "task-1"),
            worker("bound_only", "task-2"),
        )

        assert "task-1" == r1
        assert "bound" == r2
        assert ep.get_context_merge() is ep.ContextMerge.SETDEFAULT
