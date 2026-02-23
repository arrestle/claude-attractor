# Issue #36 Follow-Up: Validation Rule R15 + Section 9.6 Event System

> **Execution:** Use the subagent-driven-development workflow to implement this plan.

**Goal:** Catch misconfigured hexagon nodes at validate-time (R15) and implement the full Section 9.6 event system so pipelines never run in silence.

**Architecture:** R15 is a single validation rule following the existing R01-R14 pattern. The event system introduces a new `events.py` module with 16 typed event dataclasses, an `EventEmitter` class supporting both callback and async-stream consumption, event emission points woven into the runner and handlers, and a `--verbose` CLI flag that wires a console printer to the emitter.

**Tech Stack:** Python 3.12, dataclasses, asyncio (asyncio.Queue for async stream), pytest + pytest-asyncio

**Spec references:**
- Section 7.2 (validation rules) -- R15 pattern
- Section 9.6 lines 1610-1649 (event types, observer + stream patterns)
- Section 3.1 line 329 (emit completion events in Finalize phase)

---

## Task 1: Validation Rule R15 -- manager_has_child_graph

**Files:**
- Modify: `src/attractor_pipeline/validation.py` (add rule + register it)
- Test: `tests/test_issue36_hexagon_hang.py` (add validation tests to existing issue-36 test file)

### Step 1: Write the failing tests

Add a new test class at the end of `tests/test_issue36_hexagon_hang.py`:

```python
from attractor_pipeline.validation import Severity, validate
from attractor_pipeline.graph import Edge, Graph, Node


class TestR15ManagerHasChildGraph:
    """R15: Hexagon nodes (manager) must have a child_graph attribute."""

    def test_hexagon_without_child_graph_produces_error(self):
        """A hexagon node with no child_graph attribute triggers R15 ERROR."""
        graph = Graph(name="test")
        graph.nodes["start"] = Node(id="start", shape="Mdiamond")
        graph.nodes["mgr"] = Node(id="mgr", shape="hexagon", label="Manager")
        graph.nodes["exit"] = Node(id="exit", shape="Msquare")
        graph.edges = [
            Edge(source="start", target="mgr"),
            Edge(source="mgr", target="exit"),
        ]

        diags = validate(graph)
        r15 = [d for d in diags if d.rule == "R15"]
        assert len(r15) == 1
        assert r15[0].severity == Severity.ERROR
        assert r15[0].node_id == "mgr"
        assert "child_graph" in r15[0].message

    def test_hexagon_with_child_graph_passes(self):
        """A hexagon node WITH child_graph produces no R15 diagnostic."""
        graph = Graph(name="test")
        graph.nodes["start"] = Node(id="start", shape="Mdiamond")
        graph.nodes["mgr"] = Node(
            id="mgr",
            shape="hexagon",
            label="Manager",
            attrs={"child_graph": "child.dot"},
        )
        graph.nodes["exit"] = Node(id="exit", shape="Msquare")
        graph.edges = [
            Edge(source="start", target="mgr"),
            Edge(source="mgr", target="exit"),
        ]

        diags = validate(graph)
        r15 = [d for d in diags if d.rule == "R15"]
        assert len(r15) == 0

    def test_non_hexagon_nodes_ignored(self):
        """R15 only applies to hexagon nodes -- other shapes are unaffected."""
        graph = Graph(name="test")
        graph.nodes["start"] = Node(id="start", shape="Mdiamond")
        graph.nodes["task"] = Node(id="task", shape="box", prompt="Do something")
        graph.nodes["exit"] = Node(id="exit", shape="Msquare")
        graph.edges = [
            Edge(source="start", target="task"),
            Edge(source="task", target="exit"),
        ]

        diags = validate(graph)
        r15 = [d for d in diags if d.rule == "R15"]
        assert len(r15) == 0

    def test_r15_via_parse_dot_and_validate(self):
        """End-to-end: parse a DOT string with hexagon node, validate catches R15."""
        g = parse_dot("""
        digraph R15Test {
            graph [goal="Test R15"]
            start  [shape=Mdiamond]
            mgr    [shape=hexagon, label="Sub-pipeline"]
            done   [shape=Msquare]
            start -> mgr -> done
        }
        """)
        diags = validate(g)
        r15 = [d for d in diags if d.rule == "R15"]
        assert len(r15) == 1
        assert r15[0].severity == Severity.ERROR
        assert r15[0].node_id == "mgr"
```

Note: the file already imports `parse_dot` at line 29. The new imports (`Severity`, `validate`, `Graph`, `Node`, `Edge`) must be added at the top of the file alongside the existing imports.

### Step 2: Run tests to verify they fail

Run: `python -m pytest tests/test_issue36_hexagon_hang.py::TestR15ManagerHasChildGraph -v`
Expected: All 4 tests FAIL (no R15 rule exists yet).

### Step 3: Implement R15 in validation.py

Add the rule function after `_rule_condition_syntax` (after line 389) and before the `ALL_RULES` registry comment:

```python
def _rule_manager_has_child_graph(graph: Graph) -> list[Diagnostic]:
    """R15: Manager nodes (shape=hexagon) must have a child_graph attribute.

    Hexagon maps to ManagerHandler which requires child_graph to know
    which sub-pipeline to orchestrate.  Without it the handler fails
    silently on every retry attempt (GitHub issue #36).
    """
    results: list[Diagnostic] = []
    for node in graph.nodes.values():
        if node.shape == "hexagon" and not node.attrs.get("child_graph"):
            results.append(
                Diagnostic(
                    rule="R15",
                    severity=Severity.ERROR,
                    message=(
                        f"Manager node '{node.id}' (shape=hexagon) has no "
                        f"'child_graph' attribute. The manager handler requires "
                        f"child_graph to specify which sub-pipeline to run. "
                        f"Did you mean shape=house for a human review gate?"
                    ),
                    node_id=node.id,
                )
            )
    return results
```

Then add `_rule_manager_has_child_graph` to the `ALL_RULES` list at the end (after `_rule_condition_syntax`):

Change line 410 from:
```python
    _rule_condition_syntax,
]
```
to:
```python
    _rule_condition_syntax,
    _rule_manager_has_child_graph,
]
```

Also update the module docstring on line 4 from "14 built-in" to "15 built-in":
```python
# Line 4: change "14 built-in" to "15 built-in"
```

### Step 4: Run tests to verify they pass

Run: `python -m pytest tests/test_issue36_hexagon_hang.py::TestR15ManagerHasChildGraph -v`
Expected: All 4 tests PASS.

### Step 5: Run full validation test suite to check for regressions

Run: `python -m pytest tests/test_pipeline_engine.py::TestValidation tests/test_loop_detector_validation.py -v`
Expected: All existing validation tests still PASS.

### Step 6: Commit

Commit message: `feat: add R15 validation rule -- hexagon nodes require child_graph (issue #36)`

Files: `src/attractor_pipeline/validation.py`, `tests/test_issue36_hexagon_hang.py`

---

## Task 2: Define event type dataclasses

**Files:**
- Create: `src/attractor_pipeline/engine/events.py`
- Test: `tests/test_events.py`

### Step 1: Write the failing tests

Create `tests/test_events.py`:

```python
"""Tests for Section 9.6 event system: event types and EventEmitter."""

from __future__ import annotations

from attractor_pipeline.engine.events import (
    CheckpointSaved,
    InterviewCompleted,
    InterviewStarted,
    InterviewTimeout,
    ParallelBranchCompleted,
    ParallelBranchStarted,
    ParallelCompleted,
    ParallelStarted,
    PipelineCompleted,
    PipelineEvent,
    PipelineFailed,
    PipelineStarted,
    StageCompleted,
    StageFailed,
    StageRetrying,
    StageStarted,
)


class TestEventTypes:
    """All 16 event types from Spec Section 9.6 are importable dataclasses."""

    def test_pipeline_started(self):
        e = PipelineStarted(name="MyPipeline", id="run-123")
        assert e.name == "MyPipeline"
        assert e.id == "run-123"
        assert isinstance(e, PipelineEvent)

    def test_pipeline_completed(self):
        e = PipelineCompleted(duration=12.5, artifact_count=3)
        assert e.duration == 12.5
        assert e.artifact_count == 3

    def test_pipeline_failed(self):
        e = PipelineFailed(error="boom", duration=1.0)
        assert e.error == "boom"

    def test_stage_started(self):
        e = StageStarted(name="build", index=0)
        assert e.name == "build"
        assert e.index == 0

    def test_stage_completed(self):
        e = StageCompleted(name="build", index=0, duration=2.5)
        assert e.duration == 2.5

    def test_stage_failed(self):
        e = StageFailed(name="build", index=0, error="timeout", will_retry=True)
        assert e.will_retry is True

    def test_stage_retrying(self):
        e = StageRetrying(name="build", index=0, attempt=2, delay=1.5)
        assert e.attempt == 2
        assert e.delay == 1.5

    def test_parallel_started(self):
        e = ParallelStarted(branch_count=3)
        assert e.branch_count == 3

    def test_parallel_branch_started(self):
        e = ParallelBranchStarted(branch="branch_0", index=0)
        assert e.branch == "branch_0"

    def test_parallel_branch_completed(self):
        e = ParallelBranchCompleted(branch="b0", index=0, duration=1.0, success=True)
        assert e.success is True

    def test_parallel_completed(self):
        e = ParallelCompleted(duration=5.0, success_count=2, failure_count=1)
        assert e.success_count == 2

    def test_interview_started(self):
        e = InterviewStarted(question="Approve?", stage="review")
        assert e.question == "Approve?"
        assert e.stage == "review"

    def test_interview_completed(self):
        e = InterviewCompleted(question="Approve?", answer="yes", duration=3.0)
        assert e.answer == "yes"

    def test_interview_timeout(self):
        e = InterviewTimeout(question="Approve?", stage="review", duration=60.0)
        assert e.duration == 60.0

    def test_checkpoint_saved(self):
        e = CheckpointSaved(node_id="build_step")
        assert e.node_id == "build_step"

    def test_all_events_are_pipeline_event_subtype(self):
        """Every event type is a subclass of PipelineEvent."""
        all_types = [
            PipelineStarted, PipelineCompleted, PipelineFailed,
            StageStarted, StageCompleted, StageFailed, StageRetrying,
            ParallelStarted, ParallelBranchStarted, ParallelBranchCompleted,
            ParallelCompleted,
            InterviewStarted, InterviewCompleted, InterviewTimeout,
            CheckpointSaved,
        ]
        assert len(all_types) == 15  # 15 concrete types (PipelineEvent is base)
        for cls in all_types:
            assert issubclass(cls, PipelineEvent), f"{cls.__name__} is not a PipelineEvent"

    def test_event_description_property(self):
        """Every event has a human-readable description property."""
        e = PipelineStarted(name="Test", id="abc")
        assert isinstance(e.description, str)
        assert len(e.description) > 0
```

### Step 2: Run tests to verify they fail

Run: `python -m pytest tests/test_events.py::TestEventTypes -v`
Expected: FAIL -- `ModuleNotFoundError: No module named 'attractor_pipeline.engine.events'`

### Step 3: Create the events module

Create `src/attractor_pipeline/engine/events.py`:

```python
"""Typed events for pipeline observability. Spec Section 9.6.

The engine emits these events during execution for UI, logging, and
metrics integration.  Each event is an immutable dataclass with a
human-readable ``description`` property.

All concrete event types inherit from ``PipelineEvent`` so consumers
can type-hint against the base class.
"""

from __future__ import annotations

from dataclasses import dataclass


# ------------------------------------------------------------------ #
# Base event
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class PipelineEvent:
    """Base class for all pipeline events."""

    @property
    def description(self) -> str:
        """Human-readable one-line summary of this event."""
        return f"{type(self).__name__}"


# ------------------------------------------------------------------ #
# Pipeline lifecycle events (Spec 9.6)
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class PipelineStarted(PipelineEvent):
    """Pipeline begins."""

    name: str
    id: str

    @property
    def description(self) -> str:
        return f"Pipeline '{self.name}' started (id={self.id})"


@dataclass(frozen=True)
class PipelineCompleted(PipelineEvent):
    """Pipeline succeeded."""

    duration: float
    artifact_count: int

    @property
    def description(self) -> str:
        return f"Pipeline completed in {self.duration:.1f}s ({self.artifact_count} artifacts)"


@dataclass(frozen=True)
class PipelineFailed(PipelineEvent):
    """Pipeline failed."""

    error: str
    duration: float

    @property
    def description(self) -> str:
        return f"Pipeline failed after {self.duration:.1f}s: {self.error}"


# ------------------------------------------------------------------ #
# Stage lifecycle events (Spec 9.6)
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class StageStarted(PipelineEvent):
    """Stage begins."""

    name: str
    index: int

    @property
    def description(self) -> str:
        return f"Stage [{self.index}] '{self.name}' started"


@dataclass(frozen=True)
class StageCompleted(PipelineEvent):
    """Stage succeeded."""

    name: str
    index: int
    duration: float

    @property
    def description(self) -> str:
        return f"Stage [{self.index}] '{self.name}' completed in {self.duration:.1f}s"


@dataclass(frozen=True)
class StageFailed(PipelineEvent):
    """Stage failed."""

    name: str
    index: int
    error: str
    will_retry: bool

    @property
    def description(self) -> str:
        retry = " (will retry)" if self.will_retry else " (no retry)"
        return f"Stage [{self.index}] '{self.name}' failed: {self.error}{retry}"


@dataclass(frozen=True)
class StageRetrying(PipelineEvent):
    """Stage retrying."""

    name: str
    index: int
    attempt: int
    delay: float

    @property
    def description(self) -> str:
        return f"Stage [{self.index}] '{self.name}' retrying (attempt {self.attempt}, delay {self.delay:.1f}s)"


# ------------------------------------------------------------------ #
# Parallel execution events (Spec 9.6)
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class ParallelStarted(PipelineEvent):
    """Parallel block started."""

    branch_count: int

    @property
    def description(self) -> str:
        return f"Parallel execution started with {self.branch_count} branches"


@dataclass(frozen=True)
class ParallelBranchStarted(PipelineEvent):
    """Branch started."""

    branch: str
    index: int

    @property
    def description(self) -> str:
        return f"Branch [{self.index}] '{self.branch}' started"


@dataclass(frozen=True)
class ParallelBranchCompleted(PipelineEvent):
    """Branch done."""

    branch: str
    index: int
    duration: float
    success: bool

    @property
    def description(self) -> str:
        status = "succeeded" if self.success else "failed"
        return f"Branch [{self.index}] '{self.branch}' {status} in {self.duration:.1f}s"


@dataclass(frozen=True)
class ParallelCompleted(PipelineEvent):
    """All branches done."""

    duration: float
    success_count: int
    failure_count: int

    @property
    def description(self) -> str:
        return (
            f"Parallel execution completed in {self.duration:.1f}s "
            f"({self.success_count} succeeded, {self.failure_count} failed)"
        )


# ------------------------------------------------------------------ #
# Human interaction events (Spec 9.6)
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class InterviewStarted(PipelineEvent):
    """Question presented."""

    question: str
    stage: str

    @property
    def description(self) -> str:
        return f"Interview started at '{self.stage}': {self.question}"


@dataclass(frozen=True)
class InterviewCompleted(PipelineEvent):
    """Answer received."""

    question: str
    answer: str
    duration: float

    @property
    def description(self) -> str:
        return f"Interview completed in {self.duration:.1f}s: answered '{self.answer}'"


@dataclass(frozen=True)
class InterviewTimeout(PipelineEvent):
    """Timeout reached."""

    question: str
    stage: str
    duration: float

    @property
    def description(self) -> str:
        return f"Interview timeout at '{self.stage}' after {self.duration:.1f}s"


# ------------------------------------------------------------------ #
# Checkpoint events (Spec 9.6)
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class CheckpointSaved(PipelineEvent):
    """Checkpoint written."""

    node_id: str

    @property
    def description(self) -> str:
        return f"Checkpoint saved at node '{self.node_id}'"
```

### Step 4: Run tests to verify they pass

Run: `python -m pytest tests/test_events.py::TestEventTypes -v`
Expected: All 18 tests PASS.

### Step 5: Commit

Commit message: `feat: define 16 event types for Section 9.6 observability system`

Files: `src/attractor_pipeline/engine/events.py`, `tests/test_events.py`

---

## Task 3: EventEmitter with callback and async stream patterns

**Files:**
- Modify: `src/attractor_pipeline/engine/events.py` (add EventEmitter class)
- Test: `tests/test_events.py` (add emitter tests)

### Step 1: Write the failing tests

Add to `tests/test_events.py`:

```python
import asyncio

import pytest

from attractor_pipeline.engine.events import EventEmitter, PipelineStarted, StageStarted


class TestEventEmitter:
    """EventEmitter supports callback and async stream patterns (Spec 9.6 lines 1639-1649)."""

    def test_emit_without_callback_is_noop(self):
        """Emitting without a callback does not raise."""
        emitter = EventEmitter()
        emitter.emit(PipelineStarted(name="test", id="1"))  # no error

    def test_callback_receives_events(self):
        """Observer pattern: on_event callback receives emitted events."""
        received: list = []
        emitter = EventEmitter(on_event=received.append)
        event = PipelineStarted(name="test", id="1")
        emitter.emit(event)
        assert received == [event]

    def test_callback_receives_multiple_events(self):
        """Callback receives events in emission order."""
        received: list = []
        emitter = EventEmitter(on_event=received.append)
        e1 = PipelineStarted(name="test", id="1")
        e2 = StageStarted(name="build", index=0)
        emitter.emit(e1)
        emitter.emit(e2)
        assert received == [e1, e2]

    @pytest.mark.asyncio
    async def test_async_stream_receives_events(self):
        """Stream pattern: async for event in emitter.events() yields emitted events."""
        emitter = EventEmitter()
        e1 = PipelineStarted(name="test", id="1")
        e2 = StageStarted(name="build", index=0)

        # Emit events then close the stream
        emitter.emit(e1)
        emitter.emit(e2)
        emitter.close()

        collected: list = []
        async for event in emitter.events():
            collected.append(event)

        assert collected == [e1, e2]

    @pytest.mark.asyncio
    async def test_async_stream_terminates_on_close(self):
        """Stream terminates cleanly when emitter.close() is called."""
        emitter = EventEmitter()

        async def collect() -> list:
            result = []
            async for event in emitter.events():
                result.append(event)
            return result

        task = asyncio.create_task(collect())
        emitter.emit(PipelineStarted(name="test", id="1"))
        emitter.close()

        result = await asyncio.wait_for(task, timeout=2.0)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_both_patterns_simultaneously(self):
        """Callback and stream both receive the same events."""
        callback_events: list = []
        emitter = EventEmitter(on_event=callback_events.append)

        event = PipelineStarted(name="test", id="1")
        emitter.emit(event)
        emitter.close()

        stream_events: list = []
        async for e in emitter.events():
            stream_events.append(e)

        assert callback_events == [event]
        assert stream_events == [event]
```

### Step 2: Run tests to verify they fail

Run: `python -m pytest tests/test_events.py::TestEventEmitter -v`
Expected: FAIL -- `ImportError: cannot import name 'EventEmitter' from 'attractor_pipeline.engine.events'`

### Step 3: Add EventEmitter to events.py

Append to `src/attractor_pipeline/engine/events.py` (after `CheckpointSaved`):

```python
import asyncio
from collections.abc import AsyncIterator, Callable


# ------------------------------------------------------------------ #
# Event emitter (Spec 9.6 lines 1639-1649)
# ------------------------------------------------------------------ #


_SENTINEL = object()  # Signals end of stream


class EventEmitter:
    """Emits pipeline events via callback and/or async stream.

    Supports both consumption patterns from Spec Section 9.6:

    1. **Observer/callback:** Pass ``on_event`` callable to constructor.
       Called synchronously on each ``emit()``.

    2. **Async stream:** Iterate ``async for event in emitter.events()``.
       Uses an asyncio.Queue internally. Call ``close()`` to signal
       end-of-stream.

    Both patterns can be used simultaneously.
    """

    def __init__(
        self,
        on_event: Callable[[PipelineEvent], None] | None = None,
    ) -> None:
        self._on_event = on_event
        self._queue: asyncio.Queue[PipelineEvent | object] = asyncio.Queue()
        self._closed = False

    def emit(self, event: PipelineEvent) -> None:
        """Emit an event to callback and stream consumers."""
        if self._on_event is not None:
            self._on_event(event)
        if not self._closed:
            self._queue.put_nowait(event)

    def close(self) -> None:
        """Signal end of event stream."""
        self._closed = True
        self._queue.put_nowait(_SENTINEL)

    async def events(self) -> AsyncIterator[PipelineEvent]:
        """Async iterator over emitted events. Terminates on close()."""
        while True:
            item = await self._queue.get()
            if item is _SENTINEL:
                break
            yield item  # type: ignore[misc]
```

Note: The `import asyncio` and `from collections.abc import AsyncIterator, Callable` must be added at the top of the file alongside the existing imports.

### Step 4: Run tests to verify they pass

Run: `python -m pytest tests/test_events.py -v`
Expected: All tests PASS (both TestEventTypes and TestEventEmitter).

### Step 5: Commit

Commit message: `feat: add EventEmitter with callback and async stream patterns (Spec 9.6)`

Files: `src/attractor_pipeline/engine/events.py`, `tests/test_events.py`

---

## Task 4: Runner integration -- pipeline lifecycle + stage lifecycle events

**Files:**
- Modify: `src/attractor_pipeline/engine/runner.py` (add `on_event` param, emit events)
- Modify: `src/attractor_pipeline/engine/events.py` (add import in `__init__` if needed)
- Test: `tests/test_events.py` (add runner integration tests)

### Step 1: Write the failing tests

Add to `tests/test_events.py`:

```python
from attractor_pipeline import (
    HandlerRegistry,
    PipelineStatus,
    parse_dot,
    register_default_handlers,
    run_pipeline,
)
from attractor_pipeline.engine.events import (
    CheckpointSaved,
    EventEmitter,
    PipelineCompleted,
    PipelineFailed,
    PipelineStarted,
    StageCompleted,
    StageFailed,
    StageRetrying,
    StageStarted,
)


class TestRunnerEventEmission:
    """run_pipeline emits lifecycle events via on_event callback."""

    @pytest.mark.asyncio
    async def test_pipeline_started_and_completed_events(self):
        """A successful pipeline emits PipelineStarted and PipelineCompleted."""
        g = parse_dot("""
        digraph E {
            graph [goal="Event test"]
            start [shape=Mdiamond]
            done  [shape=Msquare]
            start -> done
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        events: list = []
        result = await run_pipeline(g, registry, on_event=events.append)

        assert result.status == PipelineStatus.COMPLETED

        # First event is PipelineStarted
        assert isinstance(events[0], PipelineStarted)
        assert events[0].name == "E"

        # Last event is PipelineCompleted
        assert isinstance(events[-1], PipelineCompleted)
        assert events[-1].duration > 0

    @pytest.mark.asyncio
    async def test_stage_started_and_completed_events(self):
        """Each node emits StageStarted and StageCompleted."""
        g = parse_dot("""
        digraph S {
            graph [goal="Stage test"]
            start [shape=Mdiamond]
            task  [shape=box, prompt="Do it"]
            done  [shape=Msquare]
            start -> task -> done
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        events: list = []
        await run_pipeline(g, registry, on_event=events.append)

        stage_started = [e for e in events if isinstance(e, StageStarted)]
        stage_completed = [e for e in events if isinstance(e, StageCompleted)]

        # 3 nodes: start, task, done
        assert len(stage_started) == 3
        assert len(stage_completed) == 3

        # Names match node IDs
        names = [e.name for e in stage_started]
        assert "start" in names
        assert "task" in names
        assert "done" in names

    @pytest.mark.asyncio
    async def test_pipeline_failed_event_on_no_start(self):
        """PipelineFailed emitted when pipeline fails."""
        from attractor_pipeline.graph import Graph

        g = Graph(name="empty")
        registry = HandlerRegistry()
        register_default_handlers(registry)

        events: list = []
        result = await run_pipeline(g, registry, on_event=events.append)

        assert result.status == PipelineStatus.FAILED
        assert isinstance(events[0], PipelineStarted)
        failed = [e for e in events if isinstance(e, PipelineFailed)]
        assert len(failed) == 1
        assert "start" in failed[0].error.lower() or "No start" in failed[0].error

    @pytest.mark.asyncio
    async def test_on_event_none_is_safe(self):
        """Passing on_event=None (default) does not error."""
        g = parse_dot("""
        digraph Safe {
            graph [goal="No events"]
            start [shape=Mdiamond]
            done  [shape=Msquare]
            start -> done
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        # Default: no on_event -- must not raise
        result = await run_pipeline(g, registry)
        assert result.status == PipelineStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_checkpoint_saved_events(self):
        """CheckpointSaved event emitted after each node when logs_root is set."""
        import tempfile
        from pathlib import Path

        g = parse_dot("""
        digraph C {
            graph [goal="Checkpoint test"]
            start [shape=Mdiamond]
            task  [shape=box, prompt="Do it"]
            done  [shape=Msquare]
            start -> task -> done
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        events: list = []
        with tempfile.TemporaryDirectory() as tmp:
            await run_pipeline(
                g, registry,
                on_event=events.append,
                logs_root=Path(tmp),
            )

        ckpt_events = [e for e in events if isinstance(e, CheckpointSaved)]
        assert len(ckpt_events) >= 1
        node_ids = [e.node_id for e in ckpt_events]
        assert "task" in node_ids
```

### Step 2: Run tests to verify they fail

Run: `python -m pytest tests/test_events.py::TestRunnerEventEmission -v`
Expected: FAIL -- `run_pipeline() got an unexpected keyword argument 'on_event'`

### Step 3: Modify run_pipeline to accept on_event and emit events

In `src/attractor_pipeline/engine/runner.py`:

**3a. Add import** (after existing imports near line 28):
```python
from attractor_pipeline.engine.events import (
    CheckpointSaved,
    EventEmitter,
    PipelineCompleted,
    PipelineFailed,
    PipelineEvent,
    PipelineStarted,
    StageCompleted,
    StageFailed,
    StageRetrying,
    StageStarted,
)
```

**3b. Add `on_event` parameter to `run_pipeline` signature** (line 543-552). Add after `transforms`:
```python
    on_event: Callable[[PipelineEvent], None] | None = None,
```
Also add to the imports at the top of the file: `from collections.abc import Callable`

**3c. Create emitter right after `start_time` assignment** (after line 589):
```python
    emitter = EventEmitter(on_event=on_event)
```

**3d. Add a `_pipeline_id` and node index counter** (after emitter creation):
```python
    import uuid
    pipeline_id = str(uuid.uuid4())[:8]
    node_index = 0
```

**3e. Emit PipelineStarted** (after line 607, before the while loop at line 642):
```python
    emitter.emit(PipelineStarted(name=graph.name, id=pipeline_id))
```

**3f. Emit events at every PipelineResult return point.** There are multiple return points in the function. Each one that returns `PipelineStatus.FAILED` should emit `PipelineFailed` before returning. Each one that returns `PipelineStatus.COMPLETED` should emit `PipelineCompleted`. Each `CANCELLED` return should emit `PipelineFailed`.

The approach: create a helper at the top of the function body:
```python
    def _emit_terminal(pr: PipelineResult) -> PipelineResult:
        """Emit the terminal pipeline event and close the emitter."""
        if pr.status == PipelineStatus.COMPLETED:
            emitter.emit(PipelineCompleted(
                duration=pr.duration_seconds,
                artifact_count=len([k for k in ctx if k.startswith("codergen.") and k.endswith(".output")]),
            ))
        else:
            emitter.emit(PipelineFailed(
                error=pr.error or str(pr.status),
                duration=pr.duration_seconds,
            ))
        emitter.close()
        return pr
```

Then wrap every `return PipelineResult(...)` with `return _emit_terminal(PipelineResult(...))`. There are ~10 return points in `run_pipeline`. Each one must be wrapped.

**IMPORTANT**: The `return PipelineResult(...)` on line 630 (no start node) happens BEFORE the emitter emits PipelineStarted, so move the `PipelineStarted` emit to just after the emitter is created, before any early returns. Adjust accordingly -- the `PipelineStarted` emit must come AFTER setting `ctx["goal"]` and before any returns.

**3g. Emit StageStarted before handler execution** (inside the while loop, before `handler.execute()` at line 698):
```python
        stage_start_time = time.monotonic()
        emitter.emit(StageStarted(name=current_node.id, index=node_index))
        node_index += 1
```

**3h. Emit StageCompleted after successful execution** (after line 720 `completed_nodes.append`):
```python
        stage_duration = time.monotonic() - stage_start_time
```
Then after `node_outcomes[current_node.id] = result.status` (line 733):
```python
        emitter.emit(StageCompleted(
            name=current_node.id,
            index=node_index - 1,
            duration=stage_duration,
        ))
```

**3i. Emit StageFailed and StageRetrying in the retry block** (lines 723-729). Replace the retry block:
```python
        if result.status in (Outcome.FAIL, Outcome.RETRY):
            if retry_count < max_retries:
                node_retry_counts[current_node.id] = retry_count + 1
                delay = _get_retry_policy(current_node).compute_delay(retry_count)
                emitter.emit(StageRetrying(
                    name=current_node.id,
                    index=node_index - 1,
                    attempt=retry_count + 1,
                    delay=delay,
                ))
                await anyio.sleep(delay)
                continue  # retry same node
            # Max retries exhausted
            emitter.emit(StageFailed(
                name=current_node.id,
                index=node_index - 1,
                error=result.failure_reason,
                will_retry=False,
            ))
```

**3j. Emit CheckpointSaved** after checkpoint write (after line 750):
```python
            emitter.emit(CheckpointSaved(node_id=current_node.id))
```

### Step 4: Run tests to verify they pass

Run: `python -m pytest tests/test_events.py -v`
Expected: All tests PASS.

### Step 5: Run full test suite to check regressions

Run: `python -m pytest tests/test_pipeline_engine.py -v`
Expected: All existing tests still PASS (the new `on_event` param defaults to `None`).

### Step 6: Commit

Commit message: `feat: emit pipeline and stage lifecycle events from runner (Spec 9.6)`

Files: `src/attractor_pipeline/engine/runner.py`, `tests/test_events.py`

---

## Task 5: HumanHandler integration -- interview events

**Files:**
- Modify: `src/attractor_pipeline/handlers/human.py` (accept emitter, emit interview events)
- Modify: `src/attractor_pipeline/engine/runner.py` (pass emitter to handler)
- Test: `tests/test_events.py` (add interview event tests)

### Step 1: Write the failing tests

Add to `tests/test_events.py`:

```python
from attractor_pipeline.engine.events import InterviewCompleted, InterviewStarted


class TestHumanHandlerEvents:
    """HumanHandler emits InterviewStarted and InterviewCompleted."""

    @pytest.mark.asyncio
    async def test_human_gate_emits_interview_events(self):
        """A pipeline with a house node emits interview events."""
        g = parse_dot("""
        digraph H {
            graph [goal="Human test"]
            start  [shape=Mdiamond]
            review [shape=house, label="Approve deployment?"]
            done   [shape=Msquare]
            start -> review -> done [label="Approve"]
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        events: list = []
        result = await run_pipeline(g, registry, on_event=events.append)

        assert result.status == PipelineStatus.COMPLETED

        interview_started = [e for e in events if isinstance(e, InterviewStarted)]
        interview_completed = [e for e in events if isinstance(e, InterviewCompleted)]

        assert len(interview_started) == 1
        assert interview_started[0].stage == "review"
        assert "Approve" in interview_started[0].question

        assert len(interview_completed) == 1
        assert interview_completed[0].duration >= 0
```

### Step 2: Run tests to verify they fail

Run: `python -m pytest tests/test_events.py::TestHumanHandlerEvents -v`
Expected: FAIL -- no InterviewStarted/InterviewCompleted events emitted yet.

### Step 3: Implement interview event emission

The approach is to thread the `EventEmitter` through to the `HumanHandler`. There are two clean options:

**Option chosen: Store emitter on handler context.** The runner already passes `context: dict[str, Any]` to every handler. We'll store the emitter as `_event_emitter` in the context dict before calling the handler. This avoids changing the `Handler` protocol signature.

**3a. In `runner.py`**, before the `handler.execute()` call (near line 698), store the emitter:
```python
        ctx["_event_emitter"] = emitter
```

**3b. In `handlers/human.py`**, add imports at the top:
```python
import time as _time

from attractor_pipeline.engine.events import (
    EventEmitter,
    InterviewCompleted,
    InterviewStarted,
)
```

**3c. In `HumanHandler.execute()`**, emit events around the interview call. Before `self._interviewer.ask()` (near line 320):
```python
        # Emit interview event (Spec 9.6)
        _emitter: EventEmitter | None = context.get("_event_emitter")
        if _emitter:
            _emitter.emit(InterviewStarted(question=question, stage=node.id))
        interview_start = _time.monotonic()
```

After the `response = await self._interviewer.ask(...)` call succeeds (after line 326, before the except):
```python
            if _emitter:
                _emitter.emit(InterviewCompleted(
                    question=question,
                    answer=response,
                    duration=_time.monotonic() - interview_start,
                ))
```

**3d. Clean up the context key** in runner.py after handler.execute() returns, to not leak internal state:
```python
        ctx.pop("_event_emitter", None)
```

### Step 4: Run tests to verify they pass

Run: `python -m pytest tests/test_events.py::TestHumanHandlerEvents -v`
Expected: PASS.

### Step 5: Run full test suite

Run: `python -m pytest tests/ -v --timeout=30`
Expected: All tests PASS.

### Step 6: Commit

Commit message: `feat: emit interview events from HumanHandler (Spec 9.6)`

Files: `src/attractor_pipeline/handlers/human.py`, `src/attractor_pipeline/engine/runner.py`, `tests/test_events.py`

---

## Task 6: ParallelHandler integration -- parallel execution events

**Files:**
- Modify: `src/attractor_pipeline/handlers/parallel.py` (emit parallel events)
- Test: `tests/test_events.py` (add parallel event tests)

### Step 1: Write the failing tests

Add to `tests/test_events.py`:

```python
from attractor_pipeline.engine.events import (
    ParallelBranchCompleted,
    ParallelBranchStarted,
    ParallelCompleted,
    ParallelStarted,
)


class TestParallelHandlerEvents:
    """ParallelHandler emits parallel execution events."""

    @pytest.mark.asyncio
    async def test_parallel_emits_events(self):
        """A pipeline with parallel branches emits all 4 parallel event types."""
        g = parse_dot("""
        digraph P {
            graph [goal="Parallel test"]
            start    [shape=Mdiamond]
            fork     [shape=component]
            branch_a [shape=box, prompt="Task A"]
            branch_b [shape=box, prompt="Task B"]
            join     [shape=tripleoctagon]
            done     [shape=Msquare]
            start -> fork
            fork -> branch_a
            fork -> branch_b
            branch_a -> join
            branch_b -> join
            join -> done
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        events: list = []
        result = await run_pipeline(g, registry, on_event=events.append)

        assert result.status == PipelineStatus.COMPLETED

        par_started = [e for e in events if isinstance(e, ParallelStarted)]
        branch_started = [e for e in events if isinstance(e, ParallelBranchStarted)]
        branch_completed = [e for e in events if isinstance(e, ParallelBranchCompleted)]
        par_completed = [e for e in events if isinstance(e, ParallelCompleted)]

        assert len(par_started) == 1
        assert par_started[0].branch_count == 2

        assert len(branch_started) == 2
        assert len(branch_completed) == 2

        assert len(par_completed) == 1
        assert par_completed[0].success_count + par_completed[0].failure_count == 2
```

### Step 2: Run tests to verify they fail

Run: `python -m pytest tests/test_events.py::TestParallelHandlerEvents -v`
Expected: FAIL -- no parallel events emitted yet.

### Step 3: Implement parallel event emission

**3a. In `handlers/parallel.py`**, add imports:
```python
from attractor_pipeline.engine.events import (
    EventEmitter,
    ParallelBranchCompleted,
    ParallelBranchStarted,
    ParallelCompleted,
    ParallelStarted,
)
```

**3b. In `ParallelHandler.execute()`**, emit `ParallelStarted` after resolving outgoing edges (after line 101, before launching branches):
```python
        _emitter: EventEmitter | None = context.get("_event_emitter")
        par_start_time = time.monotonic()
        if _emitter:
            _emitter.emit(ParallelStarted(branch_count=len(outgoing)))
```

**3c. In `ParallelHandler._run_branch()`**, emit `ParallelBranchStarted` at the beginning (after line 268 `start_time = time.monotonic()`):
```python
        _emitter: EventEmitter | None = parent_context.get("_event_emitter")
        if _emitter:
            _emitter.emit(ParallelBranchStarted(
                branch=branch_id,
                index=int(branch_id.split("_")[-1]) if "_" in branch_id else 0,
            ))
```

**3d. At the end of `_run_branch()`**, before returning the `BranchResult` (line 288), emit `ParallelBranchCompleted`:
```python
        branch_duration = time.monotonic() - start_time
        if _emitter:
            _emitter.emit(ParallelBranchCompleted(
                branch=branch_id,
                index=int(branch_id.split("_")[-1]) if "_" in branch_id else 0,
                duration=branch_duration,
                success=result.status == Outcome.SUCCESS,
            ))
```

**3e. Back in `execute()`**, emit `ParallelCompleted` after all results collected (after the `branch_results` list is built, around line 174):
```python
        if _emitter:
            _emitter.emit(ParallelCompleted(
                duration=time.monotonic() - par_start_time,
                success_count=len([br for br in branch_results if br.result.status == Outcome.SUCCESS]),
                failure_count=len([br for br in branch_results if br.result.status == Outcome.FAIL]),
            ))
```

### Step 4: Run tests to verify they pass

Run: `python -m pytest tests/test_events.py::TestParallelHandlerEvents -v`
Expected: PASS.

### Step 5: Run full test suite

Run: `python -m pytest tests/ -v --timeout=30`
Expected: All tests PASS.

### Step 6: Commit

Commit message: `feat: emit parallel execution events from ParallelHandler (Spec 9.6)`

Files: `src/attractor_pipeline/handlers/parallel.py`, `tests/test_events.py`

---

## Task 7: CLI --verbose flag

**Files:**
- Modify: `src/attractor_pipeline/cli.py` (add --verbose flag, wire event printer)
- Test: `tests/test_events.py` (add CLI-level test)

### Step 1: Write the failing tests

Add to `tests/test_events.py`:

```python
from attractor_pipeline.engine.events import EventEmitter, PipelineEvent, PipelineStarted


class TestVerboseEventPrinter:
    """The verbose console printer formats events for human consumption."""

    def test_console_printer_formats_event(self, capsys):
        """_console_event_printer writes event description to stdout."""
        from attractor_pipeline.cli import _console_event_printer

        event = PipelineStarted(name="TestPipeline", id="abc123")
        _console_event_printer(event)

        captured = capsys.readouterr()
        assert "TestPipeline" in captured.out
        assert "abc123" in captured.out
```

### Step 2: Run tests to verify they fail

Run: `python -m pytest tests/test_events.py::TestVerboseEventPrinter -v`
Expected: FAIL -- `ImportError: cannot import name '_console_event_printer'`

### Step 3: Add --verbose flag and console printer to CLI

**3a. In `cli.py`**, add the `--verbose` flag to the run_parser (after line 73, before `# --- validate command ---`):
```python
    run_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show real-time pipeline events during execution",
    )
```

**3b. Add the console printer function** (after the imports, before `def main()`):
```python
def _console_event_printer(event: Any) -> None:
    """Print pipeline events to stdout for --verbose mode."""
    description = getattr(event, "description", str(event))
    print(f"  [event] {description}")
```

**3c. In `_cmd_run()`**, wire the callback when `--verbose` is set. Before the `run_pipeline` call (line 275), build the on_event callback:
```python
        on_event = _console_event_printer if getattr(args, "verbose", False) else None
```

Then pass it to `run_pipeline`:
```python
        result = await run_pipeline(
            graph,
            registry,
            logs_root=logs_root,
            on_event=on_event,
        )
```

### Step 4: Run tests to verify they pass

Run: `python -m pytest tests/test_events.py::TestVerboseEventPrinter -v`
Expected: PASS.

### Step 5: Commit

Commit message: `feat: add --verbose CLI flag for real-time pipeline event output`

Files: `src/attractor_pipeline/cli.py`, `tests/test_events.py`

---

## Task 8: Export event types from package __init__.py

**Files:**
- Modify: `src/attractor_pipeline/engine/__init__.py`
- Modify: `src/attractor_pipeline/__init__.py`

### Step 1: Write the failing test

Add to `tests/test_events.py`:

```python
class TestPublicExports:
    """Event types are importable from the top-level package."""

    def test_event_types_importable_from_package(self):
        """All event types can be imported from attractor_pipeline."""
        from attractor_pipeline import (
            CheckpointSaved,
            EventEmitter,
            InterviewCompleted,
            InterviewStarted,
            InterviewTimeout,
            ParallelBranchCompleted,
            ParallelBranchStarted,
            ParallelCompleted,
            ParallelStarted,
            PipelineCompleted,
            PipelineEvent,
            PipelineFailed,
            PipelineStarted,
            StageCompleted,
            StageFailed,
            StageRetrying,
            StageStarted,
        )

        # Verify they're the actual classes, not None
        assert PipelineEvent is not None
        assert EventEmitter is not None
```

### Step 2: Run test to verify it fails

Run: `python -m pytest tests/test_events.py::TestPublicExports -v`
Expected: FAIL -- `ImportError: cannot import name 'PipelineEvent' from 'attractor_pipeline'`

### Step 3: Add exports

**3a. In `src/attractor_pipeline/engine/__init__.py`**, add event exports:
```python
"""Pipeline execution engine."""

from attractor_pipeline.engine.events import (
    CheckpointSaved,
    EventEmitter,
    InterviewCompleted,
    InterviewStarted,
    InterviewTimeout,
    ParallelBranchCompleted,
    ParallelBranchStarted,
    ParallelCompleted,
    ParallelStarted,
    PipelineCompleted,
    PipelineEvent,
    PipelineFailed,
    PipelineStarted,
    StageCompleted,
    StageFailed,
    StageRetrying,
    StageStarted,
)
from attractor_pipeline.engine.runner import PipelineResult, run_pipeline

__all__ = [
    "run_pipeline",
    "PipelineResult",
    # Events (Spec 9.6)
    "PipelineEvent",
    "EventEmitter",
    "PipelineStarted",
    "PipelineCompleted",
    "PipelineFailed",
    "StageStarted",
    "StageCompleted",
    "StageFailed",
    "StageRetrying",
    "ParallelStarted",
    "ParallelBranchStarted",
    "ParallelBranchCompleted",
    "ParallelCompleted",
    "InterviewStarted",
    "InterviewCompleted",
    "InterviewTimeout",
    "CheckpointSaved",
]
```

**3b. In `src/attractor_pipeline/__init__.py`**, add event imports and exports. Add to the import block:
```python
from attractor_pipeline.engine.events import (
    CheckpointSaved,
    EventEmitter,
    InterviewCompleted,
    InterviewStarted,
    InterviewTimeout,
    ParallelBranchCompleted,
    ParallelBranchStarted,
    ParallelCompleted,
    ParallelStarted,
    PipelineCompleted,
    PipelineEvent,
    PipelineFailed,
    PipelineStarted,
    StageCompleted,
    StageFailed,
    StageRetrying,
    StageStarted,
)
```

And add to the `__all__` list, after the "Engine" section:
```python
    # Events (Spec 9.6)
    "PipelineEvent",
    "EventEmitter",
    "PipelineStarted",
    "PipelineCompleted",
    "PipelineFailed",
    "StageStarted",
    "StageCompleted",
    "StageFailed",
    "StageRetrying",
    "ParallelStarted",
    "ParallelBranchStarted",
    "ParallelBranchCompleted",
    "ParallelCompleted",
    "InterviewStarted",
    "InterviewCompleted",
    "InterviewTimeout",
    "CheckpointSaved",
```

### Step 4: Run test to verify it passes

Run: `python -m pytest tests/test_events.py::TestPublicExports -v`
Expected: PASS.

### Step 5: Commit

Commit message: `feat: export event types from attractor_pipeline public API`

Files: `src/attractor_pipeline/__init__.py`, `src/attractor_pipeline/engine/__init__.py`

---

## Task 9: Full integration test -- all event types in one pipeline

**Files:**
- Test: `tests/test_events.py` (add integration test)

### Step 1: Write the integration test

Add to `tests/test_events.py`:

```python
class TestFullEventIntegration:
    """End-to-end test: a complex pipeline emits all major event categories."""

    @pytest.mark.asyncio
    async def test_complex_pipeline_emits_all_event_categories(self):
        """Pipeline with human gate + parallel branches emits events from all categories."""
        g = parse_dot("""
        digraph Full {
            graph [goal="Full event test"]
            start    [shape=Mdiamond]
            review   [shape=house, label="Approve?"]
            fork     [shape=component]
            task_a   [shape=box, prompt="Task A"]
            task_b   [shape=box, prompt="Task B"]
            join     [shape=tripleoctagon]
            done     [shape=Msquare]
            start -> review
            review -> fork [label="Approve"]
            fork -> task_a
            fork -> task_b
            task_a -> join
            task_b -> join
            join -> done
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        events: list = []
        result = await run_pipeline(g, registry, on_event=events.append)

        assert result.status == PipelineStatus.COMPLETED

        # Check all event categories present
        type_names = {type(e).__name__ for e in events}

        # Pipeline lifecycle
        assert "PipelineStarted" in type_names
        assert "PipelineCompleted" in type_names

        # Stage lifecycle
        assert "StageStarted" in type_names
        assert "StageCompleted" in type_names

        # Human interaction
        assert "InterviewStarted" in type_names
        assert "InterviewCompleted" in type_names

        # Parallel execution
        assert "ParallelStarted" in type_names
        assert "ParallelBranchStarted" in type_names
        assert "ParallelBranchCompleted" in type_names
        assert "ParallelCompleted" in type_names

    @pytest.mark.asyncio
    async def test_event_order_is_logical(self):
        """PipelineStarted is first, PipelineCompleted is last."""
        g = parse_dot("""
        digraph Order {
            graph [goal="Order test"]
            start [shape=Mdiamond]
            task  [shape=box, prompt="Work"]
            done  [shape=Msquare]
            start -> task -> done
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        events: list = []
        await run_pipeline(g, registry, on_event=events.append)

        assert isinstance(events[0], PipelineStarted)
        assert isinstance(events[-1], PipelineCompleted)

        # StageStarted always comes before its StageCompleted
        started_indices = {
            e.name: i for i, e in enumerate(events) if isinstance(e, StageStarted)
        }
        completed_indices = {
            e.name: i for i, e in enumerate(events) if isinstance(e, StageCompleted)
        }
        for name in started_indices:
            if name in completed_indices:
                assert started_indices[name] < completed_indices[name], (
                    f"StageStarted('{name}') should come before StageCompleted('{name}')"
                )

    @pytest.mark.asyncio
    async def test_async_stream_consumption(self):
        """Events can be consumed via the async stream pattern."""
        g = parse_dot("""
        digraph Stream {
            graph [goal="Stream test"]
            start [shape=Mdiamond]
            done  [shape=Msquare]
            start -> done
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        emitter = EventEmitter()
        collected: list = []

        async def consume():
            async for event in emitter.events():
                collected.append(event)

        import asyncio
        consumer_task = asyncio.create_task(consume())

        await run_pipeline(g, registry, on_event=emitter.emit)
        emitter.close()

        await asyncio.wait_for(consumer_task, timeout=2.0)

        type_names = {type(e).__name__ for e in collected}
        assert "PipelineStarted" in type_names
        assert "PipelineCompleted" in type_names
```

### Step 2: Run the integration tests

Run: `python -m pytest tests/test_events.py::TestFullEventIntegration -v`
Expected: All PASS.

### Step 3: Run the full project test suite

Run: `python -m pytest tests/ -v --timeout=60`
Expected: All tests PASS across all test files.

### Step 4: Commit

Commit message: `test: add full integration tests for Section 9.6 event system`

Files: `tests/test_events.py`

---

## Task 10: Code quality check

**Files:** All modified files

### Step 1: Run ruff format check

Run: `python -m ruff format --check src/attractor_pipeline/engine/events.py src/attractor_pipeline/engine/runner.py src/attractor_pipeline/handlers/human.py src/attractor_pipeline/handlers/parallel.py src/attractor_pipeline/cli.py src/attractor_pipeline/validation.py src/attractor_pipeline/__init__.py src/attractor_pipeline/engine/__init__.py`

Fix any formatting issues: `python -m ruff format <files>`

### Step 2: Run ruff lint check

Run: `python -m ruff check src/attractor_pipeline/ tests/test_events.py tests/test_issue36_hexagon_hang.py`

Fix any lint issues.

### Step 3: Run pyright type check

Run: `python -m pyright src/attractor_pipeline/engine/events.py src/attractor_pipeline/engine/runner.py`

Fix any type errors.

### Step 4: Run full test suite one final time

Run: `python -m pytest tests/ -v --timeout=60`
Expected: All tests PASS.

### Step 5: Commit any fixes

Commit message: `chore: fix lint and type issues in event system implementation`

---

## Summary

| Task | What | Files | Est. Time |
|------|------|-------|-----------|
| 1 | R15 validation rule | validation.py, test_issue36 | 3 min |
| 2 | 16 event type dataclasses | events.py, test_events.py | 4 min |
| 3 | EventEmitter (callback + stream) | events.py, test_events.py | 4 min |
| 4 | Runner integration | runner.py, test_events.py | 5 min |
| 5 | HumanHandler events | human.py, runner.py, test_events.py | 4 min |
| 6 | ParallelHandler events | parallel.py, test_events.py | 4 min |
| 7 | CLI --verbose flag | cli.py, test_events.py | 3 min |
| 8 | Public API exports | __init__.py (x2) | 2 min |
| 9 | Full integration test | test_events.py | 3 min |
| 10 | Code quality check | all files | 2 min |

**Total estimated time: ~35 minutes**

**New files created:** 1 (`src/attractor_pipeline/engine/events.py`)
**Test file created:** 1 (`tests/test_events.py`)
**Files modified:** 7 (validation.py, runner.py, human.py, parallel.py, cli.py, `__init__.py` x2)