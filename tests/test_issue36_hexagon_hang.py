"""Regression tests for GitHub issue #36: Pipeline hangs during generation.

Root cause: shape=hexagon maps to ManagerHandler (expects child_graph),
not HumanHandler (shape=house). Without child_graph, the manager handler
returns Outcome.FAIL on every attempt. The runner then:

  1. Burns through retries (instant, zero delay)
  2. Falls through to edge selection (outcome-agnostic)
  3. Picks the unconditional forward edge and advances

The pipeline silently swallows the failure and COMPLETES -- but the
hexagon nodes never actually performed a human review. Combined with
zero progress output during execution, the user sees the pipeline
"hang" for minutes (real API calls on codergen nodes) with no feedback.

Fix: Use shape=house for human review gates.

https://github.com/samueljklee/attractor/issues/36
"""

from __future__ import annotations

import pytest

from attractor_pipeline import (
    HandlerRegistry,
    PipelineStatus,
    parse_dot,
    register_default_handlers,
    run_pipeline,
)
from attractor_pipeline.graph import Edge, Graph, Node, NodeShape
from attractor_pipeline.validation import Severity, validate


class TestIssue36ShapeRouting:
    """Verify shape-to-handler mapping for human review gates."""

    def test_hexagon_maps_to_manager(self):
        """shape=hexagon routes to 'manager', NOT 'wait.human'."""
        assert NodeShape.handler_for_shape("hexagon") == "manager"

    def test_house_maps_to_human(self):
        """shape=house routes to 'wait.human' (HumanHandler)."""
        assert NodeShape.handler_for_shape("house") == "wait.human"


class TestIssue36PipelineExecution:
    """End-to-end pipeline tests proving the fix."""

    @pytest.mark.asyncio
    async def test_hexagon_without_child_graph_silently_completes(self):
        """A hexagon node without child_graph completes but retries many times.

        This demonstrates issue #36: the ManagerHandler fails every
        attempt (no child_graph), the runner burns through all retries,
        then advances via the unconditional forward edge -- silently
        swallowing the failure.  The node appears many times in
        completed_nodes (once per attempt).
        """
        g = parse_dot("""
        digraph HexagonFail {
            graph [goal="Test hexagon failure"]
            start [shape=Mdiamond]
            task  [shape=box, prompt="Do something"]
            review [shape=hexagon, label="Human Check"]
            done  [shape=Msquare]
            start -> task -> review -> done
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        result = await run_pipeline(g, registry)

        # Pipeline completes -- the runner advances via unconditional edges
        # even when a node fails.  This is the silent-swallow behavior.
        assert result.status == PipelineStatus.COMPLETED

        # The hexagon node was retried many times (default max_retry=50),
        # each attempt appended to completed_nodes.
        review_count = result.completed_nodes.count("review")
        assert review_count > 1, (
            f"Expected hexagon node to be retried (appear >1 times), "
            f"but appeared {review_count} time(s)"
        )

    @pytest.mark.asyncio
    async def test_house_human_gate_completes_cleanly(self):
        """A house node (HumanHandler) auto-approves on first attempt.

        This is the correct fix for issue #36: use shape=house instead
        of shape=hexagon for human review gates.  The node succeeds on
        the first attempt with no retries.
        """
        g = parse_dot("""
        digraph HousePass {
            graph [goal="Test house success"]
            start  [shape=Mdiamond]
            task   [shape=box, prompt="Do something"]
            review [shape=house, label="Human Check"]
            done   [shape=Msquare]
            start -> task -> review -> done
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        result = await run_pipeline(g, registry)
        assert result.status == PipelineStatus.COMPLETED
        assert "review" in result.completed_nodes

        # House node succeeds on first attempt -- appears exactly once
        review_count = result.completed_nodes.count("review")
        assert review_count == 1, (
            f"Expected house node to succeed on first attempt (appear once), "
            f"but appeared {review_count} time(s)"
        )

    @pytest.mark.asyncio
    async def test_house_vs_hexagon_retry_contrast(self):
        """Direct comparison: house succeeds once, hexagon retries many times.

        Both pipelines COMPLETE (runner swallows failures via unconditional
        edges), but the retry counts reveal the problem.
        """
        hexagon_dot = """
        digraph Hex {
            graph [goal="Hexagon test"]
            start  [shape=Mdiamond]
            review [shape=hexagon, label="Human Check"]
            done   [shape=Msquare]
            start -> review -> done
        }
        """
        house_dot = """
        digraph House {
            graph [goal="House test"]
            start  [shape=Mdiamond]
            review [shape=house, label="Human Check"]
            done   [shape=Msquare]
            start -> review -> done
        }
        """
        registry = HandlerRegistry()
        register_default_handlers(registry)

        hex_result = await run_pipeline(parse_dot(hexagon_dot), registry)
        house_result = await run_pipeline(parse_dot(house_dot), registry)

        # Both pipelines complete
        assert hex_result.status == PipelineStatus.COMPLETED
        assert house_result.status == PipelineStatus.COMPLETED

        # The retry counts tell the real story
        hex_review_count = hex_result.completed_nodes.count("review")
        house_review_count = house_result.completed_nodes.count("review")

        assert house_review_count == 1, "House node should succeed on first attempt"
        assert hex_review_count > house_review_count, (
            f"Hexagon node should retry many times ({hex_review_count}) "
            f"vs house node once ({house_review_count})"
        )

    @pytest.mark.asyncio
    async def test_fixture_dot_file_validates_and_completes(self):
        """The fixture DOT file with shape=house parses, validates, and runs."""
        import pathlib

        dot_path = pathlib.Path(__file__).parent / "fixtures" / "validate_house_shape.dot"
        dot_content = dot_path.read_text()

        g = parse_dot(dot_content)

        # Verify structure
        assert len(g.nodes) == 4
        review_node = g.nodes["review1"]
        assert NodeShape.handler_for_shape(review_node.shape) == "wait.human"

        # Run pipeline
        registry = HandlerRegistry()
        register_default_handlers(registry)

        result = await run_pipeline(g, registry)
        assert result.status == PipelineStatus.COMPLETED
        assert "review1" in result.completed_nodes

        # No excessive retries
        assert result.completed_nodes.count("review1") == 1


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
