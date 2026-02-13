"""Pydantic request/response models for the HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# ------------------------------------------------------------------ #
# Request models
# ------------------------------------------------------------------ #


class StartPipelineRequest(BaseModel):
    """POST /pipelines request body."""

    dot_source: str = Field(..., description="DOT graph source text")
    context: dict[str, Any] = Field(default_factory=dict, description="Initial context variables")
    provider: str | None = Field(None, description="LLM provider override")
    model: str | None = Field(None, description="LLM model override")


class AnswerRequest(BaseModel):
    """POST /pipelines/{id}/questions/{qid}/answer request body."""

    answer: str = Field(..., description="Answer to the question")


# ------------------------------------------------------------------ #
# Response models
# ------------------------------------------------------------------ #


class PipelineResponse(BaseModel):
    """Pipeline status response."""

    id: str
    status: str
    goal: str = ""
    current_node: str | None = None
    completed_nodes: list[str] = Field(default_factory=list)
    duration: float | None = None
    error: str | None = None


class PipelineCreatedResponse(BaseModel):
    """Response after starting a pipeline."""

    id: str
    status: str = "running"


class QuestionResponse(BaseModel):
    """A pending human gate question."""

    qid: str
    question: str
    stage: str
    timestamp: float


class AnswerResponse(BaseModel):
    """Response after answering a question."""

    qid: str
    accepted: bool = True


class GraphResponse(BaseModel):
    """Pipeline graph structure."""

    nodes: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(default_factory=list)


class ContextResponse(BaseModel):
    """Pipeline context key-value store."""

    values: dict[str, Any] = Field(default_factory=dict)


class CheckpointResponse(BaseModel):
    """Checkpoint state."""

    current_node: str | None = None
    completed_nodes: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    status: str = ""


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: str | None = None
