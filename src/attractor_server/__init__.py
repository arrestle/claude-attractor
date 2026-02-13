"""Attractor HTTP Server -- REST API + SSE events for pipeline execution.

Exposes the pipeline engine as an HTTP service with 9 endpoints:
- POST /pipelines -- start a pipeline
- GET /pipelines/{id} -- get status
- GET /pipelines/{id}/events -- SSE event stream
- POST /pipelines/{id}/cancel -- cancel a pipeline
- GET /pipelines/{id}/graph -- graph structure
- GET /pipelines/{id}/questions -- pending human questions
- POST /pipelines/{id}/questions/{qid}/answer -- answer a question
- GET /pipelines/{id}/checkpoint -- checkpoint state
- GET /pipelines/{id}/context -- context key-value store

Spec reference: attractor-spec §9.5-9.6.
"""
