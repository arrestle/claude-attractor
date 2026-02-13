"""SSE (Server-Sent Events) formatting and streaming.

Formats pipeline events into the SSE wire protocol and handles
connection lifecycle including late-connecting clients and
graceful termination.

Spec reference: attractor-spec §9.5-9.6.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from attractor_server.pipeline_manager import PipelineRun, SSEEvent


async def sse_stream(run: PipelineRun) -> AsyncIterator[str]:
    """Generate SSE-formatted text chunks from a pipeline run.

    Handles:
    - Normal streaming: yields events as they arrive
    - Late connection: if pipeline already finished, yields final status
    - Full queue: sentinel drain prevents stuck clients
    - Keepalive: sends comments every 30s to prevent timeout

    Usage with Starlette::

        return StreamingResponse(
            sse_stream(run),
            media_type="text/event-stream",
        )
    """
    # Late-connect: if pipeline already finished, replay full event history
    if run.is_terminal:
        for event in run._event_history:
            yield format_sse_event(event)
        return

    queue = run.subscribe()

    try:
        while True:
            # Use short timeout so we can check done_event frequently
            try:
                event = await asyncio.wait_for(queue.get(), timeout=1.0)
            except TimeoutError:
                # Check if pipeline finished (primary termination signal)
                if run._done_event.is_set():
                    # Drain any remaining events before exiting
                    while not queue.empty():
                        remaining = queue.get_nowait()
                        if remaining is not None:
                            yield format_sse_event(remaining)
                    break
                # Send keepalive every ~30s (count timeouts)
                continue

            if event is None:
                # Sentinel received (best-effort signal from close_subscribers)
                break

            yield format_sse_event(event)

    finally:
        run.unsubscribe(queue)


def format_sse_event(event: SSEEvent) -> str:
    """Format an SSEEvent into the SSE wire protocol.

    Returns a string like:
        event: pipeline.started
        data: {"name": "Pipeline", "id": "abc"}
        \\n
    """
    data_json = json.dumps(event.data, default=str)
    return f"event: {event.event_type}\ndata: {data_json}\n\n"
