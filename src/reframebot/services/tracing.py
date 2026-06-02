"""Optional LangSmith tracing helpers.

This module keeps instrumentation details out of the API and service flow.
When LangSmith is unavailable or disabled, every helper degrades to a no-op.
"""
from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any, Callable, ContextManager, Mapping, Sequence

from reframebot.config import Settings

try:
    from langsmith import traceable
    from langsmith.run_helpers import get_current_run_tree, tracing_context
except Exception:  # pragma: no cover - tracing is optional
    traceable = None
    get_current_run_tree = None
    tracing_context = None


def enabled(settings: Settings) -> bool:
    return bool(
        settings.langsmith_tracing
        and os.environ.get("LANGSMITH_API_KEY")
        and tracing_context is not None
        and traceable is not None
    )


def context(
    settings: Settings,
    *,
    tags: Sequence[str],
    metadata: Mapping[str, Any],
) -> ContextManager[Any]:
    if not enabled(settings):
        return nullcontext()
    return tracing_context(
        project_name=settings.langsmith_project,
        tags=list(tags),
        metadata=dict(metadata),
        enabled=True,
    )


def decorate(
    fn: Callable[..., Any],
    *,
    name: str,
    run_type: str,
) -> Callable[..., Any]:
    if traceable is None:
        return fn
    return traceable(name=name, run_type=run_type)(fn)


def add_metadata(**metadata: object) -> None:
    if get_current_run_tree is None:
        return
    run = get_current_run_tree()
    if run is not None:
        run.add_metadata(metadata)


def add_outputs(outputs: dict[str, object]) -> None:
    if get_current_run_tree is None:
        return
    run = get_current_run_tree()
    if run is not None:
        run.add_outputs(outputs)


def set_usage(
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    input_cost: float = 0.0,
    output_cost: float = 0.0,
) -> None:
    if get_current_run_tree is None:
        return
    run = get_current_run_tree()
    if run is None:
        return
    run.set(
        usage_metadata={
            "input_tokens": int(input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
            "total_tokens": int((input_tokens or 0) + (output_tokens or 0)),
            "input_cost": float(input_cost),
            "output_cost": float(output_cost),
            "total_cost": float(input_cost + output_cost),
        }
    )
