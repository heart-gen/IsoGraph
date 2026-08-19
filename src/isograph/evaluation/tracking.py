"""Optional MLflow tracking."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def tracking_run(uri: str | None, run_name: str) -> Iterator[object | None]:
    if uri is None:
        yield None
        return
    try:
        import mlflow
    except ImportError:
        yield None
        return
    mlflow.set_tracking_uri(uri)
    with mlflow.start_run(run_name=run_name) as run:
        yield run
