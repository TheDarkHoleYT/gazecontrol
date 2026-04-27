"""PipelineStage — Protocol defining the interface for all pipeline stages.

Every stage must implement:
- ``name``    : a human-readable string identifier (used by profiler).
- ``start()`` : called once when the pipeline starts.
- ``stop()``  : called once on shutdown (should be idempotent).
- ``process(ctx) -> FrameContext`` : process one frame tick.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from gazecontrol.pipeline.context import FrameContext


@runtime_checkable
class PipelineStage(Protocol):
    """Interface for a single pipeline processing stage."""

    @property
    def name(self) -> str:
        """Stage name used in profiler output and logs."""
        ...

    def start(self) -> bool:
        """Initialise the stage (open resources, threads, etc.).

        Returns:
            True on success, False on unrecoverable failure.
        """
        ...

    def stop(self) -> None:
        """Release resources.  Must be idempotent."""
        ...

    def process(self, ctx: FrameContext) -> FrameContext:
        """Process a single pipeline tick.

        Args:
            ctx: Mutable frame context populated by previous stages.

        Returns:
            The (potentially mutated) frame context.
        """
        ...
