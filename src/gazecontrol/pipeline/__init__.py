"""GazeControl pipeline package.

Public surface:

- :class:`PipelineEngine`  — Qt-free synchronous frame-loop runner.
- :class:`PipelineStage`    — Protocol every stage implements.
- :class:`FrameContext`    — typed accumulator passed between stages.
- :class:`QtPipelineThread` — QThread adapter (Qt-only consumers).
- ``GazeControlPipeline`` — deprecated alias for ``QtPipelineThread``;
  removed in 0.9.x.
"""

from gazecontrol.pipeline.context import FrameContext
from gazecontrol.pipeline.engine import PipelineEngine
from gazecontrol.pipeline.qt_adapter import QtPipelineThread
from gazecontrol.pipeline.stage import PipelineStage

#: Deprecated alias kept for the 0.8 → 0.9 transition.  Will be removed in 0.9.
GazeControlPipeline = QtPipelineThread

__all__ = [
    "FrameContext",
    "GazeControlPipeline",
    "PipelineEngine",
    "PipelineStage",
    "QtPipelineThread",
]
