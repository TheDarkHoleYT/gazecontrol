"""GazeControl pipeline package.

Stages: CaptureStage → GazeStage → GestureStage → IntentStage
Orchestrated by GazeControlPipeline (QThread).
"""
from gazecontrol.pipeline.context import FrameContext
from gazecontrol.pipeline.orchestrator import GazeControlPipeline

__all__ = ["FrameContext", "GazeControlPipeline"]
