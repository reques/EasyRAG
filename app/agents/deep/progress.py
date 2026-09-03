"""兼容层：统一进度投影器已迁移到 app.agents.progress。"""
from app.agents.progress import ProgressProjector as DeepResearchProgressProjector

__all__ = ["DeepResearchProgressProjector"]
