"""Face detection and embedding."""

from cheta.face.embedder import (
    FaceEmbedder,
    FaceObservation,
    SFaceEmbedder,
    cosine_similarity,
    head_yaw,
)

__all__ = ["FaceEmbedder", "FaceObservation", "SFaceEmbedder", "cosine_similarity", "head_yaw"]
