"""
HuBERT module for speech unit extraction.

Classes:
- Hubert: Base HuBERT model
- HubertSoft: Soft speech units encoder  
- HubertDiscrete: Discrete speech units encoder (layer 7)
- CentroidKMeans: Lightweight KMeans predictor (no sklearn dependency)

Constants:
- FEATURE_LAYER: Default layer for feature extraction (7)
"""

from .model import (
    Hubert,
    HubertDiscrete,
    HubertSoft,
    CentroidKMeans,
    FEATURE_LAYER,
)
from .utils import (
    Metric,
    save_checkpoint,
    load_checkpoint,
)

__all__ = [
    "Hubert",
    "HubertDiscrete", 
    "HubertSoft",
    "CentroidKMeans",
    "FEATURE_LAYER",
    "Metric",
    "save_checkpoint",
    "load_checkpoint",
]
