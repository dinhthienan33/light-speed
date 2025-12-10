"""
HuBERT model implementation.

Classes:
- Hubert: Base HuBERT model
- HubertSoft: Soft speech units encoder
- HubertDiscrete: Discrete speech units encoder (supports both sklearn KMeans and CentroidKMeans)
"""

import copy
from typing import Optional, Tuple, Union
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# Configuration
# ================================================================
FEATURE_LAYER = 7  # Default layer for feature extraction


# ================================================================
# CentroidKMeans - Lightweight predictor without sklearn dependency
# ================================================================
class CentroidKMeans:
    """
    Lightweight predictor that assigns samples to the nearest centroid.
    Uses only stored cluster centers, avoiding scikit-learn internal attrs.
    Compatible across sklearn versions.
    """
    def __init__(self, cluster_centers: np.ndarray):
        centers = np.asarray(cluster_centers, dtype=np.float32)
        if centers.ndim != 2:
            raise ValueError("cluster_centers must be 2D array [n_clusters, n_features]")
        self.cluster_centers_ = centers
        self.n_clusters = centers.shape[0]
        self.n_features_in_ = centers.shape[1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for samples."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"X must be shape [n_samples, {self.n_features_in_}]")
        # Compute squared distances efficiently
        x2 = np.sum(X * X, axis=1, keepdims=True)
        c2 = np.sum(self.cluster_centers_ * self.cluster_centers_, axis=1)
        xc = X @ self.cluster_centers_.T
        d2 = x2 + c2[None, :] - 2.0 * xc
        return np.argmin(d2, axis=1)

    @classmethod
    def from_joblib(cls, path: str) -> "CentroidKMeans":
        """
        Load KMeans from joblib file, extracting only cluster centers.
        Compatible across sklearn versions.
        """
        import joblib
        
        # Handle numpy random state compatibility issues
        try:
            from numpy.random import MT19937, BitGenerator
            np.random.bit_generator = BitGenerator
            np.random._mt19937 = MT19937
        except (ImportError, AttributeError):
            pass
        
        obj = joblib.load(path)
        
        if isinstance(obj, dict) and 'cluster_centers_' in obj:
            centers = obj['cluster_centers_']
        elif hasattr(obj, 'cluster_centers_'):
            centers = obj.cluster_centers_
        else:
            raise ValueError("Loaded object lacks cluster_centers_")
        
        return cls(centers)


# Type alias for KMeans-like objects
KMeansLike = Union[CentroidKMeans, "sklearn.cluster.KMeans"]


# ================================================================
# Base Hubert Model
# ================================================================
class Hubert(nn.Module):
    def __init__(self, num_label_embeddings: int = 100, mask: bool = True):
        super().__init__()
        self._mask = mask
        self.feature_extractor = FeatureExtractor()
        self.feature_projection = FeatureProjection()
        self.positional_embedding = PositionalConvEmbedding()
        self.norm = nn.LayerNorm(768)
        self.dropout = nn.Dropout(0.1)
        self.encoder = TransformerEncoder(
            nn.TransformerEncoderLayer(
                768, 12, 3072, activation="gelu", batch_first=True
            ),
            12,
        )
        self.proj = nn.Linear(768, 256)

        self.masked_spec_embed = nn.Parameter(torch.FloatTensor(768).uniform_())
        self.label_embedding = nn.Embedding(num_label_embeddings, 256)

    def mask(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = None
        if self.training and self._mask:
            mask = _compute_mask((x.size(0), x.size(1)), 0.8, 10, x.device, 2)
            x[mask] = self.masked_spec_embed.to(x.dtype)
        return x, mask

    def encode(
        self, x: torch.Tensor, layer: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.feature_extractor(x)
        x = self.feature_projection(x.transpose(1, 2))
        x, mask = self.mask(x)
        x = x + self.positional_embedding(x)
        x = self.dropout(self.norm(x))
        x = self.encoder(x, output_layer=layer)
        return x, mask

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.cosine_similarity(
            x.unsqueeze(2),
            self.label_embedding.weight.unsqueeze(0).unsqueeze(0),
            dim=-1,
        )
        return logits / 0.1

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, mask = self.encode(x)
        x = self.proj(x)
        logits = self.logits(x)
        return logits, mask


# ================================================================
# HubertSoft - Soft speech units
# ================================================================
class HubertSoft(Hubert):
    """HuBERT-Soft content encoder."""

    def __init__(self):
        super().__init__()

    @torch.inference_mode()
    def units(self, wav: torch.Tensor) -> torch.Tensor:
        """Extract soft speech units.

        Args:
            wav: Audio waveform of shape (1, 1, T)

        Returns:
            Soft speech units of shape (1, N, D)
        """
        wav = F.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))
        x, _ = self.encode(wav)
        return self.proj(x)


# ================================================================
# HubertDiscrete - Discrete speech units (Layer 7)
# ================================================================
class HubertDiscrete(Hubert):
    """
    HuBERT-Discrete content encoder.
    
    Supports both sklearn KMeans and lightweight CentroidKMeans.
    Uses layer 7 for feature extraction by default.
    """

    def __init__(
        self,
        kmeans: KMeansLike,
        layer: int = FEATURE_LAYER,
        num_label_embeddings: Optional[int] = None,
    ):
        """
        Args:
            kmeans: KMeans model (sklearn or CentroidKMeans)
            layer: Transformer layer for feature extraction (default: 7)
            num_label_embeddings: Number of label embeddings (default: kmeans.n_clusters)
        """
        n_clusters = kmeans.n_clusters
        super().__init__(num_label_embeddings or n_clusters)
        self.kmeans = kmeans
        self.layer = layer

    @torch.inference_mode()
    def units(self, wav: torch.Tensor) -> torch.LongTensor:
        """Extract discrete speech units.

        Args:
            wav: Audio waveform of shape (1, 1, T)

        Returns:
            Discrete speech units of shape (N,)
        """
        wav = F.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))
        x, _ = self.encode(wav, layer=self.layer)
        x = self.kmeans.predict(x.squeeze().cpu().numpy())
        return torch.tensor(x, dtype=torch.long, device=wav.device)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        kmeans_path: str,
        layer: int = FEATURE_LAYER,
        device: str = "cpu",
    ) -> "HubertDiscrete":
        """
        Load HubertDiscrete from checkpoint and kmeans files.
        
        Args:
            model_path: Path to model checkpoint (.pt file)
            kmeans_path: Path to kmeans model (.joblib file)
            layer: Transformer layer for feature extraction
            device: Device to load model on
            
        Returns:
            HubertDiscrete model ready for inference
        """
        # Load kmeans
        kmeans = CentroidKMeans.from_joblib(kmeans_path)
        print(f"✅ Loaded KMeans with {kmeans.n_clusters} clusters")
        
        # Create model
        model = cls(kmeans=kmeans, layer=layer)
        
        # Load weights
        state_dict = torch.load(model_path, map_location='cpu')
        if isinstance(state_dict, dict):
            if 'hubert' in state_dict:
                state_dict = state_dict['hubert']
            elif 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        model = model.to(device).eval()
        print(f"✅ HubertDiscrete loaded (layer {layer})")
        
        return model


# ================================================================
# Feature Extractor Components
# ================================================================
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv1d(1, 512, 10, 5, bias=False)
        self.norm0 = nn.GroupNorm(512, 512)
        self.conv1 = nn.Conv1d(512, 512, 3, 2, bias=False)
        self.conv2 = nn.Conv1d(512, 512, 3, 2, bias=False)
        self.conv3 = nn.Conv1d(512, 512, 3, 2, bias=False)
        self.conv4 = nn.Conv1d(512, 512, 3, 2, bias=False)
        self.conv5 = nn.Conv1d(512, 512, 2, 2, bias=False)
        self.conv6 = nn.Conv1d(512, 512, 2, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.norm0(self.conv0(x)))
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        x = F.gelu(self.conv4(x))
        x = F.gelu(self.conv5(x))
        x = F.gelu(self.conv6(x))
        return x


class FeatureProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(512)
        self.projection = nn.Linear(512, 768)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.projection(x)
        x = self.dropout(x)
        return x


class PositionalConvEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(
            768,
            768,
            kernel_size=128,
            padding=128 // 2,
            groups=16,
        )
        self.conv = nn.utils.parametrizations.weight_norm(
            self.conv, name="weight", dim=2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x.transpose(1, 2))
        x = F.gelu(x[:, :, :-1])
        return x.transpose(1, 2)


class TransformerEncoder(nn.Module):
    def __init__(
        self, encoder_layer: nn.TransformerEncoderLayer, num_layers: int
    ) -> None:
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers

    def forward(
        self,
        src: torch.Tensor,
        mask: torch.Tensor = None,
        src_key_padding_mask: torch.Tensor = None,
        output_layer: Optional[int] = None,
    ) -> torch.Tensor:
        output = src
        for layer in self.layers[:output_layer]:
            output = layer(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask
            )
        return output


# ================================================================
# Utility Functions
# ================================================================
def _compute_mask(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    device: torch.device,
    min_masks: int = 0,
) -> torch.Tensor:
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length} and `sequence_length`: {sequence_length}`"
        )

    # compute number of masked spans in batch
    num_masked_spans = int(mask_prob * sequence_length / mask_length + random.random())
    num_masked_spans = max(num_masked_spans, min_masks)

    # make sure num masked indices <= sequence_length
    if num_masked_spans * mask_length > sequence_length:
        num_masked_spans = sequence_length // mask_length

    # SpecAugment mask to fill
    mask = torch.zeros((batch_size, sequence_length), device=device, dtype=torch.bool)

    # uniform distribution to sample from, make sure that offset samples are < sequence_length
    uniform_dist = torch.ones(
        (batch_size, sequence_length - (mask_length - 1)), device=device
    )

    # get random indices to mask
    mask_indices = torch.multinomial(uniform_dist, num_masked_spans)

    # expand masked indices to masked spans
    mask_indices = (
        mask_indices.unsqueeze(dim=-1)
        .expand((batch_size, num_masked_spans, mask_length))
        .reshape(batch_size, num_masked_spans * mask_length)
    )
    offsets = (
        torch.arange(mask_length, device=device)[None, None, :]
        .expand((batch_size, num_masked_spans, mask_length))
        .reshape(batch_size, num_masked_spans * mask_length)
    )
    mask_idxs = mask_indices + offsets

    # scatter indices to mask
    mask = mask.scatter(1, mask_idxs, True)

    return mask
