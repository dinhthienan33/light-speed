"""
Simplified HuBERT Vocoder models for SPIRIT-LM style synthesis.

Architecture:
┌─────────────────┐
│ HuBERT Embedding│  (Simple Embedding layer)
├─────────────────┤
│ DurationNet     │  (Predict duration per token) - reused from models.py
├─────────────────┤
│ Generator       │  (HiFi-GAN decoder) - reused from models.py
└─────────────────┘

This module inherits from models.py to avoid code duplication.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d
from torch.nn.utils import weight_norm

import commons

# Reuse existing components from models.py
from models import Generator, DurationNet


class HuBERTEncoder(nn.Module):
    """
    Simple encoder for HuBERT tokens.
    Maps discrete tokens to continuous representations.
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_channels: int,
        out_channels: int,
        n_layers: int = 4,
        kernel_size: int = 3,
        p_dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        
        # Token embedding
        self.embed = nn.Embedding(vocab_size, hidden_channels)
        nn.init.normal_(self.embed.weight, 0.0, hidden_channels ** -0.5)
        
        # Convolutional layers for local context
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(
                nn.Sequential(
                    weight_norm(Conv1d(
                        hidden_channels,
                        hidden_channels,
                        kernel_size,
                        padding=kernel_size // 2,
                    )),
                    nn.ReLU(),
                    nn.Dropout(p_dropout),
                )
            )
        
        # Output projection (mean and log variance)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
    
    def forward(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: [B, T] token indices
            lengths: [B] sequence lengths
            
        Returns:
            h: [B, C, T] hidden states
            m: [B, C, T] mean
            logs: [B, C, T] log variance
            mask: [B, 1, T] mask
        """
        # Embed tokens
        x = self.embed(tokens) * math.sqrt(self.hidden_channels)  # [B, T, H]
        x = x.transpose(1, 2)  # [B, H, T]
        
        # Create mask
        mask = commons.sequence_mask(lengths, tokens.size(1)).unsqueeze(1)  # [B, 1, T]
        x = x * mask
        
        # Apply convolutions with residual connections
        for conv in self.convs:
            x = conv(x) * mask + x
        
        # Project to mean and log variance
        stats = self.proj(x) * mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        
        return x, m, logs, mask


class HuBERTVocoder(nn.Module):
    """
    HuBERT Vocoder: Converts discrete HuBERT tokens to waveform.
    
    Simplified architecture compared to full VITS:
    1. HuBERT Embedding (no complex Transformer encoder)
    2. Duration Predictor (predict frame duration per token) - uses DurationNet from models.py
    3. HiFi-GAN Generator (synthesize waveform) - uses Generator from models.py
    """
    
    def __init__(
        self,
        vocab_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: list,
        resblock_dilation_sizes: list,
        upsample_rates: list,
        upsample_initial_channel: int,
        upsample_kernel_sizes: list,
        n_speakers: int = 0,
        gin_channels: int = 0,
        segment_size: int = 32,
        hop_length: int = 256,
        duration_dim: int = 256,
        duration_layers: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.segment_size = segment_size
        self.hop_length = hop_length
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        
        # HuBERT encoder
        self.encoder = HuBERTEncoder(
            vocab_size=vocab_size,
            hidden_channels=hidden_channels,
            out_channels=inter_channels,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
        )
        
        # Duration predictor - reuse DurationNet from models.py
        self.duration_predictor = DurationNet(
            vocab_size=vocab_size,
            dim=duration_dim,
            num_layers=duration_layers,
        )
        
        # HiFi-GAN generator - reuse Generator from models.py
        self.decoder = Generator(
            initial_channel=inter_channels,
            resblock=resblock,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
            upsample_rates=upsample_rates,
            upsample_initial_channel=upsample_initial_channel,
            upsample_kernel_sizes=upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        
        # Speaker embedding
        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)
    
    def forward(
        self,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
        gt_durations: Optional[torch.Tensor] = None,
        sid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training forward pass.
        
        Args:
            tokens: [B, T] HuBERT token indices
            token_lengths: [B] token sequence lengths
            gt_durations: [B, T] ground truth durations (in frames)
            sid: [B] speaker IDs (optional)
            
        Returns:
            wav_hat: [B, 1, T*hop] generated waveform
            pred_durations: [B, T] predicted durations
            h: [B, C, T_expanded] expanded hidden states
            mask: [B, 1, T_expanded] mask
        """
        # Get speaker embedding
        g = self.emb_g(sid).unsqueeze(-1) if self.n_speakers > 1 and sid is not None else None
        
        # Encode tokens
        h, m, logs, mask = self.encoder(tokens, token_lengths)
        
        # Predict durations
        pred_durations = self.duration_predictor(tokens, token_lengths).squeeze(-1)  # [B, T]
        
        # Use GT durations during training if available
        durations = gt_durations if gt_durations is not None else pred_durations
        
        # Expand hidden states by duration (length regulation)
        h_expanded, expanded_lengths = self._expand_by_duration(h, durations, token_lengths)
        
        # Create expanded mask
        max_expanded_len = h_expanded.size(2)
        expanded_mask = commons.sequence_mask(expanded_lengths, max_expanded_len).unsqueeze(1)
        
        # Generate waveform
        wav_hat = self.decoder(h_expanded * expanded_mask, g=g)
        
        return wav_hat, pred_durations, h_expanded, expanded_mask
    
    def _expand_by_duration(
        self,
        h: torch.Tensor,
        durations: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Expand hidden states by duration.
        
        Args:
            h: [B, C, T] hidden states
            durations: [B, T] durations (frames per token)
            lengths: [B] sequence lengths
            
        Returns:
            h_expanded: [B, C, T_expanded] expanded hidden states
            expanded_lengths: [B] expanded sequence lengths
        """
        B, C, T = h.shape
        device = h.device
        
        # Round durations to integers
        durations_int = torch.round(durations).long()
        durations_int = torch.clamp(durations_int, min=1)  # At least 1 frame per token
        
        # Mask out padding
        mask = commons.sequence_mask(lengths, T)  # [B, T]
        durations_int = durations_int * mask.long()
        
        # Calculate expanded lengths
        expanded_lengths = durations_int.sum(dim=1)  # [B]
        max_expanded_len = expanded_lengths.max().item()
        
        # Expand each sample
        h_expanded = torch.zeros(B, C, max_expanded_len, device=device, dtype=h.dtype)
        
        for b in range(B):
            idx = 0
            for t in range(lengths[b].item()):
                dur = durations_int[b, t].item()
                if dur > 0 and idx + dur <= max_expanded_len:
                    h_expanded[b, :, idx:idx + dur] = h[b, :, t:t+1].expand(-1, dur)
                    idx += dur
        
        return h_expanded, expanded_lengths
    
    @torch.no_grad()
    def infer(
        self,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
        sid: Optional[torch.Tensor] = None,
        duration_scale: float = 1.0,
        noise_scale: float = 0.667,
    ) -> torch.Tensor:
        """
        Inference: Generate waveform from tokens.
        
        Args:
            tokens: [B, T] HuBERT token indices
            token_lengths: [B] token sequence lengths
            sid: [B] speaker IDs (optional)
            duration_scale: Scale factor for durations
            noise_scale: Noise scale (not used in this simplified model)
            
        Returns:
            wav: [B, 1, T*hop] generated waveform
        """
        # Get speaker embedding
        g = self.emb_g(sid).unsqueeze(-1) if self.n_speakers > 1 and sid is not None else None
        
        # Encode tokens
        h, m, logs, mask = self.encoder(tokens, token_lengths)
        
        # Predict durations
        pred_durations = self.duration_predictor(tokens, token_lengths).squeeze(-1)
        pred_durations = pred_durations * duration_scale
        
        # Expand hidden states
        h_expanded, expanded_lengths = self._expand_by_duration(h, pred_durations, token_lengths)
        
        # Create expanded mask
        max_expanded_len = h_expanded.size(2)
        expanded_mask = commons.sequence_mask(expanded_lengths, max_expanded_len).unsqueeze(1)
        
        # Generate waveform
        wav = self.decoder(h_expanded * expanded_mask, g=g)
        
        return wav
    
    def remove_weight_norm(self):
        self.decoder.remove_weight_norm()


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = HuBERTVocoder(
        vocab_size=100,
        inter_channels=192,
        hidden_channels=192,
        filter_channels=768,
        n_layers=4,
        kernel_size=3,
        p_dropout=0.1,
        resblock="1",
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[8, 8, 2, 2],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[16, 16, 4, 4],
        n_speakers=1,
        gin_channels=256,
        segment_size=32,
        hop_length=256,
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward
    B, T = 2, 50
    tokens = torch.randint(0, 100, (B, T)).to(device)
    token_lengths = torch.tensor([50, 40]).to(device)
    gt_durations = torch.randint(1, 5, (B, T)).float().to(device)
    
    wav_hat, pred_dur, h, mask = model(tokens, token_lengths, gt_durations)
    print(f"Output wav shape: {wav_hat.shape}")
    print(f"Predicted durations shape: {pred_dur.shape}")
    
    # Test inference
    wav_infer = model.infer(tokens, token_lengths)
    print(f"Inference wav shape: {wav_infer.shape}")
