"""
Dataset loader for HuBERT vocoder training.

Loads:
- Audio files from InfoRE dataset
- Uses HubertDiscrete from hubert module to extract discrete tokens
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pooch
from pooch import Unzip
from tqdm import tqdm

# Import from local hubert module
from hubert import HubertDiscrete, FEATURE_LAYER
from config_helper import get_wandb_artifact_path

try:
    import torchaudio
except ImportError:
    torchaudio = None

try:
    import librosa
except ImportError:
    librosa = None


# ================================================================
# InfoRE Dataset Download
# ================================================================
def download_infore_data(data_root: Path = Path("./train_data")) -> Path:
    """
    Download InfoRE dataset and textgrid files.
    
    Args:
        data_root: Directory to store downloaded data
        
    Returns:
        Path to data directory containing wav and TextGrid files
    """
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    
    # Check if data already exists
    wav_files = list(data_root.glob("*.wav"))
    if len(wav_files) > 0:
        print(f"InfoRE data already exists at {data_root} ({len(wav_files)} wav files)")
        return data_root
    
    print("Downloading InfoRE wav files...")
    files = pooch.retrieve(
        url="https://huggingface.co/datasets/ntt123/infore/resolve/main/infore_16k_denoised.zip",
        known_hash="2445527b345fb0b1816ce3c8f09bae419d6bbe251f16d6c74d8dd95ef9fb0737",
        processor=Unzip(),
        progressbar=True,
    )
    wav_dir = Path(sorted(files)[0]).parent
    
    print("Downloading TextGrid files...")
    files = pooch.retrieve(
        url="https://huggingface.co/datasets/ntt123/infore/resolve/main/infore_tg.zip",
        known_hash="26e4f53025220097ea95dc266657de8d65104b0a17a6ffba778fc016c8dd36d7",
        processor=Unzip(),
        progressbar=True,
    )
    tg_dir = Path(sorted(files)[0]).parent
    
    # Copy files to data_root
    print(f"Copying files to {data_root}...")
    for path in tqdm(list(tg_dir.glob("*.TextGrid"))):
        wav_name = path.with_suffix(".wav").name
        wav_src = wav_dir / wav_name
        if wav_src.exists():
            shutil.copy(path, data_root)
            shutil.copy(wav_src, data_root)
    
    wav_files = list(data_root.glob("*.wav"))
    print(f"✅ Downloaded {len(wav_files)} audio files to {data_root}")
    return data_root


# ================================================================
# Audio Loading
# ================================================================
def load_audio(path: str, sample_rate: int = 16000) -> Tuple[np.ndarray, int]:
    """Load audio file and resample to target sample rate."""
    if torchaudio is not None:
        wav_tensor, sr = torchaudio.load(path)
        if wav_tensor.ndim > 1:
            wav_tensor = wav_tensor.mean(dim=0, keepdim=True)
        wav = wav_tensor.squeeze(0).numpy()
        if sr != sample_rate:
            wav = torchaudio.functional.resample(
                torch.from_numpy(wav), orig_freq=sr, new_freq=sample_rate
            ).numpy()
    elif librosa is not None:
        wav, _ = librosa.load(path, sr=sample_rate)
    else:
        raise ImportError("Either torchaudio or librosa is required for audio loading")
    
    return wav, sample_rate


# ================================================================
# Token Processing
# ================================================================
def deduplicate_with_duration(tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deduplicate consecutive tokens and compute durations."""
    if len(tokens) == 0:
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.float)
    
    tokens_list = tokens.tolist()
    deduped = [tokens_list[0]]
    durations = [1]
    
    for token in tokens_list[1:]:
        if token == deduped[-1]:
            durations[-1] += 1
        else:
            deduped.append(token)
            durations.append(1)
    
    return (
        torch.tensor(deduped, dtype=torch.long),
        torch.tensor(durations, dtype=torch.float),
    )


# ================================================================
# Dataset
# ================================================================
class InfoREHuBERTDataset(Dataset):
    """
    Dataset for HuBERT vocoder training using InfoRE audio data.
    Extracts HuBERT tokens on-the-fly using HubertDiscrete model.
    """
    
    def __init__(
        self,
        data_dir: Path,
        hubert_discrete: HubertDiscrete,
        sample_rate: int = 16000,
        max_wav_length: int = 16000 * 10,
        deduplicate: bool = True,
        precomputed_tokens: Optional[Dict[str, np.ndarray]] = None,
        device: str = "cuda",
    ):
        self.data_dir = Path(data_dir)
        self.hubert_discrete = hubert_discrete
        self.sample_rate = sample_rate
        self.max_wav_length = max_wav_length
        self.deduplicate = deduplicate
        self.precomputed_tokens = precomputed_tokens or {}
        self.device = device
        
        self.wav_files = sorted(list(self.data_dir.glob("*.wav")))
        print(f"Found {len(self.wav_files)} wav files in {data_dir}")
    
    def __len__(self):
        return len(self.wav_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        wav_path = self.wav_files[idx]
        filename = wav_path.stem
        
        wav, _ = load_audio(str(wav_path), self.sample_rate)
        
        if len(wav) > self.max_wav_length:
            wav = wav[:self.max_wav_length]
        
        wav_tensor = torch.from_numpy(wav).float()
        
        # Get tokens
        if filename in self.precomputed_tokens:
            tokens = torch.tensor(self.precomputed_tokens[filename], dtype=torch.long)
        else:
            wav_input = wav_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
            tokens = self.hubert_discrete.units(wav_input).cpu()
        
        if self.deduplicate:
            tokens, durations = deduplicate_with_duration(tokens)
        else:
            durations = torch.ones(len(tokens), dtype=torch.float)
        
        return {
            "filename": filename,
            "wav": wav_tensor,
            "wav_length": len(wav_tensor),
            "tokens": tokens,
            "token_length": len(tokens),
            "durations": durations,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader."""
    max_wav_len = max(item["wav_length"] for item in batch)
    max_token_len = max(item["token_length"] for item in batch)
    
    wavs = torch.stack([
        F.pad(item["wav"], (0, max_wav_len - item["wav_length"]))
        for item in batch
    ])
    tokens = torch.stack([
        F.pad(item["tokens"], (0, max_token_len - item["token_length"]), value=0)
        for item in batch
    ])
    durations = torch.stack([
        F.pad(item["durations"], (0, max_token_len - item["token_length"]), value=0)
        for item in batch
    ])
    
    return {
        "wav": wavs,
        "wav_lengths": torch.tensor([item["wav_length"] for item in batch], dtype=torch.long),
        "tokens": tokens,
        "token_lengths": torch.tensor([item["token_length"] for item in batch], dtype=torch.long),
        "durations": durations,
    }


def create_dataloaders(
    data_dir: Path,
    hubert_discrete: HubertDiscrete,
    batch_size: int = 16,
    train_split: float = 0.95,
    sample_rate: int = 16000,
    num_workers: int = 0,
    seed: int = 42,
    precomputed_tokens: Optional[Dict[str, np.ndarray]] = None,
    device: str = "cuda",
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    full_dataset = InfoREHuBERTDataset(
        data_dir=data_dir,
        hubert_discrete=hubert_discrete,
        sample_rate=sample_rate,
        precomputed_tokens=precomputed_tokens,
        device=device,
    )
    
    n_total = len(full_dataset)
    n_train = int(n_total * train_split)
    n_val = n_total - n_train
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val], generator=generator
    )
    
    print(f"Train samples: {n_train}, Validation samples: {n_val}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    return train_loader, val_loader


# ================================================================
# Precompute Tokens
# ================================================================
def precompute_all_tokens(
    data_dir: Path,
    hubert_discrete: HubertDiscrete,
    output_path: Optional[Path] = None,
    sample_rate: int = 16000,
    device: str = "cuda",
) -> Dict[str, np.ndarray]:
    """Precompute HuBERT tokens for all audio files."""
    data_dir = Path(data_dir)
    wav_files = sorted(list(data_dir.glob("*.wav")))
    
    print(f"Precomputing tokens for {len(wav_files)} files...")
    
    tokens_dict = {}
    for wav_path in tqdm(wav_files, desc="Extracting tokens"):
        filename = wav_path.stem
        wav, _ = load_audio(str(wav_path), sample_rate)
        wav_tensor = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        tokens = hubert_discrete.units(wav_tensor).cpu().numpy()
        tokens_dict[filename] = tokens
    
    if output_path:
        output_path = Path(output_path)
        np.savez(output_path, **tokens_dict)
        print(f"✅ Saved precomputed tokens to {output_path}")
    
    return tokens_dict


def load_precomputed_tokens(path: Path) -> Dict[str, np.ndarray]:
    """Load precomputed tokens from npz file."""
    data = np.load(path)
    return {key: data[key] for key in data.files}


# ================================================================
# Load from W&B Artifact
# ================================================================
def load_hubert_from_artifact(
    model_artifact: str = None,
    kmeans_artifact: str = None,
    wandb_run=None,
    device: str = "cuda",
) -> HubertDiscrete:
    """
    Load HubertDiscrete from W&B artifacts.
    
    Args:
        model_artifact: W&B artifact path for HuBERT model (defaults to env var or default)
        kmeans_artifact: W&B artifact path for KMeans model (defaults to env var or default)
        wandb_run: W&B run object
        device: Device to load model on
        
    Returns:
        HubertDiscrete model ready for inference
    """
    import wandb
    from config_helper import get_wandb_project
    
    # Use env vars or defaults if not provided
    if model_artifact is None:
        model_artifact = get_wandb_artifact_path(
            "hubert_model",
            "spirit-vilm/fine-tune-hubert-vietnamese-layer7/hubert_best_layer7_step_8400:v0"
        )
    if kmeans_artifact is None:
        kmeans_artifact = get_wandb_artifact_path(
            "kmeans_model",
            "spirit-vilm/fine-tune-hubert-vietnamese-layer7/kmeans_model_layer7:v0"
        )
    
    if wandb_run is None:
        wandb.login()
        wandb_run = wandb.init(project=get_wandb_project(), job_type="model-loading")
    
    # Download model artifact
    print(f"Downloading HuBERT model: {model_artifact}")
    artifact = wandb_run.use_artifact(model_artifact, type='model')
    model_dir = artifact.download()
    
    pt_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
    model_path = os.path.join(model_dir, pt_files[0] if pt_files else "hubert_best_layer7.pt")
    
    # Download KMeans artifact
    print(f"Downloading KMeans model: {kmeans_artifact}")
    artifact = wandb_run.use_artifact(kmeans_artifact, type='model')
    kmeans_dir = artifact.download()
    kmeans_path = os.path.join(kmeans_dir, "kmeans_model.joblib")
    
    # Use HubertDiscrete.from_pretrained()
    hubert_discrete = HubertDiscrete.from_pretrained(
        model_path=model_path,
        kmeans_path=kmeans_path,
        layer=FEATURE_LAYER,
        device=device,
    )
    
    return hubert_discrete


if __name__ == "__main__":
    print("Testing InfoRE dataset loading...")
    data_dir = download_infore_data(Path("./train_data"))
    
    print(f"\nData directory: {data_dir}")
    wav_files = list(data_dir.glob("*.wav"))
    print(f"Found {len(wav_files)} wav files")
    
    if wav_files:
        wav, sr = load_audio(str(wav_files[0]), 16000)
        print(f"Sample audio shape: {wav.shape}, sample rate: {sr}")
