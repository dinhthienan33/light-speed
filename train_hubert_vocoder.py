"""
Training script for HuBERT Vocoder.

Usage:
    python train_hubert_vocoder.py --config config_hubert.json --batch-size 16

This script trains a HuBERT vocoder that converts discrete HuBERT tokens to waveforms.
It loads:
- Audio files from InfoRE dataset
- HuBERT model from W&B artifact for token extraction
"""

import json
import os
from argparse import ArgumentParser
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import wandb

import commons
from config_helper import (
    get_wandb_entity,
    get_wandb_project,
    get_wandb_api_key,
    get_wandb_artifact_path,
)
from hubert_dataset import (
    download_infore_data,
    load_hubert_from_artifact,
    create_dataloaders,
    precompute_all_tokens,
    load_precomputed_tokens,
)
from hubert_models import HuBERTVocoder
from losses import discriminator_loss, feature_loss, generator_loss
from mel_processing import mel_spectrogram_torch
from models import MultiPeriodDiscriminator


# ================================================================
# Command Line Arguments
# ================================================================
parser = ArgumentParser()
parser.add_argument("--config", type=str, default="config_hubert.json")
parser.add_argument("--data-dir", type=Path, default=Path("./train_data"))
parser.add_argument("--log-dir", type=Path, default="logs_hubert")
parser.add_argument("--ckpt-dir", type=Path, default="ckpts_hubert")
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--compile", action="store_true", default=False)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--ckpt-interval", type=int, default=5000)
parser.add_argument("--eval-interval", type=int, default=1000)
parser.add_argument("--rm-old-ckpt", action="store_true", default=False)
parser.add_argument("--num-workers", type=int, default=0)  # 0 for GPU extraction
parser.add_argument("--wandb-project", type=str, default=None)
parser.add_argument("--wandb-entity", type=str, default=None)
parser.add_argument("--wandb-key", type=str, default=None)
parser.add_argument(
    "--hubert-artifact",
    type=str,
    default=None,
)
parser.add_argument(
    "--kmeans-artifact",
    type=str,
    default=None,
)
parser.add_argument("--precompute-tokens", action="store_true", default=False)
parser.add_argument("--tokens-path", type=Path, default=None)
parser.add_argument("--resume", action="store_true", default=False)
FLAGS = parser.parse_args()

# Load config
config_path = Path(__file__).parent / FLAGS.config
with open(config_path, "r", encoding="utf-8") as f:
    hps = json.load(f, object_hook=lambda x: SimpleNamespace(**x))


# ================================================================
# Distributed Training Setup
# ================================================================
if "RANK" in os.environ:
    torch.distributed.init_process_group(backend="nccl")
    RANK = int(os.environ.get("RANK", "0"))
    WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    torch.cuda.set_device(RANK)
else:
    RANK = 0
    WORLD_SIZE = 1

matplotlib.use("Agg")


# ================================================================
# CUDA & Mixed Precision Setup
# ================================================================
torch.backends.cudnn.benchmark = True
torch.manual_seed(FLAGS.seed)
torch.cuda.manual_seed(FLAGS.seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = FLAGS.device if torch.cuda.is_available() else "cpu"
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

print(f"Device: {device}, dtype: {dtype}")
d_scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
g_scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))


# ================================================================
# Logging Setup
# ================================================================
FLAGS.ckpt_dir.mkdir(exist_ok=True, parents=True)
FLAGS.log_dir.mkdir(exist_ok=True, parents=True)

if RANK == 0:
    train_writer = SummaryWriter(FLAGS.log_dir / "train", flush_secs=300)
    test_writer = SummaryWriter(FLAGS.log_dir / "test", flush_secs=300)
    
    # Initialize W&B - use args, env vars, or defaults
    wandb_project = FLAGS.wandb_project or get_wandb_project()
    wandb_entity = FLAGS.wandb_entity or get_wandb_entity()
    wandb_key = FLAGS.wandb_key or get_wandb_api_key()
    
    if wandb_key:
        wandb.login(key=wandb_key)
    else:
        wandb.login()
    
    wandb_run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=f"hubert-vocoder-{FLAGS.seed}",
        config={
            "batch_size": FLAGS.batch_size,
            "learning_rate": hps.train.learning_rate,
            "vocab_size": hps.data.vocab_size,
            "n_speakers": hps.data.n_speakers,
            "segment_size": hps.train.segment_size,
        },
    )
else:
    wandb_run = None


# ================================================================
# Data Loading
# ================================================================
print("=" * 60)
print("STAGE 1: Loading Data")
print("=" * 60)

# Download InfoRE dataset
print("Downloading InfoRE dataset...")
data_dir = download_infore_data(FLAGS.data_dir)

# Load HubertDiscrete model from W&B artifact
print("\nLoading HubertDiscrete model from W&B artifact...")
# Use args, env vars, or defaults for artifact paths
hubert_artifact = FLAGS.hubert_artifact or get_wandb_artifact_path(
    "hubert_model",
    "spirit-vilm/fine-tune-hubert-vietnamese-layer7/hubert_best_layer7_step_8400:v0"
)
kmeans_artifact = FLAGS.kmeans_artifact or get_wandb_artifact_path(
    "kmeans_model",
    "spirit-vilm/fine-tune-hubert-vietnamese-layer7/kmeans_model_layer7:v0"
)
hubert_discrete = load_hubert_from_artifact(
    model_artifact=hubert_artifact,
    kmeans_artifact=kmeans_artifact,
    wandb_run=wandb_run,
    device=device,
)

# Precompute tokens if requested
precomputed_tokens = None
if FLAGS.precompute_tokens:
    tokens_path = FLAGS.tokens_path or (FLAGS.data_dir / "precomputed_tokens.npz")
    
    if tokens_path.exists():
        print(f"Loading precomputed tokens from {tokens_path}")
        precomputed_tokens = load_precomputed_tokens(tokens_path)
    else:
        print("Precomputing tokens for all audio files...")
        precomputed_tokens = precompute_all_tokens(
            data_dir=data_dir,
            hubert_discrete=hubert_discrete,
            output_path=tokens_path,
            sample_rate=hps.data.sampling_rate,
            device=device,
        )

# Create dataloaders
print("\nCreating dataloaders...")
train_loader, val_loader = create_dataloaders(
    data_dir=data_dir,
    hubert_discrete=hubert_discrete,
    batch_size=FLAGS.batch_size,
    train_split=0.95,
    sample_rate=hps.data.sampling_rate,
    num_workers=FLAGS.num_workers,
    seed=FLAGS.seed,
    precomputed_tokens=precomputed_tokens,
    device=device,
)


# ================================================================
# Model Initialization
# ================================================================
print("\n" + "=" * 60)
print("STAGE 2: Initializing Models")
print("=" * 60)

net_g = HuBERTVocoder(
    vocab_size=hps.data.vocab_size,
    inter_channels=hps.model.inter_channels,
    hidden_channels=hps.model.hidden_channels,
    filter_channels=hps.model.filter_channels,
    n_layers=hps.model.n_layers,
    kernel_size=hps.model.kernel_size,
    p_dropout=hps.model.p_dropout,
    resblock=hps.model.resblock,
    resblock_kernel_sizes=hps.model.resblock_kernel_sizes,
    resblock_dilation_sizes=hps.model.resblock_dilation_sizes,
    upsample_rates=hps.model.upsample_rates,
    upsample_initial_channel=hps.model.upsample_initial_channel,
    upsample_kernel_sizes=hps.model.upsample_kernel_sizes,
    n_speakers=hps.data.n_speakers,
    gin_channels=hps.model.gin_channels,
    segment_size=hps.train.segment_size // hps.data.hop_length,
    hop_length=hps.data.hop_length,
    duration_dim=hps.model.duration_dim,
    duration_layers=hps.model.duration_layers,
).to(device)

net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).to(device)

print(f"Generator parameters: {sum(p.numel() for p in net_g.parameters()):,}")
print(f"Discriminator parameters: {sum(p.numel() for p in net_d.parameters()):,}")

if WORLD_SIZE > 1:
    net_g = torch.nn.parallel.DistributedDataParallel(
        net_g, device_ids=[RANK], output_device=RANK
    )
    net_d = torch.nn.parallel.DistributedDataParallel(
        net_d, device_ids=[RANK], output_device=RANK
    )

if FLAGS.compile:
    net_d = torch.compile(net_d)


# ================================================================
# Optimizers
# ================================================================
optim_g = torch.optim.AdamW(
    net_g.parameters(),
    hps.train.learning_rate,
    betas=hps.train.betas,
    eps=hps.train.eps,
)
optim_d = torch.optim.AdamW(
    net_d.parameters(),
    hps.train.learning_rate,
    betas=hps.train.betas,
    eps=hps.train.eps,
)

scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay)
scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay)


# ================================================================
# Checkpoint Loading
# ================================================================
all_ckpts = sorted(FLAGS.ckpt_dir.glob("ckpt_*.pth"))
if len(all_ckpts) > 0 and FLAGS.resume:
    ckpt = all_ckpts[-1]
    print(f"Loading checkpoint: {ckpt}")
    ckpt_data = torch.load(ckpt, map_location=device)
    net_g.load_state_dict(ckpt_data["net_g"])
    net_d.load_state_dict(ckpt_data["net_d"])
    optim_g.load_state_dict(ckpt_data["optim_g"])
    optim_d.load_state_dict(ckpt_data["optim_d"])
    scheduler_g.load_state_dict(ckpt_data["scheduler_g"])
    scheduler_d.load_state_dict(ckpt_data["scheduler_d"])
    step = ckpt_data["step"]
    d_scaler.load_state_dict(ckpt_data["d_scaler"])
    g_scaler.load_state_dict(ckpt_data["g_scaler"])
    _epoch = ckpt_data.get("epoch", 0)
    del ckpt_data
else:
    step = -1
    _epoch = 0


# ================================================================
# Helper Functions
# ================================================================
def plot_spectrogram_to_numpy(spectrogram):
    """Plot spectrogram and return as numpy array."""
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def evaluate(step):
    """Run evaluation on validation set."""
    net_g.eval()
    
    with torch.no_grad():
        for batch in val_loader:
            # Get batch data
            wav = batch["wav"].to(device)
            wav_lengths = batch["wav_lengths"].to(device)
            tokens = batch["tokens"].to(device)
            token_lengths = batch["token_lengths"].to(device)
            gt_durations = batch["durations"].to(device)
            
            # Take first sample only
            wav = wav[:1]
            wav_lengths = wav_lengths[:1]
            tokens = tokens[:1]
            token_lengths = token_lengths[:1]
            gt_durations = gt_durations[:1]
            
            # Generate
            model = net_g.module if WORLD_SIZE > 1 else net_g
            wav_hat = model.infer(tokens, token_lengths)
            
            # Compute mel spectrograms
            wav_for_mel = wav[:, :wav_lengths[0]]
            if wav_for_mel.dim() == 2:
                wav_for_mel = wav_for_mel.unsqueeze(1)  # [B, 1, T]
            
            y_mel = mel_spectrogram_torch(
                wav_for_mel.squeeze(1).float(),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            
            y_hat_mel = mel_spectrogram_torch(
                wav_hat.squeeze(1).float(),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            
            break  # Only evaluate one batch
    
    # Log to tensorboard
    if RANK == 0:
        gen_mel = plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
        gt_mel = plot_spectrogram_to_numpy(y_mel[0].cpu().numpy())
        
        test_writer.add_audio(
            "gt/audio", wav[0, :wav_lengths[0]], step, hps.data.sampling_rate
        )
        test_writer.add_audio(
            "gen/audio", wav_hat[0].squeeze(), step, hps.data.sampling_rate
        )
        test_writer.add_image("mel/generated", gen_mel, step, dataformats="HWC")
        test_writer.add_image("mel/ground_truth", gt_mel, step, dataformats="HWC")
        
        # Log to wandb
        if wandb.run is not None:
            wandb.log({
                "eval/gt_audio": wandb.Audio(
                    wav[0, :wav_lengths[0]].cpu().numpy(),
                    sample_rate=hps.data.sampling_rate,
                ),
                "eval/gen_audio": wandb.Audio(
                    wav_hat[0].squeeze().cpu().numpy(),
                    sample_rate=hps.data.sampling_rate,
                ),
                "step": step,
            })
    
    net_g.train()


# ================================================================
# Training Loop
# ================================================================
print("\n" + "=" * 60)
print("STAGE 3: Training")
print("=" * 60)

net_g.train()
net_d.train()

for epoch in range(_epoch + 1, 100_000):
    if RANK == 0:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    else:
        pbar = train_loader
    
    for batch in pbar:
        step += 1
        
        # Get batch data
        wav = batch["wav"].to(device, non_blocking=True)
        wav_lengths = batch["wav_lengths"].to(device, non_blocking=True)
        tokens = batch["tokens"].to(device, non_blocking=True)
        token_lengths = batch["token_lengths"].to(device, non_blocking=True)
        gt_durations = batch["durations"].to(device, non_blocking=True)
        
        # Add channel dimension to wav if needed
        if wav.dim() == 2:
            wav = wav.unsqueeze(1)  # [B, 1, T]
        
        # Forward pass
        with ctx:
            wav_hat, pred_durations, h, mask = net_g(
                tokens, token_lengths, gt_durations
            )
            
            # Match lengths for loss computation
            min_len = min(wav.size(-1), wav_hat.size(-1))
            wav_slice = wav[:, :, :min_len]
            wav_hat_slice = wav_hat[:, :, :min_len]
            
            # Compute mel spectrograms
            y_mel = mel_spectrogram_torch(
                wav_slice.squeeze(1).float(),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            
            y_hat_mel = mel_spectrogram_torch(
                wav_hat_slice.squeeze(1).float(),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
        
        # ================== Discriminator Update ==================
        with ctx:
            y_d_hat_r, y_d_hat_g, _, _ = net_d(wav_slice, wav_hat_slice.detach())
        
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
            y_d_hat_r, y_d_hat_g
        )
        loss_disc_all = loss_disc
        
        optim_d.zero_grad()
        d_scaler.scale(loss_disc_all).backward()
        d_scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        d_scaler.step(optim_d)
        d_scaler.update()
        
        # ================== Generator Update ==================
        with ctx:
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wav_slice, wav_hat_slice)
        
        # Mel reconstruction loss
        loss_mel = F.l1_loss(y_mel, y_hat_mel)
        
        # Duration loss
        loss_duration = F.mse_loss(pred_durations, gt_durations)
        
        # Feature matching loss
        loss_fm = feature_loss(fmap_r, fmap_g)
        
        # Generator adversarial loss
        loss_gen, losses_gen = generator_loss(y_d_hat_g)
        
        # Total generator loss
        loss_gen_all = (
            loss_gen
            + loss_fm
            + loss_mel * hps.train.c_mel
            + loss_duration * hps.train.c_duration
        )
        
        optim_g.zero_grad()
        g_scaler.scale(loss_gen_all).backward()
        g_scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        g_scaler.step(optim_g)
        g_scaler.update()
        
        # ================== Logging ==================
        if RANK == 0:
            train_writer.add_scalar("loss/disc_all", loss_disc_all.float(), step)
            train_writer.add_scalar("loss/gen_all", loss_gen_all.float(), step)
            train_writer.add_scalar("loss/gen", loss_gen.float(), step)
            train_writer.add_scalar("loss/fm", loss_fm.float(), step)
            train_writer.add_scalar("loss/mel", loss_mel.float(), step)
            train_writer.add_scalar("loss/duration", loss_duration.float(), step)
            train_writer.add_scalar("grad/norm_d", grad_norm_d, step)
            train_writer.add_scalar("grad/norm_g", grad_norm_g, step)
            
            if wandb.run is not None:
                wandb.log({
                    "train/loss_disc": loss_disc_all.float().item(),
                    "train/loss_gen": loss_gen_all.float().item(),
                    "train/loss_mel": loss_mel.float().item(),
                    "train/loss_duration": loss_duration.float().item(),
                    "train/loss_fm": loss_fm.float().item(),
                    "train/grad_norm_d": grad_norm_d,
                    "train/grad_norm_g": grad_norm_g,
                    "step": step,
                    "epoch": epoch,
                })
            
            if isinstance(pbar, tqdm):
                pbar.set_postfix({
                    "loss_g": f"{loss_gen_all.item():.3f}",
                    "loss_d": f"{loss_disc_all.item():.3f}",
                    "loss_mel": f"{loss_mel.item():.3f}",
                })
        
        # ================== Evaluation ==================
        if step % FLAGS.eval_interval == 0 and RANK == 0:
            evaluate(step)
        
        # ================== Checkpointing ==================
        if step % FLAGS.ckpt_interval == 0 and RANK == 0:
            ckpt_path = FLAGS.ckpt_dir / f"ckpt_{step:08d}.pth"
            torch.save(
                {
                    "step": step,
                    "epoch": epoch,
                    "net_g": net_g.state_dict(),
                    "net_d": net_d.state_dict(),
                    "d_scaler": d_scaler.state_dict(),
                    "g_scaler": g_scaler.state_dict(),
                    "optim_d": optim_d.state_dict(),
                    "optim_g": optim_g.state_dict(),
                    "scheduler_g": scheduler_g.state_dict(),
                    "scheduler_d": scheduler_d.state_dict(),
                },
                ckpt_path,
            )
            print(f"Saved checkpoint: {ckpt_path}")
            
            all_ckpts.append(ckpt_path)
            
            # Keep only 10 latest checkpoints
            if len(all_ckpts) >= 11 and FLAGS.rm_old_ckpt:
                all_ckpts[0].unlink()
                del all_ckpts[0]
            
            # Log checkpoint to wandb
            if wandb.run is not None:
                artifact = wandb.Artifact(
                    name=f"hubert-vocoder-ckpt-{step}",
                    type="model",
                    metadata={"step": step, "epoch": epoch},
                )
                artifact.add_file(str(ckpt_path))
                wandb.log_artifact(artifact)
    
    # Update learning rate
    if RANK == 0:
        lr = optim_g.param_groups[0]["lr"]
        train_writer.add_scalar("lr", lr, step)
        if wandb.run is not None:
            wandb.log({"lr": lr, "step": step})
    
    scheduler_g.step()
    scheduler_d.step()

# Finish wandb
if RANK == 0 and wandb.run is not None:
    wandb.finish()
