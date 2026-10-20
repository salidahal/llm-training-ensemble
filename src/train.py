# adapted from the modded-nanogpt repo: https://github.com/KellerJordan/modded-nanogpt
import os
import uuid
import glob
import time
import argparse
from dataclasses import dataclass
import tiktoken
import wandb

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

flex_attention = torch.compile(flex_attention, dynamic=False)
create_block_mask = torch.compile(create_block_mask, dynamic=False)


# -----------------------------------------------------------------------------
# Muon optimizer

def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

@torch.compile
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Quintic Newton–Schulz style iteration in bf16 for fast orthogonalization.
    Returns an approximation to the nearest orthogonal matrix.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    transposed = False
    if G.size(0) > G.size(1):
        X = X.T
        transposed = True
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + (B @ X)
    if transposed:
        X = X.T
    return X

zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)

class Muon(torch.optim.Optimizer):
    """
    Muon - Momentum + Orthogonalization post-processing for 2D parameters only.
    Supports both single-GPU and multi-GPU (DDP) training.
    """
    def __init__(self, params, lr=0.01, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=5,
                 world_size=1, rank=0, device='cuda'):
        self.world_size = world_size
        self.rank = rank
        self.device = device
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]

            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device=self.device, dtype=torch.bfloat16)

            curr_idx = 0

            for i, p in enumerate(group['params']):
                # In single-GPU mode, process all params; in DDP, distribute across ranks
                if self.world_size == 1 or i % self.world_size == self.rank:
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)

                    g = zeropower_backend(g, steps=group['backend_steps'])
                    g *= max(1.0, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr_idx:curr_idx + p.numel()] = g.flatten()

                curr_idx += p.numel()

            # Only do all_reduce if using DDP (world_size > 1)
            if self.world_size > 1:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr_idx = 0
            for p in group['params']:
                g = updates_flat[curr_idx:curr_idx + p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()

# -----------------------------------------------------------------------------
# Model components (same architecture mods)

class Rotary(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = None
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        # x: (B, T, n_head, head_dim)
        T = x.shape[1]
        if T != self.seq_len_cached:
            self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=x.device).float() / self.dim))
            self.seq_len_cached = T
            t = torch.arange(T, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=-1).type_as(x)

class CastedLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype))

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head

        self.c_q = CastedLinear(cfg.n_embd, cfg.n_embd, bias=False)
        self.c_k = CastedLinear(cfg.n_embd, cfg.n_embd, bias=False)
        self.c_v = CastedLinear(cfg.n_embd, cfg.n_embd, bias=False)

        self.c_proj = CastedLinear(cfg.n_embd, cfg.n_embd, bias=False)
        self.c_proj.weight.data.zero_()

        self.rotary = Rotary(self.head_dim)
        self.lamb = nn.Parameter(torch.tensor(0.5))  # value residual interpolation

    def forward(self, x, v1, block_mask):
        B, T, C = x.shape
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)

        if v1 is None:
            v1 = v
        v = (1 - self.lamb) * v + self.lamb * v1.view_as(v)

        cos, sin = self.rotary(q)

        # QK RMS norm
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # flex_attention expects (B, n_head, T, head_dim)
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y, v1

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = CastedLinear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.c_proj = CastedLinear(4 * cfg.n_embd, cfg.n_embd, bias=False)
        self.c_proj.weight.data.zero_()

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()  # squared ReLU
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = CausalSelfAttention(cfg)
        self.mlp = MLP(cfg)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x, v1, x0, block_mask):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        x1, v1 = self.attn(F.rms_norm(x, (x.size(-1),)), v1, block_mask)
        x = x + x1
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x, v1

@dataclass
class GPTConfig:
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 768

class GPT(nn.Module):
    """
    12-layer, 6-head, 768-dim transformer with U-Net skip connections.
    """
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.num_encoder_layers = cfg.n_layer // 2
        self.num_decoder_layers = cfg.n_layer - self.num_encoder_layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
        ))
        self.lm_head = CastedLinear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight.data.zero_()

    def forward(self, idx, target, block_mask):
        # idx: (B,T), target: (B,T)
        x = self.transformer.wte(idx)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        v1 = None

        skips = []
        for i in range(self.num_encoder_layers):
            x, v1 = self.transformer.h[i](x, v1, x0, block_mask)
            skips.append(x)

        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skips.pop()
            x, v1 = self.transformer.h[self.num_encoder_layers + i](x, v1, x0, block_mask)

        x = F.rms_norm(x, (x.size(-1),))
        logits = self.lm_head(x)
        logits = 30 * torch.tanh(logits / 30)
        logits = logits.float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        return loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50):
        """
        Generate tokens autoregressively.
        idx: (B, T) tensor of input token indices
        Returns: (B, T + max_new_tokens) tensor
        """
        for _ in range(max_new_tokens):
            # Crop to last sequence_length tokens if needed
            idx_cond = idx if idx.size(1) <= 1024 else idx[:, -1024:]

            # Get logits from forward pass (need to create block mask for current length)
            T = idx_cond.size(1)
            def causal_mask_fn(_b, _h, q_idx, kv_idx):
                return q_idx >= kv_idx
            block_mask = create_block_mask(
                causal_mask_fn, None, None, T, T, device=idx.device, _compile=True
            )

            # Forward pass to get logits
            x = self.transformer.wte(idx_cond)
            x = F.rms_norm(x, (x.size(-1),))
            x0 = x
            v1 = None

            skips = []
            for i in range(self.num_encoder_layers):
                x, v1 = self.transformer.h[i](x, v1, x0, block_mask)
                skips.append(x)

            for i in range(self.num_decoder_layers):
                x = x + self.skip_weights[i] * skips.pop()
                x, v1 = self.transformer.h[self.num_encoder_layers + i](x, v1, x0, block_mask)

            x = F.rms_norm(x, (x.size(-1),))
            logits = self.lm_head(x)
            logits = 30 * torch.tanh(logits / 30)

            # Get logits for last position only
            logits = logits[:, -1, :] / temperature

            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

        return idx

# -----------------------------------------------------------------------------
# IID 1024-token sequence loader (replaces document-stream 64K context loader)

def _peek_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    if header[0] != 20240520:
        raise RuntimeError("bad magic number; wrong .bin format?")
    if header[1] != 1:
        raise RuntimeError("unsupported version")
    return int(header[2])

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520 and header[1] == 1
        ntok = int(header[2])
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok
    return tokens

class EpochDataLoader:
    """
    Epoch-based data loader that ensures each sample is seen exactly once per epoch.
    Creates non-overlapping windows of size T+1 and shuffles their order each epoch.
    """
    def __init__(self, filename_pattern, B, T, rank, world_size, device='cuda', seed=1337, num_shards=None):
        self.B = B
        self.T = T
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.seed = seed

        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"no files match: {filename_pattern}"

        # Limit to num_shards if specified
        if num_shards is not None:
            self.files = self.files[:num_shards]

        # Load all tokens from specified shards
        self.tokens = self._load_shards()
        self.ntok_total = len(self.tokens)

        # Create non-overlapping windows of size T+1
        # Each window provides T input tokens and T target tokens (shifted by 1)
        self.num_windows = (len(self.tokens) - 1) // T

        # For DDP: each rank gets a subset of windows
        # Ensure all ranks have same number of batches by truncating
        windows_per_rank = self.num_windows // world_size
        batches_per_rank = windows_per_rank // B
        self.windows_per_rank = batches_per_rank * B  # Truncate to full batches

        # Create global indices and select this rank's portion
        self._rng = np.random.default_rng(seed)  # Same seed for all ranks initially
        self.global_indices = np.arange(self.num_windows)

        self.current_idx = 0
        self.epoch = 0
        self.epoch_complete = False

        # Shuffle and assign indices for this rank
        self._start_new_epoch()

    def _load_shards(self):
        all_tokens = []
        for f in self.files:
            tokens = _load_data_shard(f)
            all_tokens.append(tokens)
        return np.concatenate(all_tokens)

    def _start_new_epoch(self):
        """Shuffle indices and reset position for new epoch."""
        # Use epoch number in seed so each epoch has different shuffle
        # but all ranks see same global shuffle
        rng = np.random.default_rng(self.seed + self.epoch * 1000)
        rng.shuffle(self.global_indices)

        # Each rank takes its portion of shuffled indices
        start = self.rank * self.windows_per_rank
        end = start + self.windows_per_rank
        self.indices = self.global_indices[start:end].copy()

        self.current_idx = 0
        self.epoch_complete = False

    def next_batch(self):
        """
        Returns (x, y, epoch_complete) where epoch_complete signals end of epoch.
        When epoch_complete is True, x and y are still valid (last batch of epoch).
        """
        if self.current_idx + self.B > len(self.indices):
            # Start new epoch
            self.epoch += 1
            self._start_new_epoch()
            self.epoch_complete = True

        batch_indices = self.indices[self.current_idx:self.current_idx + self.B]
        self.current_idx += self.B

        x = np.empty((self.B, self.T), dtype=np.int32)
        y = np.empty((self.B, self.T), dtype=np.int32)

        for i, idx in enumerate(batch_indices):
            start = idx * self.T
            buf = self.tokens[start:start + self.T + 1].astype(np.int32)
            x[i] = buf[:-1]
            y[i] = buf[1:]

        x = torch.tensor(x, dtype=torch.long, device=self.device)
        y = torch.tensor(y, dtype=torch.long, device=self.device)

        was_complete = self.epoch_complete
        self.epoch_complete = False
        return x, y, was_complete

    def get_batches_per_epoch(self):
        """Returns number of batches per epoch (before gradient accumulation)."""
        return self.windows_per_rank // self.B


class ValidationDataLoader:
    """
    Simple validation data loader - iterates through data once without shuffling.
    """
    def __init__(self, filename_pattern, B, T, rank, world_size, device='cuda'):
        self.B = B
        self.T = T
        self.rank = rank
        self.world_size = world_size
        self.device = device

        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"no files match: {filename_pattern}"

        # Load all validation tokens
        all_tokens = []
        for f in self.files:
            tokens = _load_data_shard(f)
            all_tokens.append(tokens)
        self.tokens = np.concatenate(all_tokens)
        self.ntok_total = len(self.tokens)

        # Create non-overlapping windows
        self.num_windows = (len(self.tokens) - 1) // T

        # For DDP: each rank gets a subset
        windows_per_rank = self.num_windows // world_size
        batches_per_rank = windows_per_rank // B
        self.windows_per_rank = batches_per_rank * B

        # This rank's window indices (no shuffling for validation)
        start = rank * self.windows_per_rank
        self.indices = np.arange(start, start + self.windows_per_rank)
        self.current_idx = 0

    def reset(self):
        """Reset to beginning for new validation pass."""
        self.current_idx = 0

    def next_batch(self):
        """Returns (x, y) or (None, None) if exhausted."""
        if self.current_idx + self.B > len(self.indices):
            return None, None

        batch_indices = self.indices[self.current_idx:self.current_idx + self.B]
        self.current_idx += self.B

        x = np.empty((self.B, self.T), dtype=np.int32)
        y = np.empty((self.B, self.T), dtype=np.int32)

        for i, idx in enumerate(batch_indices):
            start = idx * self.T
            buf = self.tokens[start:start + self.T + 1].astype(np.int32)
            x[i] = buf[:-1]
            y[i] = buf[1:]

        x = torch.tensor(x, dtype=torch.long, device=self.device)
        y = torch.tensor(y, dtype=torch.long, device=self.device)
        return x, y

    def get_num_batches(self):
        """Returns total number of validation batches."""
        return self.windows_per_rank // self.B

# -----------------------------------------------------------------------------
# Training script

@dataclass
class Config:
    """Training configuration. CLI arguments override these defaults."""
    # data paths
    input_bin: str = "edu_fineweb_fixed/edufineweb_train_*.bin"
    input_val_bin: str = "edu_fineweb_fixed/edufineweb_val_*.bin"
    input_test_bin: str = "edu_fineweb_fixed/edufineweb_test_*.bin"

    # model
    sequence_length: int = 1024

    # training (CLI-configurable)
    num_epochs: int = 1
    num_shards: int = 1
    device_batch_size: int = 64
    global_batch_size: int = 512  # effective batch size across all devices
    # grad_accum_steps: int = 1     # computed: global_batch_size // (device_batch_size * world_size)

    # early stopping (CLI-configurable)
    early_stopping: bool = True
    patience: int = 3
    save_best: bool = True

    # validation/checkpointing (CLI-configurable)
    run_name: str = ""            # prefix for checkpoint files
    val_loss_every: int = 0       # 0 = end of epoch only
    save_every: int = 0           # 0 = end of epoch only
    print_every: int = 10         # print training loss every N steps
    checkpoint_dir: str = "checkpoints"

    # text generation samples (CLI-configurable)
    sample_every: int = 0         # 0 = disabled
    sample_prompt: str = "The meaning of life is"
    sample_tokens: int = 64

    # optimization (fixed - from paper)
    warmup_steps: int = 10
    weight_decay: float = 0.05
    lr_embed: float = 0.15
    lr_lm_head: float = 0.002
    lr_muon: float = 0.01
    lr_scalars: float = 0.01
    muon_momentum_start: float = 0.85
    muon_momentum_end: float = 0.95
    muon_momentum_warmup_steps: int = 500

    # wandb (CLI-configurable)
    use_wandb: bool = True
    wandb_project: str = "llm-training"

    # test evaluation
    run_test: bool = True


def parse_args():
    parser = argparse.ArgumentParser(description="Train LLM on FineWeb-Edu data")

    # Training
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs to train (default: 5)")
    parser.add_argument("--shards", type=int, default=1,
                        help="Number of data shards to use, 1 shard ≈ 50M tokens (default: 1)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Device batch size - sequences per device (default: 128)")
    parser.add_argument("--global-batch-size", type=int, default=512,
                        help="Global batch size - effective batch across all devices (default: 512)")

    # Early stopping
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience - epochs without improvement (default: 3)")
    parser.add_argument("--no-early-stopping", action="store_true",
                        help="Disable early stopping")

    # Validation/Checkpointing
    parser.add_argument("--run-name", type=str, default=None,
                        help="Prefix for checkpoint files (default: auto-generated from config)")
    parser.add_argument("--val-every", type=int, default=10,
                        help="Validate every N steps, 0 = end of epoch only (default: 0)")
    parser.add_argument("--save-every", type=int, default=0,
                        help="Save checkpoint every N steps, 0 = end of epoch only (default: 0)")
    parser.add_argument("--print-every", type=int, default=10,
                        help="Print training loss every N steps (default: 10)")
    parser.add_argument("--sample-every", type=int, default=0,
                        help="Generate text samples every N steps to see improvement, 0 = disabled (default: 0)")
    parser.add_argument("--sample-prompt", type=str, default="The meaning of life is",
                        help="Prompt for text generation samples (default: 'The meaning of life is')")
    parser.add_argument("--sample-tokens", type=int, default=64,
                        help="Number of tokens to generate per sample (default: 64)")

    # Logging
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable wandb logging")

    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    # Test evaluation
    parser.add_argument("--test-data", type=str, default=None,
                        help="Path pattern for test data (default: edu_fineweb_fixed/edufineweb_test_*.bin)")
    parser.add_argument("--no-test", action="store_true",
                        help="Skip final test set evaluation")

    return parser.parse_args()

def save_checkpoint(model, optimizers, schedulers, step, val_loss, args, checkpoint_path, is_ddp=False):
    """Save model checkpoint."""
    raw_model = model.module if is_ddp else model
    # Handle compiled model
    if hasattr(raw_model, '_orig_mod'):
        raw_model = raw_model._orig_mod

    checkpoint = {
        'step': step,
        'val_loss': val_loss,
        'model_state_dict': raw_model.state_dict(),
        'optimizer_states': [opt.state_dict() for opt in optimizers],
        'scheduler_states': [sch.state_dict() for sch in schedulers],
        'args': args,
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizers, schedulers, device, is_ddp=False):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    raw_model = model.module if is_ddp else model
    # Handle compiled model
    if hasattr(raw_model, '_orig_mod'):
        raw_model = raw_model._orig_mod

    raw_model.load_state_dict(checkpoint['model_state_dict'])

    for opt, state in zip(optimizers, checkpoint['optimizer_states']):
        opt.load_state_dict(state)

    for sch, state in zip(schedulers, checkpoint['scheduler_states']):
        sch.load_state_dict(state)

    return checkpoint['step'], checkpoint['val_loss']


def main():
    # Parse command line arguments
    cli_args = parse_args()

    # Generate run name if not provided (e.g., "s1_e5" for 1 shard, 5 epochs)
    run_name = cli_args.run_name
    if run_name is None:
        run_name = f"s{cli_args.shards}_e{cli_args.epochs}"

    WANDB_AVAILABLE=True

    # Create config from CLI arguments
    args = Config(
        num_epochs=cli_args.epochs,
        num_shards=cli_args.shards,
        device_batch_size=cli_args.batch_size,
        global_batch_size=cli_args.global_batch_size,
        run_name=run_name,
        patience=cli_args.patience,
        early_stopping=not cli_args.no_early_stopping,
        val_loss_every=cli_args.val_every,
        save_every=cli_args.save_every,
        print_every=cli_args.print_every,
        sample_every=cli_args.sample_every,
        sample_prompt=cli_args.sample_prompt,
        sample_tokens=cli_args.sample_tokens,
        use_wandb=not cli_args.no_wandb and WANDB_AVAILABLE,
        input_test_bin=cli_args.test_data if cli_args.test_data else "edu_fineweb_fixed/edufineweb_test_*.bin",
        run_test=not cli_args.no_test,
    )

    # -------------------------------------------------------------------------
    # Device and distributed setup
    # -------------------------------------------------------------------------
    assert torch.cuda.is_available(), "CUDA is required for training"

    # Check if running with torchrun/DDP
    use_ddp = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ

    if use_ddp:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        master = (rank == 0)
        print(f"[DDP] Rank {rank}/{world_size} using device {device}")
    else:
        # Single GPU mode
        num_gpus = torch.cuda.device_count()
        print(f"Running in single-GPU mode. {num_gpus} GPU(s) available.")
        rank = 0
        local_rank = 0
        world_size = 1
        device = "cuda:0"
        torch.cuda.set_device(device)
        master = True

    # -------------------------------------------------------------------------
    # Logging setup
    # -------------------------------------------------------------------------
    run_id = str(uuid.uuid4())[:8]
    logdir = f"logs/{args.run_name}_{run_id}"

    # Compute grad accumulation steps early (needed for wandb config)
    B = args.device_batch_size
    T = args.sequence_length
    assert args.global_batch_size % (B * world_size) == 0, \
        f"global_batch_size ({args.global_batch_size}) must be divisible by B*world_size ({B*world_size})"
    grad_accum_steps = args.global_batch_size // (B * world_size)

    if master:
        os.makedirs(logdir, exist_ok=True)
        os.makedirs(args.checkpoint_dir, exist_ok=True)

        # Initialize wandb
        if args.use_wandb:
            wandb.init(
                project=args.wandb_project,
                name=f"{args.run_name}_{run_id}",
                config={
                    "run_name": args.run_name,
                    "num_epochs": args.num_epochs,
                    "num_shards": args.num_shards,
                    "sequence_length": args.sequence_length,
                    "device_batch_size": args.device_batch_size,
                    "global_batch_size": args.global_batch_size,
                    "grad_accum_steps": grad_accum_steps,
                    "early_stopping": args.early_stopping,
                    "patience": args.patience,
                    "lr_embed": args.lr_embed,
                    "lr_lm_head": args.lr_lm_head,
                    "lr_muon": args.lr_muon,
                    "lr_scalars": args.lr_scalars,
                    "world_size": world_size,
                }
            )

    if use_ddp:
        dist.barrier()

    def print0(msg):
        if master:
            print(msg, flush=True)

    # -------------------------------------------------------------------------
    # Data loaders
    # -------------------------------------------------------------------------
    train_loader = EpochDataLoader(
        args.input_bin, B, T, rank, world_size,
        device=device, num_shards=args.num_shards
    )
    val_loader = ValidationDataLoader(
        args.input_val_bin, B, T, rank, world_size, device=device
    )

    # Test loader (optional - may not exist)
    test_loader = None
    if args.run_test:
        test_files = sorted(glob.glob(args.input_test_bin))
        if test_files:
            test_loader = ValidationDataLoader(
                args.input_test_bin, B, T, rank, world_size, device=device
            )
        else:
            print0(f"Warning: No test files found matching '{args.input_test_bin}', skipping test evaluation")

    batches_per_epoch = train_loader.get_batches_per_epoch()
    steps_per_epoch = batches_per_epoch // grad_accum_steps
    val_batches = val_loader.get_num_batches()

    print0(f"=" * 60)
    print0(f"Training Configuration:")
    print0(f"  Run name: {args.run_name}")
    print0(f"  Epochs: {args.num_epochs}")
    print0(f"  Data shards: {args.num_shards}")
    print0(f"  Train tokens (loaded): {train_loader.ntok_total:,}")
    print0(f"  Batches per epoch: {batches_per_epoch:,}")
    print0(f"  Steps per epoch: {steps_per_epoch:,} (after grad accum)")
    print0(f"  Device batch size: {B}")
    print0(f"  Gradient accumulation steps: {grad_accum_steps}")
    print0(f"  World size: {world_size}")
    print0(f"  Early stopping: {'enabled (patience=' + str(args.patience) + ')' if args.early_stopping else 'disabled'}")
    print0(f"  WandB: {'enabled' if args.use_wandb else 'disabled'}")
    if args.sample_every > 0:
        print0(f"  Text samples: every {args.sample_every} steps (prompt: '{args.sample_prompt[:30]}...')" if len(args.sample_prompt) > 30 else f"  Text samples: every {args.sample_every} steps (prompt: '{args.sample_prompt}')")
    else:
        print0(f"  Text samples: end of epoch only")
    print0(f"=" * 60)
    print0(f"Validation: {val_batches} batches, {val_loader.ntok_total:,} tokens")
    if test_loader:
        print0(f"Test: {test_loader.get_num_batches()} batches, {test_loader.ntok_total:,} tokens")
    else:
        print0(f"Test: disabled or no test files found")

    # -------------------------------------------------------------------------
    # Build causal mask for flex_attention
    # -------------------------------------------------------------------------
    def causal_mask_fn(_b, _h, q_idx, kv_idx):
        return q_idx >= kv_idx

    block_mask = create_block_mask(
        causal_mask_fn, None, None, T, T, device=device, _compile=True
    )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    cfg = GPTConfig(vocab_size=50304, n_layer=12, n_head=6, n_embd=768)
    model = GPT(cfg).to(device).bfloat16()

    # Preserve the speedrun casting behavior: keep CastedLinear weights in fp32
    for m in model.modules():
        if isinstance(m, CastedLinear):
            m.float()

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print0(f"Model parameters: {num_params:,} ({num_params/1e6:.1f}M)")

    model = torch.compile(model)

    if use_ddp:
        model = DDP(model, device_ids=[local_rank])
        raw_model = model.module
    else:
        raw_model = model

    # Handle compiled model for accessing parameters
    if hasattr(raw_model, '_orig_mod'):
        param_model = raw_model._orig_mod
    else:
        param_model = raw_model

    # -------------------------------------------------------------------------
    # Optimizers
    # -------------------------------------------------------------------------
    opt_embed = torch.optim.AdamW(
        [param_model.transformer.wte.weight],
        lr=args.lr_embed,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        fused=True
    )
    opt_head = torch.optim.AdamW(
        [param_model.lm_head.weight],
        lr=args.lr_lm_head,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        fused=True
    )

    block_params = list(param_model.transformer.h.parameters())
    matrix_params = [p for p in block_params if p.ndim == 2]
    scalar_params = [p for p in block_params if p.ndim < 2] + [param_model.skip_weights]

    opt_muon = Muon(
        matrix_params,
        lr=args.lr_muon,
        momentum=args.muon_momentum_start,
        nesterov=True,
        backend="newtonschulz5",
        backend_steps=5,
        world_size=world_size,
        rank=rank,
        device=device
    )
    opt_scalars = torch.optim.AdamW(
        scalar_params,
        lr=args.lr_scalars,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        fused=True
    )

    optimizers = [opt_embed, opt_head, opt_muon, opt_scalars]

    # LR schedule: no warmdown (paper). Optional warmup only.
    def lr_mult(step):
        if args.warmup_steps > 0 and step < args.warmup_steps:
            return float(step + 1) / float(args.warmup_steps)
        return 1.0

    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, lr_mult) for opt in optimizers]

    # -------------------------------------------------------------------------
    # Resume from checkpoint if specified
    # -------------------------------------------------------------------------
    start_step = 0
    start_epoch = 0
    best_val_loss = float('inf')

    if cli_args.resume:
        if os.path.exists(cli_args.resume):
            print0(f"Resuming from checkpoint: {cli_args.resume}")
            start_step, loaded_val_loss = load_checkpoint(
                cli_args.resume, model, optimizers, schedulers, device, is_ddp=use_ddp
            )
            best_val_loss = loaded_val_loss
            start_epoch = start_step // steps_per_epoch
            print0(f"Resumed from step {start_step} (epoch {start_epoch}), val_loss {loaded_val_loss:.4f}")
        else:
            print0(f"Warning: checkpoint {cli_args.resume} not found, starting fresh")

    # -------------------------------------------------------------------------
    # Validation function
    # -------------------------------------------------------------------------
    def run_validation():
        model.eval()
        val_loader.reset()
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            while True:
                xv, yv = val_loader.next_batch()
                if xv is None:
                    break
                val_loss_sum += model(xv, yv, block_mask)
                val_count += 1

        if use_ddp:
            val_loss_tensor = torch.tensor([val_loss_sum, val_count], device=device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            val_loss_sum, val_count = val_loss_tensor[0].item(), val_loss_tensor[1].item()

        return val_loss_sum / val_count if val_count > 0 else float('inf')

    def run_test():
        """Run evaluation on test set."""
        if test_loader is None:
            return None

        model.eval()
        test_loader.reset()
        test_loss_sum = 0.0
        test_count = 0

        with torch.no_grad():
            while True:
                xt, yt = test_loader.next_batch()
                if xt is None:
                    break
                test_loss_sum += model(xt, yt, block_mask)
                test_count += 1

        if use_ddp:
            test_loss_tensor = torch.tensor([test_loss_sum, test_count], device=device)
            dist.all_reduce(test_loss_tensor, op=dist.ReduceOp.SUM)
            test_loss_sum, test_count = test_loss_tensor[0].item(), test_loss_tensor[1].item()

        return test_loss_sum / test_count if test_count > 0 else float('inf')

    # -------------------------------------------------------------------------
    # Text generation sampling function
    # -------------------------------------------------------------------------
    TIKTOKEN_AVAILABLE = True
    enc = tiktoken.get_encoding("gpt2") if TIKTOKEN_AVAILABLE else None

    def generate_sample(step, prompt_text):
        """Generate a text sample from the model to visualize training progress."""
        if not TIKTOKEN_AVAILABLE:
            print0("  [sample] tiktoken not available, skipping generation")
            return

        model.eval()

        # Get the underlying model for generation
        gen_model = model.module if use_ddp else model
        if hasattr(gen_model, '_orig_mod'):
            gen_model = gen_model._orig_mod

        # Encode prompt
        prompt_tokens = enc.encode(prompt_text)
        x = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

        # Generate
        with torch.no_grad():
            output_tokens = gen_model.generate(x, max_new_tokens=args.sample_tokens, temperature=0.8, top_k=50)

        # Decode and print
        generated_text = enc.decode(output_tokens[0].tolist())

        print0(f"\n  {'─'*50}")
        print0(f"  [step {step}] Sample generation:")
        print0(f"  {'─'*50}")
        # Print with some formatting - wrap long lines
        lines = generated_text.split('\n')
        for line in lines[:10]:  # Limit to 10 lines
            if len(line) > 80:
                print0(f"  {line[:80]}...")
            else:
                print0(f"  {line}")
        if len(lines) > 10:
            print0(f"  ... ({len(lines) - 10} more lines)")
        print0(f"  {'─'*50}\n")

        # Log to wandb if available
        if master and args.use_wandb:
            wandb.log({"sample_text": wandb.Html(f"<pre>{generated_text}</pre>"), "step": step})

        model.train()

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    training_time_ms = 0.0
    global_step = start_step
    patience_counter = 0
    stopped_early = False

    # Generate initial sample to show untrained model output (baseline)
    if master and TIKTOKEN_AVAILABLE and start_step == 0:
        print0("\nGenerating initial sample (untrained model baseline):")
        generate_sample(0, args.sample_prompt)

    torch.cuda.synchronize()
    t0 = time.time()

    for epoch in range(start_epoch, args.num_epochs):
        print0(f"\n{'='*60}")
        print0(f"Epoch {epoch + 1}/{args.num_epochs}")
        print0(f"{'='*60}")

        epoch_step = 0
        epoch_loss_sum = 0.0

        while True:
            # -----------------------------------------------------------------
            # Get next batch
            # -----------------------------------------------------------------
            x, y, epoch_complete = train_loader.next_batch()

            # -----------------------------------------------------------------
            # Mid-epoch validation (if configured)
            # -----------------------------------------------------------------
            if args.val_loss_every > 0 and global_step > 0 and global_step % args.val_loss_every == 0:
                torch.cuda.synchronize()
                training_time_ms += 1000.0 * (time.time() - t0)

                val_loss_value = run_validation()
                print0(f"  [step {global_step}] val_loss {val_loss_value:.4f}")

                if master and args.use_wandb:
                    wandb.log({"val_loss": val_loss_value, "step": global_step, "epoch": epoch})

                # Check for improvement
                if val_loss_value < best_val_loss:
                    best_val_loss = val_loss_value
                    patience_counter = 0
                    if master and args.save_best:
                        best_path = os.path.join(args.checkpoint_dir, f"{args.run_name}_best.pt")
                        save_checkpoint(model, optimizers, schedulers, global_step, val_loss_value, args, best_path, is_ddp=use_ddp)
                        print0(f"  -> New best model saved (val_loss={val_loss_value:.4f})")
                else:
                    patience_counter += 1
                    if args.early_stopping and patience_counter >= args.patience:
                        print0(f"  -> Early stopping triggered (no improvement for {args.patience} validations)")
                        stopped_early = True
                        break

                torch.cuda.synchronize()
                t0 = time.time()

            # -----------------------------------------------------------------
            # Text generation sampling (if configured)
            # -----------------------------------------------------------------
            if master and args.sample_every > 0 and global_step > 0 and global_step % args.sample_every == 0:
                torch.cuda.synchronize()
                training_time_ms += 1000.0 * (time.time() - t0)

                generate_sample(global_step, args.sample_prompt)

                torch.cuda.synchronize()
                t0 = time.time()

            # -----------------------------------------------------------------
            # Mid-epoch checkpointing (if configured)
            # -----------------------------------------------------------------
            if master and args.save_every > 0 and global_step > 0 and global_step % args.save_every == 0:
                ckpt_path = os.path.join(args.checkpoint_dir, f"{args.run_name}_step_{global_step:06d}.pt")
                save_checkpoint(model, optimizers, schedulers, global_step, best_val_loss, args, ckpt_path, is_ddp=use_ddp)
                print0(f"  -> Checkpoint saved: {ckpt_path}")

            # -----------------------------------------------------------------
            # Training step
            # -----------------------------------------------------------------
            model.train()

            for micro in range(grad_accum_steps):
                loss = model(x, y, block_mask)
                train_loss = loss.detach()

                if micro < grad_accum_steps - 1:
                    x, y, micro_epoch_complete = train_loader.next_batch()
                    if micro_epoch_complete:
                        epoch_complete = True

                if use_ddp and micro < grad_accum_steps - 1:
                    with model.no_sync():
                        loss.backward()
                else:
                    loss.backward()

            # Average grads
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.div_(grad_accum_steps)

            # Muon momentum warmup: 0.85 -> 0.95 over first 500 steps
            frac = min(global_step / float(args.muon_momentum_warmup_steps), 1.0)
            opt_muon.param_groups[0]["momentum"] = (1 - frac) * args.muon_momentum_start + frac * args.muon_momentum_end

            for opt, sch in zip(optimizers, schedulers):
                opt.step()
                sch.step()

            model.zero_grad(set_to_none=True)

            train_loss_value = train_loss.item()
            epoch_loss_sum += train_loss_value
            epoch_step += 1
            global_step += 1

            # Log training loss to wandb
            if master and args.use_wandb and global_step % 5 == 0:
                wandb.log({
                    "train_loss": train_loss_value,
                    "step": global_step,
                    "epoch": epoch,
                    "lr_embed": schedulers[0].get_last_lr()[0],
                    "muon_momentum": opt_muon.param_groups[0]["momentum"],
                })

            # Print progress
            approx_time = training_time_ms + 1000.0 * (time.time() - t0)
            if epoch_step % args.print_every == 0 or epoch_step <= 3:
                print0(f"  step {epoch_step:>5}/{steps_per_epoch}  train_loss {train_loss_value:.4f}  time {approx_time/1000:.1f}s")

            # Check if epoch complete
            if epoch_complete:
                break

        if stopped_early:
            break

        # ---------------------------------------------------------------------
        # End of epoch validation
        # ---------------------------------------------------------------------
        torch.cuda.synchronize()
        training_time_ms += 1000.0 * (time.time() - t0)

        avg_epoch_loss = epoch_loss_sum / epoch_step if epoch_step > 0 else 0
        val_loss_value = run_validation()

        print0(f"\nEpoch {epoch + 1} complete:")
        print0(f"  avg_train_loss: {avg_epoch_loss:.4f}")
        print0(f"  val_loss: {val_loss_value:.4f}")
        print0(f"  time: {training_time_ms/1000:.1f}s")

        # Generate sample at end of epoch to show progress
        if master and TIKTOKEN_AVAILABLE:
            generate_sample(global_step, args.sample_prompt)

        if master and args.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "epoch_train_loss": avg_epoch_loss,
                "val_loss": val_loss_value,
                "step": global_step,
                "training_time_s": training_time_ms / 1000,
            })

        # Check for improvement
        if val_loss_value < best_val_loss:
            best_val_loss = val_loss_value
            patience_counter = 0
            if master and args.save_best:
                best_path = os.path.join(args.checkpoint_dir, f"{args.run_name}_best.pt")
                save_checkpoint(model, optimizers, schedulers, global_step, val_loss_value, args, best_path, is_ddp=use_ddp)
                print0(f"  -> New best model saved (val_loss={val_loss_value:.4f})")
        else:
            patience_counter += 1
            print0(f"  -> No improvement ({patience_counter}/{args.patience})")
            if args.early_stopping and patience_counter >= args.patience:
                print0(f"  -> Early stopping triggered (no improvement for {args.patience} epochs)")
                stopped_early = True

        # Save epoch checkpoint
        if master:
            ckpt_path = os.path.join(args.checkpoint_dir, f"{args.run_name}_epoch_{epoch+1:02d}.pt")
            save_checkpoint(model, optimizers, schedulers, global_step, val_loss_value, args, ckpt_path, is_ddp=use_ddp)
            print0(f"  -> Epoch checkpoint saved: {ckpt_path}")

        torch.cuda.synchronize()
        t0 = time.time()

        if stopped_early:
            break

    # -------------------------------------------------------------------------
    # Final test evaluation
    # -------------------------------------------------------------------------
    test_loss_value = None
    if test_loader is not None:
        print0(f"\n{'='*60}")
        print0(f"Running final test set evaluation...")
        test_loss_value = run_test()
        print0(f"  Test loss: {test_loss_value:.4f}")
        print0(f"{'='*60}")

        if master and args.use_wandb:
            wandb.log({
                "test_loss": test_loss_value,
                "step": global_step,
            })

    # -------------------------------------------------------------------------
    # Final save and cleanup
    # -------------------------------------------------------------------------
    if master:
        final_path = os.path.join(args.checkpoint_dir, f"{args.run_name}_final.pt")
        save_checkpoint(model, optimizers, schedulers, global_step, best_val_loss, args, final_path, is_ddp=use_ddp)
        print0(f"\n{'='*60}")
        print0(f"Training complete!")
        print0(f"  Final model saved: {final_path}")
        print0(f"  Best val_loss: {best_val_loss:.4f}")
        if test_loss_value is not None:
            print0(f"  Test loss: {test_loss_value:.4f}")
        print0(f"  Total steps: {global_step}")
        print0(f"  Peak memory: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
        if stopped_early:
            print0(f"  (Stopped early at epoch {epoch + 1})")
        print0(f"{'='*60}")

        if args.use_wandb:
            wandb.finish()

    if use_ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
