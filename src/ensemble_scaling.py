"""
Ensemble Scaling Analysis - Show how ensembling 1,2,3,4 models improves loss/perplexity.

For each token budget (50M, 100M, 200M):
  - Evaluate all 4 individual models
  - Evaluate all C(4,2)=6 pairs, all C(4,3)=4 triples, and the full quartet
  - Plot mean loss/perplexity vs ensemble size with std error bars

Usage:
    python src/ensemble_scaling.py
    python src/ensemble_scaling.py --batch_size 2   # reduce if OOM
    python src/ensemble_scaling.py --max_batches 50  # quick test run
"""

import argparse
import json
import os
from itertools import combinations
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Model Definition (same as ensemble_evaluate.py)

class Rotary(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = None
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, dtype):
        T = x.shape[1]
        if T != self.seq_len_cached:
            self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=x.device).float() / self.dim))
            self.seq_len_cached = T
            t = torch.arange(T, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            self.cos_cached = freqs.cos().to(dtype)
            self.sin_cached = freqs.sin().to(dtype)
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]


def apply_rotary_emb(x, cos, sin):
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
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.c_q = CastedLinear(cfg.n_embd, cfg.n_embd, bias=False)
        self.c_k = CastedLinear(cfg.n_embd, cfg.n_embd, bias=False)
        self.c_v = CastedLinear(cfg.n_embd, cfg.n_embd, bias=False)
        self.c_proj = CastedLinear(cfg.n_embd, cfg.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim)
        self.lamb = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, v1):
        B, T, C = x.shape
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        if v1 is None:
            v1 = v
        v = (1 - self.lamb) * v + self.lamb * v1.view_as(v)
        cos, sin = self.rotary(q, x.dtype)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y), v1


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = CastedLinear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.c_proj = CastedLinear(4 * cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = CausalSelfAttention(cfg)
        self.mlp = MLP(cfg)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x, v1, x0):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        x1, v1 = self.attn(F.rms_norm(x, (x.size(-1),)), v1)
        x = x + x1
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x, v1


@dataclass
class GPTConfig:
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 768

Config = GPTConfig


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.num_encoder_layers = cfg.n_layer // 2
        self.num_decoder_layers = cfg.n_layer - self.num_encoder_layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
        ))
        self.lm_head = CastedLinear(cfg.n_embd, cfg.vocab_size, bias=False)

    def forward(self, idx):
        x = self.transformer.wte(idx)
        x = F.rms_norm(x, (x.size(-1),))
        x0, v1 = x, None
        skips = []
        for i in range(self.num_encoder_layers):
            x, v1 = self.transformer.h[i](x, v1, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skips.pop()
            x, v1 = self.transformer.h[self.num_encoder_layers + i](x, v1, x0)
        x = F.rms_norm(x, (x.size(-1),))
        logits = self.lm_head(x)
        logits = 30 * torch.tanh(logits / 30)
        return logits


# -----------------------------------------------------------------------------
# Data Loading

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520 and header[1] == 1
        ntok = int(header[2])
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok
    return tokens


def load_all_tokens(data_pattern):
    """Load and concatenate all tokens matching the pattern."""
    import glob as glob_module
    files = sorted(glob_module.glob(data_pattern))
    if not files:
        raise FileNotFoundError(f"No files found matching: {data_pattern}")
    all_tokens = []
    for f in files:
        all_tokens.append(_load_data_shard(f))
    tokens = np.concatenate(all_tokens)
    print(f"  Loaded {len(tokens):,} tokens from {len(files)} file(s)")
    return tokens


# -----------------------------------------------------------------------------
# Model Loading

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(checkpoint_path, device, dtype=torch.float32):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = GPTConfig()
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        if v.dtype == torch.bfloat16:
            v = v.to(dtype)
        new_state_dict[k] = v
    model = GPT(cfg)
    model.load_state_dict(new_state_dict)
    model = model.to(device).to(dtype)
    model.eval()
    return model


# -----------------------------------------------------------------------------
# Evaluation

@torch.no_grad()
def evaluate_loss(models: List[nn.Module], tokens: np.ndarray, device: str,
                  batch_size: int = 4, seq_len: int = 1024, max_batches: int = None):
    """
    Evaluate ensemble loss via logit averaging.
    If models has 1 element, evaluates that single model.
    Returns average cross-entropy loss (per token).
    """
    tokens_t = torch.tensor(tokens, dtype=torch.long)
    n_windows = len(tokens_t) // (seq_len + 1)
    n_models = len(models)

    total_loss = 0.0
    total_tokens = 0
    batch_count = 0

    for i in range(0, n_windows, batch_size):
        if max_batches and batch_count >= max_batches:
            break

        batch_end = min(i + batch_size, n_windows)
        x_list, y_list = [], []
        for j in range(i, batch_end):
            start = j * (seq_len + 1)
            window = tokens_t[start:start + seq_len + 1]
            x_list.append(window[:-1])
            y_list.append(window[1:])

        x = torch.stack(x_list).to(device)
        y = torch.stack(y_list).to(device)

        # Logit averaging
        ensemble_logits = None
        for model in models:
            logits = model(x)
            if ensemble_logits is None:
                ensemble_logits = logits
            else:
                ensemble_logits = ensemble_logits + logits
        ensemble_logits = ensemble_logits / n_models

        loss = F.cross_entropy(
            ensemble_logits.float().view(-1, ensemble_logits.size(-1)),
            y.view(-1),
            reduction='sum'
        )
        total_loss += loss.item()
        total_tokens += y.numel()
        batch_count += 1

    avg_loss = total_loss / total_tokens
    return avg_loss


# -----------------------------------------------------------------------------
# Main

def main():
    parser = argparse.ArgumentParser(description="Ensemble scaling analysis")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory containing model checkpoints")
    parser.add_argument("--val_data", type=str, default="edu_fineweb_fixed/edufineweb_val_*.bin")
    parser.add_argument("--test_data", type=str, default="edu_fineweb_fixed/edufineweb_test_*.bin")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Limit batches for quick testing")
    parser.add_argument("--budgets", nargs='+', default=None,
                        help="Token budgets to evaluate (default: all). E.g. --budgets 50m 100m")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    device = args.device or get_device()
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # All available budgets
    all_budgets = ["50m", "100m", "200m"]
    budget_labels = {"50m": "50M tokens", "100m": "100M tokens", "200m": "200M tokens"}
    colors = {"50m": "#2196F3", "100m": "#FF9800", "200m": "#4CAF50"}
    x_vals = [1, 2, 3, 4]

    # Filter to requested budgets
    budgets_to_run = args.budgets if args.budgets else all_budgets
    for b in budgets_to_run:
        if b not in all_budgets:
            print(f"Error: unknown budget '{b}'. Choose from: {all_budgets}")
            return

    # Load val and test tokens once
    print("\nLoading validation data...")
    val_tokens = load_all_tokens(args.val_data)
    print("Loading test data...")
    test_tokens = load_all_tokens(args.test_data)

    for budget_name in budgets_to_run:
        model_ids = [1, 2, 3, 4]

        print(f"\n{'='*60}")
        print(f"TOKEN BUDGET: {budget_name}")
        print(f"{'='*60}")

        # Load all 4 models for this budget
        models = {}
        for mid in model_ids:
            ckpt_name = f"model_{budget_name}_{mid}_best.pt"
            ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)
            print(f"  Loading {ckpt_name}...")
            models[mid] = load_model(ckpt_path, device)

        budget_results = {
            "val": {1: [], 2: [], 3: [], 4: []},
            "test": {1: [], 2: [], 3: [], 4: []},
        }

        # Evaluate all combinations for ensemble sizes 1..4
        for k in range(1, 5):
            combos = list(combinations(model_ids, k))
            print(f"\n  Ensemble size {k}: {len(combos)} combination(s)")

            for combo in combos:
                combo_models = [models[i] for i in combo]
                combo_str = "+".join(str(i) for i in combo)

                # Validation
                val_loss = evaluate_loss(
                    combo_models, val_tokens, device,
                    batch_size=args.batch_size, max_batches=args.max_batches
                )
                budget_results["val"][k].append(val_loss)

                # Test
                test_loss = evaluate_loss(
                    combo_models, test_tokens, device,
                    batch_size=args.batch_size, max_batches=args.max_batches
                )
                budget_results["test"][k].append(test_loss)

                val_ppl = np.exp(val_loss)
                test_ppl = np.exp(test_loss)
                print(f"    [{combo_str}] val_loss={val_loss:.4f} val_ppl={val_ppl:.2f} | "
                      f"test_loss={test_loss:.4f} test_ppl={test_ppl:.2f}")

        # Free memory
        del models
        if device == "cuda":
            torch.cuda.empty_cache()

        # ---------------------------------------------------------------------
        # Print summary for this budget
        # ---------------------------------------------------------------------
        print(f"\n{'='*70}")
        print(f"SUMMARY — {budget_labels[budget_name]}")
        print(f"{'='*70}")

        for split in ["val", "test"]:
            print(f"\n--- {split.upper()} ---")
            print(f"  {'k':<4} {'Loss (mean±std)':<22} {'PPL (mean±std)':<22}")
            print(f"  {'-'*4} {'-'*22} {'-'*22}")

            for k in x_vals:
                losses = budget_results[split][k]
                mean_loss = np.mean(losses)
                std_loss = np.std(losses)
                ppls = [np.exp(l) for l in losses]
                mean_ppl = np.mean(ppls)
                std_ppl = np.std(ppls)

                if len(losses) > 1:
                    loss_str = f"{mean_loss:.4f} ± {std_loss:.4f}"
                    ppl_str = f"{mean_ppl:.2f} ± {std_ppl:.2f}"
                else:
                    loss_str = f"{mean_loss:.4f}"
                    ppl_str = f"{mean_ppl:.2f}"

                print(f"  {k:<4} {loss_str:<22} {ppl_str:<22}")

        # ---------------------------------------------------------------------
        # Plot for this budget
        # ---------------------------------------------------------------------
        color = colors[budget_name]
        val_data = budget_results["val"]
        test_data = budget_results["test"]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # --- Loss ---
        ax = axes[0]
        for split_data, label, ls, marker in [
            (test_data, "Test", "-", "o"),
            (val_data, "Validation", "--", "^"),
        ]:
            means = [np.mean(split_data[k]) for k in x_vals]
            stds = [np.std(split_data[k]) for k in x_vals]
            ax.errorbar(x_vals, means, yerr=stds, marker=marker, color=color,
                        linestyle=ls, capsize=5, linewidth=2, markersize=8, label=label)
        ax.set_xlabel("Number of models in ensemble", fontsize=12)
        ax.set_ylabel("Loss (cross-entropy)", fontsize=12)
        ax.set_xticks(x_vals)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # --- Perplexity ---
        ax = axes[1]
        for split_data, label, ls, marker in [
            (test_data, "Test", "-", "o"),
            (val_data, "Validation", "--", "^"),
        ]:
            ppl_per_k = {k: [np.exp(l) for l in split_data[k]] for k in x_vals}
            means = [np.mean(ppl_per_k[k]) for k in x_vals]
            stds = [np.std(ppl_per_k[k]) for k in x_vals]
            ax.errorbar(x_vals, means, yerr=stds, marker=marker, color=color,
                        linestyle=ls, capsize=5, linewidth=2, markersize=8, label=label)
        ax.set_xlabel("Number of models in ensemble", fontsize=12)
        ax.set_ylabel("Perplexity", fontsize=12)
        ax.set_xticks(x_vals)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        fig.suptitle(f"Ensemble Scaling — {budget_labels[budget_name]}",
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        plot_path = os.path.join(args.output_dir, f"ensemble_scaling_{budget_name}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {plot_path}")
        plt.close()

        # ---------------------------------------------------------------------
        # Save JSON for this budget
        # ---------------------------------------------------------------------
        json_result = {}
        for split in ["val", "test"]:
            json_result[split] = {}
            for k in x_vals:
                losses = budget_results[split][k]
                ppls = [np.exp(l) for l in losses]
                json_result[split][str(k)] = {
                    "losses": losses,
                    "perplexities": ppls,
                    "mean_loss": float(np.mean(losses)),
                    "std_loss": float(np.std(losses)),
                    "mean_ppl": float(np.mean(ppls)),
                    "std_ppl": float(np.std(ppls)),
                }

        json_path = os.path.join(args.output_dir, f"ensemble_scaling_{budget_name}.json")
        with open(json_path, 'w') as f:
            json.dump(json_result, f, indent=2)
        print(f"Results saved to: {json_path}")


if __name__ == "__main__":
    main()
