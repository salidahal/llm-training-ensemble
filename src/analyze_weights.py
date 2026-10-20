"""
Weight Analysis Script - Compare model parameters across training budgets.

Analyzes:
1. Weight statistics (mean, std, min, max, L2 norm) per layer
2. Weight distributions (histograms)
3. Singular value spectra (effective rank analysis)
4. Pairwise positional weight comparison (--compare_pairwise)

Usage:
    # Single checkpoint
    python analyze_weights.py --checkpoints model_1000.pt

    # Multiple checkpoints (compare training budgets)
    python analyze_weights.py --checkpoints model_1000.pt model_5000.pt model_10000.pt

    # Custom output directory
    python analyze_weights.py --checkpoints *.pt --output analysis_results

    # Pairwise comparison (exactly 2 checkpoints required)
    python analyze_weights.py --checkpoints model_a.pt model_b.pt --compare_pairwise

    # Prefix output filenames for organization
    python analyze_weights.py --checkpoints model_a.pt model_b.pt --compare_pairwise --prefix pairwise_200m
"""

import argparse
import os
from glob import glob
from collections import defaultdict
from dataclasses import dataclass

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# Config class needed for unpickling checkpoints
@dataclass
class GPTConfig:
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 768

# Alias for backward compatibility
Config = GPTConfig


def prefixed(filename, prefix):
    """Prepend prefix to filename if provided."""
    if prefix:
        return f"{prefix}_{filename}"
    return filename


def load_state_dict(checkpoint_path):
    """Load state dict from checkpoint, handling DDP format."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Remove "module." prefix if present (from DDP)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v.float()  # Convert to float32 for analysis

    # Extract step/epoch info if available
    step = checkpoint.get('step', None)
    epoch = checkpoint.get('epoch', None)

    return new_state_dict, step, epoch


def get_layer_name(param_name):
    """Extract readable layer name from parameter name."""
    # e.g., "transformer.h.0.attn.c_q.weight" -> "h.0.attn.c_q"
    parts = param_name.replace('.weight', '').replace('.bias', '')
    return parts


def compute_weight_statistics(state_dict):
    """Compute statistics for each parameter tensor."""
    stats = {}
    for name, tensor in state_dict.items():
        if tensor.ndim == 0:  # Skip scalars
            continue

        flat = tensor.flatten().float()
        stats[name] = {
            'shape': tuple(tensor.shape),
            'numel': tensor.numel(),
            'mean': flat.mean().item(),
            'std': flat.std().item(),
            'min': flat.min().item(),
            'max': flat.max().item(),
            'l2_norm': torch.norm(flat, p=2).item(),
            'l1_norm': torch.norm(flat, p=1).item(),
            'sparsity': (flat.abs() < 1e-6).float().mean().item(),
        }
    return stats


def compute_singular_values(tensor):
    """Compute singular values for 2D weight matrices."""
    if tensor.ndim != 2:
        return None
    try:
        # Use SVD to get singular values
        U, S, V = torch.svd(tensor.float())
        return S.numpy()
    except Exception:
        return None


def plot_weight_statistics(all_stats, checkpoint_names, output_dir, prefix=""):
    """Plot weight statistics comparison across checkpoints."""

    # Get all parameter names (use first checkpoint as reference)
    param_names = list(all_stats[checkpoint_names[0]].keys())

    # Filter to only weight matrices (not biases, lambdas, etc.)
    weight_params = [p for p in param_names if 'weight' in p or 'skip_weights' in p]

    # Create figure for statistics comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    metrics = ['mean', 'std', 'l2_norm', 'sparsity']
    titles = ['Mean', 'Standard Deviation', 'L2 Norm', 'Sparsity (|w| < 1e-6)']

    colors = cm.viridis(np.linspace(0, 0.8, len(checkpoint_names)))

    for ax, metric, title in zip(axes.flat, metrics, titles):
        x = np.arange(len(weight_params))
        width = 0.8 / len(checkpoint_names)

        for i, ckpt_name in enumerate(checkpoint_names):
            values = [all_stats[ckpt_name][p][metric] for p in weight_params]
            offset = (i - len(checkpoint_names)/2 + 0.5) * width
            ax.bar(x + offset, values, width, label=ckpt_name, color=colors[i])

        ax.set_ylabel(title)
        ax.set_xticks(x)
        ax.set_xticklabels([get_layer_name(p) for p in weight_params], rotation=45, ha='right', fontsize=7)
        ax.legend(fontsize=8)
        ax.set_title(f'{title} by Layer')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fname = prefixed('weight_statistics.png', prefix)
    plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")

    # Create summary table
    print("\n" + "="*80)
    print("WEIGHT STATISTICS SUMMARY")
    print("="*80)

    for ckpt_name in checkpoint_names:
        print(f"\n{ckpt_name}:")
        print("-" * 60)
        print(f"{'Layer':<40} {'Mean':>10} {'Std':>10} {'L2 Norm':>10}")
        print("-" * 60)
        for p in weight_params[:15]:  # Show first 15 layers
            s = all_stats[ckpt_name][p]
            layer = get_layer_name(p)[:38]
            print(f"{layer:<40} {s['mean']:>10.4f} {s['std']:>10.4f} {s['l2_norm']:>10.2f}")


def plot_weight_distributions(all_state_dicts, checkpoint_names, output_dir, prefix=""):
    """Plot weight distribution histograms."""

    # Get all parameter names
    param_names = list(all_state_dicts[checkpoint_names[0]].keys())
    weight_params = [p for p in param_names if 'weight' in p]

    # Select representative layers for visualization
    # (embedding, early attention, middle MLP, late attention, output)
    selected_layers = []
    for pattern in ['wte.weight', 'h.0.attn', 'h.0.mlp', 'h.5.attn', 'h.5.mlp', 'h.11.attn', 'lm_head']:
        for p in weight_params:
            if pattern in p:
                selected_layers.append(p)
                break

    if not selected_layers:
        selected_layers = weight_params[:6]

    n_layers = len(selected_layers)
    n_ckpts = len(checkpoint_names)

    fig, axes = plt.subplots(n_layers, n_ckpts, figsize=(4*n_ckpts, 3*n_layers))
    if n_layers == 1:
        axes = axes.reshape(1, -1)
    if n_ckpts == 1:
        axes = axes.reshape(-1, 1)

    colors = cm.viridis(np.linspace(0, 0.8, n_ckpts))

    for i, param_name in enumerate(selected_layers):
        for j, ckpt_name in enumerate(checkpoint_names):
            ax = axes[i, j]
            tensor = all_state_dicts[ckpt_name][param_name]
            values = tensor.flatten().numpy()

            # Clip extreme values for better visualization
            vmin, vmax = np.percentile(values, [1, 99])
            values_clipped = np.clip(values, vmin, vmax)

            ax.hist(values_clipped, bins=100, color=colors[j], alpha=0.7, density=True)
            ax.set_title(f"{get_layer_name(param_name)}\n{ckpt_name}", fontsize=9)
            ax.set_xlabel('Weight Value', fontsize=8)
            ax.set_ylabel('Density', fontsize=8)
            ax.tick_params(labelsize=7)

            # Add stats annotation
            mean, std = values.mean(), values.std()
            ax.axvline(mean, color='red', linestyle='--', linewidth=1, alpha=0.7)
            ax.text(0.95, 0.95, f'μ={mean:.3f}\nσ={std:.3f}',
                   transform=ax.transAxes, fontsize=7, va='top', ha='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    fname = prefixed('weight_distributions.png', prefix)
    plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")


def plot_singular_values(all_state_dicts, checkpoint_names, output_dir, prefix=""):
    """Plot singular value spectra for weight matrices."""

    # Get 2D weight matrices
    param_names = list(all_state_dicts[checkpoint_names[0]].keys())
    weight_2d = [p for p in param_names
                 if 'weight' in p and all_state_dicts[checkpoint_names[0]][p].ndim == 2]

    # Select representative layers
    selected_layers = []
    for pattern in ['wte.weight', 'h.0.attn.c_q', 'h.0.mlp.c_fc', 'h.5.attn.c_q',
                    'h.5.mlp.c_fc', 'h.11.attn.c_q', 'lm_head']:
        for p in weight_2d:
            if pattern in p:
                selected_layers.append(p)
                break

    if not selected_layers:
        selected_layers = weight_2d[:6]

    n_layers = len(selected_layers)
    colors = cm.viridis(np.linspace(0, 0.8, len(checkpoint_names)))

    # Plot 1: Singular value spectra
    fig, axes = plt.subplots(2, (n_layers + 1) // 2, figsize=(5 * ((n_layers + 1) // 2), 8))
    axes = axes.flatten()

    for i, param_name in enumerate(selected_layers):
        ax = axes[i]

        for j, ckpt_name in enumerate(checkpoint_names):
            tensor = all_state_dicts[ckpt_name][param_name]
            sv = compute_singular_values(tensor)
            if sv is not None:
                # Normalize by largest singular value
                sv_normalized = sv / sv[0]
                ax.semilogy(sv_normalized, label=ckpt_name, color=colors[j], linewidth=1.5)

        ax.set_title(get_layer_name(param_name), fontsize=10)
        ax.set_xlabel('Index', fontsize=9)
        ax.set_ylabel('Normalized Singular Value', fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([1e-4, 1.5])

    # Hide unused subplots
    for i in range(n_layers, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    fname = prefixed('singular_values.png', prefix)
    plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")

    # Plot 2: Effective rank comparison
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(weight_2d))
    width = 0.8 / len(checkpoint_names)

    for i, ckpt_name in enumerate(checkpoint_names):
        effective_ranks = []
        for param_name in weight_2d:
            tensor = all_state_dicts[ckpt_name][param_name]
            sv = compute_singular_values(tensor)
            if sv is not None and len(sv) > 0:
                # Effective rank: sum(sv)^2 / sum(sv^2)
                sv_normalized = sv / sv.sum()
                entropy = -np.sum(sv_normalized * np.log(sv_normalized + 1e-10))
                effective_rank = np.exp(entropy)
                effective_ranks.append(effective_rank)
            else:
                effective_ranks.append(0)

        offset = (i - len(checkpoint_names)/2 + 0.5) * width
        ax.bar(x + offset, effective_ranks, width, label=ckpt_name, color=colors[i])

    ax.set_ylabel('Effective Rank (exp entropy)')
    ax.set_xticks(x)
    ax.set_xticklabels([get_layer_name(p) for p in weight_2d], rotation=45, ha='right', fontsize=7)
    ax.legend()
    ax.set_title('Effective Rank by Layer (higher = more capacity used)')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fname = prefixed('effective_rank.png', prefix)
    plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")


def plot_layer_evolution(all_state_dicts, checkpoint_names, output_dir, prefix=""):
    """Plot how specific layers evolve across training."""

    if len(checkpoint_names) < 2:
        return

    param_names = list(all_state_dicts[checkpoint_names[0]].keys())
    weight_params = [p for p in param_names if 'weight' in p]

    # Compute L2 distance from initial checkpoint
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: L2 distance from first checkpoint
    ax = axes[0]
    first_ckpt = checkpoint_names[0]

    for param_name in weight_params[:20]:  # Limit to avoid clutter
        distances = []
        for ckpt_name in checkpoint_names:
            diff = all_state_dicts[ckpt_name][param_name] - all_state_dicts[first_ckpt][param_name]
            dist = torch.norm(diff).item()
            distances.append(dist)

        ax.plot(range(len(checkpoint_names)), distances, 'o-',
                label=get_layer_name(param_name), markersize=4, linewidth=1)

    ax.set_xlabel('Checkpoint')
    ax.set_ylabel(f'L2 Distance from {first_ckpt}')
    ax.set_xticks(range(len(checkpoint_names)))
    ax.set_xticklabels(checkpoint_names, rotation=45, ha='right')
    ax.set_title('Parameter Change from Initial Checkpoint')
    ax.legend(fontsize=6, ncol=2, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Plot 2: Cosine similarity with first checkpoint
    ax = axes[1]

    for param_name in weight_params[:20]:
        similarities = []
        first_flat = all_state_dicts[first_ckpt][param_name].flatten()

        for ckpt_name in checkpoint_names:
            current_flat = all_state_dicts[ckpt_name][param_name].flatten()
            cos_sim = torch.nn.functional.cosine_similarity(
                first_flat.unsqueeze(0), current_flat.unsqueeze(0)
            ).item()
            similarities.append(cos_sim)

        ax.plot(range(len(checkpoint_names)), similarities, 'o-',
                label=get_layer_name(param_name), markersize=4, linewidth=1)

    ax.set_xlabel('Checkpoint')
    ax.set_ylabel(f'Cosine Similarity with {first_ckpt}')
    ax.set_xticks(range(len(checkpoint_names)))
    ax.set_xticklabels(checkpoint_names, rotation=45, ha='right')
    ax.set_title('Parameter Direction Change from Initial')
    ax.legend(fontsize=6, ncol=2, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    fname = prefixed('layer_evolution.png', prefix)
    plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")


def plot_attention_patterns(all_state_dicts, checkpoint_names, output_dir, prefix=""):
    """Visualize attention weight matrices as heatmaps."""

    # Find attention projection weights
    param_names = list(all_state_dicts[checkpoint_names[0]].keys())
    attn_weights = [p for p in param_names if 'attn' in p and 'weight' in p and 'c_q' in p]

    if not attn_weights:
        return

    # Select a few layers
    selected = attn_weights[:3] if len(attn_weights) >= 3 else attn_weights

    n_ckpts = len(checkpoint_names)
    n_layers = len(selected)

    fig, axes = plt.subplots(n_layers, n_ckpts, figsize=(4*n_ckpts, 4*n_layers))
    if n_layers == 1:
        axes = axes.reshape(1, -1)
    if n_ckpts == 1:
        axes = axes.reshape(-1, 1)

    for i, param_name in enumerate(selected):
        for j, ckpt_name in enumerate(checkpoint_names):
            ax = axes[i, j]
            tensor = all_state_dicts[ckpt_name][param_name].numpy()

            # Downsample if too large
            if tensor.shape[0] > 128 or tensor.shape[1] > 128:
                step0 = max(1, tensor.shape[0] // 128)
                step1 = max(1, tensor.shape[1] // 128)
                tensor = tensor[::step0, ::step1]

            vmax = np.abs(tensor).max()
            im = ax.imshow(tensor, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
            ax.set_title(f"{get_layer_name(param_name)}\n{ckpt_name}", fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fname = prefixed('attention_heatmaps.png', prefix)
    plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")


# -----------------------------------------------------------------------------
# Pairwise Comparison Functions
# -----------------------------------------------------------------------------

def compute_pairwise_statistics(state_dict_a, state_dict_b, name_a, name_b):
    """Compute pairwise statistics between two models."""
    stats = {}

    for param_name in state_dict_a.keys():
        if param_name not in state_dict_b:
            continue

        tensor_a = state_dict_a[param_name]
        tensor_b = state_dict_b[param_name]

        if tensor_a.shape != tensor_b.shape:
            continue

        if tensor_a.ndim == 0:  # Skip scalars
            continue

        flat_a = tensor_a.flatten().float()
        flat_b = tensor_b.flatten().float()
        diff = flat_a - flat_b

        # Correlation coefficient
        mean_a, mean_b = flat_a.mean(), flat_b.mean()
        std_a, std_b = flat_a.std(), flat_b.std()
        if std_a > 0 and std_b > 0:
            correlation = ((flat_a - mean_a) * (flat_b - mean_b)).mean() / (std_a * std_b)
            correlation = correlation.item()
        else:
            correlation = 0.0

        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            flat_a.unsqueeze(0), flat_b.unsqueeze(0)
        ).item()

        stats[param_name] = {
            'shape': tuple(tensor_a.shape),
            'correlation': correlation,
            'cosine_similarity': cos_sim,
            'l2_distance': torch.norm(diff, p=2).item(),
            'mean_abs_diff': diff.abs().mean().item(),
            'max_abs_diff': diff.abs().max().item(),
            'diff_std': diff.std().item(),
        }

    return stats


def plot_pairwise_scatter(state_dict_a, state_dict_b, name_a, name_b, output_dir, prefix=""):
    """Create scatter plots of weight_A[i,j] vs weight_B[i,j] for each layer."""

    param_names = list(state_dict_a.keys())
    weight_params = [p for p in param_names if 'weight' in p and p in state_dict_b]

    # Select representative layers
    selected_layers = []
    for pattern in ['wte.weight', 'h.0.attn.c_q', 'h.0.mlp.c_fc',
                    'h.5.attn.c_q', 'h.5.mlp.c_fc', 'h.11.attn.c_q', 'lm_head']:
        for p in weight_params:
            if pattern in p:
                selected_layers.append(p)
                break

    if not selected_layers:
        selected_layers = weight_params[:6]

    n_layers = len(selected_layers)
    n_cols = 3
    n_rows = (n_layers + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    axes = axes.flatten() if n_layers > 1 else [axes]

    for i, param_name in enumerate(selected_layers):
        ax = axes[i]

        weights_a = state_dict_a[param_name].flatten().numpy()
        weights_b = state_dict_b[param_name].flatten().numpy()

        # Subsample for visualization if too many points
        n_points = len(weights_a)
        if n_points > 50000:
            idx = np.random.choice(n_points, 50000, replace=False)
            weights_a_plot = weights_a[idx]
            weights_b_plot = weights_b[idx]
        else:
            weights_a_plot = weights_a
            weights_b_plot = weights_b

        # Scatter plot
        ax.scatter(weights_a_plot, weights_b_plot, alpha=0.1, s=1, c='blue')

        # Add diagonal line (perfect correlation)
        lims = [
            min(weights_a.min(), weights_b.min()),
            max(weights_a.max(), weights_b.max())
        ]
        ax.plot(lims, lims, 'r--', linewidth=1, label='y=x')

        # Compute correlation
        correlation = np.corrcoef(weights_a, weights_b)[0, 1]

        ax.set_xlabel(f'{name_a}', fontsize=9)
        ax.set_ylabel(f'{name_b}', fontsize=9)
        ax.set_title(f'{get_layer_name(param_name)}\nr = {correlation:.4f}', fontsize=10)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_layers, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f'Positional Weight Comparison: {name_a} vs {name_b}', fontsize=12, y=1.02)
    plt.tight_layout()
    fname = prefixed('pairwise_scatter.png', prefix)
    plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")


def plot_pairwise_difference_distributions(state_dict_a, state_dict_b, name_a, name_b, output_dir, prefix=""):
    """Plot distributions of positional differences (w_A - w_B) for each layer."""

    param_names = list(state_dict_a.keys())
    weight_params = [p for p in param_names if 'weight' in p and p in state_dict_b]

    # Select representative layers
    selected_layers = []
    for pattern in ['wte.weight', 'h.0.attn.c_q', 'h.0.mlp.c_fc',
                    'h.5.attn.c_q', 'h.5.mlp.c_fc', 'h.11.attn.c_q', 'lm_head']:
        for p in weight_params:
            if pattern in p:
                selected_layers.append(p)
                break

    if not selected_layers:
        selected_layers = weight_params[:6]

    n_layers = len(selected_layers)
    n_cols = 3
    n_rows = (n_layers + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten() if n_layers > 1 else [axes]

    for i, param_name in enumerate(selected_layers):
        ax = axes[i]

        diff = (state_dict_a[param_name] - state_dict_b[param_name]).flatten().numpy()

        # Clip extreme values for visualization
        vmin, vmax = np.percentile(diff, [0.5, 99.5])
        diff_clipped = np.clip(diff, vmin, vmax)

        ax.hist(diff_clipped, bins=100, color='steelblue', alpha=0.7, density=True)
        ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='zero')
        ax.axvline(diff.mean(), color='orange', linestyle='-', linewidth=1.5, label=f'mean={diff.mean():.4f}')

        ax.set_xlabel('Weight Difference (A - B)', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.set_title(f'{get_layer_name(param_name)}\nstd={diff.std():.4f}', fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_layers, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f'Positional Difference Distribution: {name_a} - {name_b}', fontsize=12, y=1.02)
    plt.tight_layout()
    fname = prefixed('pairwise_diff_distributions.png', prefix)
    plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")


def plot_pairwise_difference_heatmaps(state_dict_a, state_dict_b, name_a, name_b, output_dir, prefix=""):
    """Create heatmaps of (weight_A - weight_B) for 2D weight matrices."""

    param_names = list(state_dict_a.keys())
    weight_2d = [p for p in param_names
                 if 'weight' in p and p in state_dict_b
                 and state_dict_a[p].ndim == 2]

    # Select representative layers
    selected_layers = []
    for pattern in ['wte.weight', 'h.0.attn.c_q', 'h.0.mlp.c_fc',
                    'h.5.attn.c_q', 'h.11.attn.c_q', 'lm_head']:
        for p in weight_2d:
            if pattern in p:
                selected_layers.append(p)
                break

    if not selected_layers:
        selected_layers = weight_2d[:6]

    n_layers = len(selected_layers)
    n_cols = 3
    n_rows = (n_layers + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    axes = axes.flatten() if n_layers > 1 else [axes]

    for i, param_name in enumerate(selected_layers):
        ax = axes[i]

        diff = (state_dict_a[param_name] - state_dict_b[param_name]).numpy()

        # Downsample if too large
        if diff.shape[0] > 128 or diff.shape[1] > 128:
            step0 = max(1, diff.shape[0] // 128)
            step1 = max(1, diff.shape[1] // 128)
            diff = diff[::step0, ::step1]

        vmax = np.abs(diff).max()
        im = ax.imshow(diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
        ax.set_title(f'{get_layer_name(param_name)}\nmax|diff|={vmax:.4f}', fontsize=10)
        ax.set_xlabel('Column', fontsize=9)
        ax.set_ylabel('Row', fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused subplots
    for i in range(n_layers, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f'Weight Difference Heatmaps: {name_a} - {name_b}', fontsize=12, y=1.02)
    plt.tight_layout()
    fname = prefixed('pairwise_diff_heatmaps.png', prefix)
    plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")


def plot_pairwise_summary(pairwise_stats, name_a, name_b, output_dir, prefix=""):
    """Create summary bar plots of pairwise statistics across all layers."""

    param_names = list(pairwise_stats.keys())
    weight_params = [p for p in param_names if 'weight' in p]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    metrics = ['correlation', 'cosine_similarity', 'l2_distance', 'mean_abs_diff']
    titles = ['Correlation Coefficient', 'Cosine Similarity', 'L2 Distance', 'Mean Absolute Difference']
    colors = ['steelblue', 'seagreen', 'coral', 'mediumpurple']

    for ax, metric, title, color in zip(axes.flat, metrics, titles, colors):
        values = [pairwise_stats[p][metric] for p in weight_params]
        x = np.arange(len(weight_params))

        ax.bar(x, values, color=color, alpha=0.7)
        ax.set_ylabel(title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([get_layer_name(p) for p in weight_params],
                          rotation=45, ha='right', fontsize=7)
        ax.set_title(f'{title} by Layer', fontsize=11)
        ax.grid(axis='y', alpha=0.3)

        # Add mean line
        mean_val = np.mean(values)
        ax.axhline(mean_val, color='red', linestyle='--', linewidth=1,
                   label=f'mean={mean_val:.4f}')
        ax.legend(fontsize=8)

    plt.suptitle(f'Pairwise Comparison Summary: {name_a} vs {name_b}', fontsize=12, y=1.02)
    plt.tight_layout()
    fname = prefixed('pairwise_summary.png', prefix)
    plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")

    # Print summary table
    print("\n" + "="*80)
    print(f"PAIRWISE COMPARISON: {name_a} vs {name_b}")
    print("="*80)
    print(f"\n{'Layer':<40} {'Corr':>8} {'Cosine':>8} {'L2 Dist':>10} {'Mean|Δ|':>10}")
    print("-" * 80)
    for p in weight_params:
        s = pairwise_stats[p]
        layer = get_layer_name(p)[:38]
        print(f"{layer:<40} {s['correlation']:>8.4f} {s['cosine_similarity']:>8.4f} "
              f"{s['l2_distance']:>10.4f} {s['mean_abs_diff']:>10.6f}")

    # Overall summary
    print("-" * 80)
    avg_corr = np.mean([pairwise_stats[p]['correlation'] for p in weight_params])
    avg_cos = np.mean([pairwise_stats[p]['cosine_similarity'] for p in weight_params])
    total_l2 = np.sqrt(sum([pairwise_stats[p]['l2_distance']**2 for p in weight_params]))
    print(f"{'AVERAGE/TOTAL':<40} {avg_corr:>8.4f} {avg_cos:>8.4f} {total_l2:>10.4f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze model weights across training checkpoints")
    parser.add_argument("--checkpoints", nargs='+', required=True,
                        help="Paths to checkpoint files (supports glob patterns)")
    parser.add_argument("--output", type=str, default="weight_analysis",
                        help="Output directory for plots")
    parser.add_argument("--labels", nargs='+', default=None,
                        help="Custom labels for checkpoints (e.g., '1k steps' '5k steps')")
    parser.add_argument("--compare_pairwise", action="store_true",
                        help="Pairwise positional comparison (requires exactly 2 checkpoints)")
    parser.add_argument("--prefix", type=str, default="",
                        help="Prefix for output filenames (e.g., 'pairwise_200m' -> 'pairwise_200m_scatter.png')")
    args = parser.parse_args()

    # Expand glob patterns
    checkpoint_paths = []
    for pattern in args.checkpoints:
        expanded = glob(pattern)
        if expanded:
            checkpoint_paths.extend(expanded)
        else:
            checkpoint_paths.append(pattern)

    checkpoint_paths = sorted(set(checkpoint_paths))

    if not checkpoint_paths:
        print("Error: No checkpoint files found")
        return

    # Validate pairwise mode
    if args.compare_pairwise and len(checkpoint_paths) != 2:
        print(f"Error: --compare_pairwise requires exactly 2 checkpoints, got {len(checkpoint_paths)}")
        return

    print(f"Found {len(checkpoint_paths)} checkpoints:")
    for p in checkpoint_paths:
        print(f"  - {p}")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load all checkpoints
    all_state_dicts = {}
    all_stats = {}
    checkpoint_names = []

    for i, path in enumerate(checkpoint_paths):
        if args.labels and i < len(args.labels):
            name = args.labels[i]
        else:
            # Extract meaningful name from path
            basename = os.path.basename(path)
            name = basename.replace('.pt', '').replace('checkpoint_', '')

        print(f"\nLoading {path}...")
        state_dict, step, epoch = load_state_dict(path)

        if step is not None:
            name = f"{name} (step {step})"

        checkpoint_names.append(name)
        all_state_dicts[name] = state_dict
        all_stats[name] = compute_weight_statistics(state_dict)

    print(f"\n{'='*60}")
    print("Generating analysis plots...")
    print('='*60)

    # Pairwise comparison mode
    if args.compare_pairwise:
        name_a, name_b = checkpoint_names[0], checkpoint_names[1]
        state_dict_a = all_state_dicts[name_a]
        state_dict_b = all_state_dicts[name_b]

        print(f"\nRunning pairwise comparison: {name_a} vs {name_b}")

        # Compute pairwise statistics
        pairwise_stats = compute_pairwise_statistics(state_dict_a, state_dict_b, name_a, name_b)

        # Generate pairwise plots
        plot_pairwise_scatter(state_dict_a, state_dict_b, name_a, name_b, args.output, args.prefix)
        plot_pairwise_difference_distributions(state_dict_a, state_dict_b, name_a, name_b, args.output, args.prefix)
        plot_pairwise_difference_heatmaps(state_dict_a, state_dict_b, name_a, name_b, args.output, args.prefix)
        plot_pairwise_summary(pairwise_stats, name_a, name_b, args.output, args.prefix)

    else:
        # Standard analysis mode
        plot_weight_statistics(all_stats, checkpoint_names, args.output, args.prefix)
        plot_weight_distributions(all_state_dicts, checkpoint_names, args.output, args.prefix)
        plot_singular_values(all_state_dicts, checkpoint_names, args.output, args.prefix)

        if len(checkpoint_names) > 1:
            plot_layer_evolution(all_state_dicts, checkpoint_names, args.output, args.prefix)

        plot_attention_patterns(all_state_dicts, checkpoint_names, args.output, args.prefix)

    print(f"\n{'='*60}")
    print(f"Analysis complete! Results saved to: {args.output}/")
    print('='*60)


if __name__ == "__main__":
    main()
