"""
Inference script for GPT model - works on MPS (Mac), CPU, and CUDA.
This is a standalone script that doesn't require flex_attention or torch.compile.
"""

import argparse
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

# Tokenizer
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("tiktoken not installed. Install with: pip install tiktoken")


# -----------------------------------------------------------------------------
# Model components (MPS/CPU compatible - no flex_attention or torch.compile)

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
        # x: (B, T, n_head, head_dim)
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

        # QK RMS norm
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Use standard scaled_dot_product_attention (works on MPS/CPU/CUDA)
        # Transpose to (B, n_head, T, head_dim) for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # scaled_dot_product_attention with causal mask
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

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

# Alias for backward compatibility with checkpoints saved as 'Config'
Config = GPTConfig


class GPT(nn.Module):
    """
    GPT model with U-Net skip connections (inference-only, no flex_attention).
    """
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
        self.lm_head.weight.data.zero_()

    def forward(self, idx):
        """Forward pass for generation (no target/loss)."""
        x = self.transformer.wte(idx)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        v1 = None

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

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50):
        """
        Generate tokens autoregressively.
        idx: (B, T) tensor of input token indices
        Returns: (B, T + max_new_tokens) tensor
        """
        for _ in range(max_new_tokens):
            # Crop to last 1024 tokens if needed
            idx_cond = idx if idx.size(1) <= 1024 else idx[:, -1024:]

            # Forward pass
            logits = self.forward(idx_cond)

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


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_model(checkpoint_path, device, dtype=torch.float32):
    """
    Load model from checkpoint trained with train.py.
    Handles DDP checkpoints and converts to inference-compatible format.
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract config from checkpoint if available
    if 'config' in checkpoint:
        cfg_dict = checkpoint['config']
        cfg = GPTConfig(
            vocab_size=cfg_dict.get('vocab_size', 50304),
            n_layer=cfg_dict.get('n_layer', 12),
            n_head=cfg_dict.get('n_head', 6),
            n_embd=cfg_dict.get('n_embd', 768),
        )
    else:
        # Use default config
        cfg = GPTConfig()

    model = GPT(cfg)

    # Handle DDP checkpoint (keys start with "module.")
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove "module." prefix if present (from DDP)
        if k.startswith('module.'):
            k = k[7:]
        # Convert to target dtype
        if v.dtype == torch.bfloat16:
            v = v.to(dtype)
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model = model.to(device).to(dtype)
    model.eval()

    print(f"Model loaded: {cfg.n_layer} layers, {cfg.n_head} heads, {cfg.n_embd} dim")
    print(f"Device: {device}, dtype: {dtype}")

    return model, cfg


def main():
    parser = argparse.ArgumentParser(description="GPT Inference (MPS/CPU/CUDA)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                        help="Text prompt for generation")
    parser.add_argument("--max_tokens", type=int, default=100,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (higher = more random)")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling (0 to disable)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda, mps, cpu). Auto-detected if not specified.")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16"],
                        help="Data type for inference")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode - keep generating from user prompts")
    args = parser.parse_args()

    # Setup device
    device = args.device if args.device else get_device()
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    # MPS doesn't fully support float16, use float32
    if device == "mps" and dtype == torch.float16:
        print("Note: MPS has limited float16 support, using float32 instead")
        dtype = torch.float32

    # Load model
    model, cfg = load_model(args.checkpoint, device, dtype)

    # Setup tokenizer
    if not TIKTOKEN_AVAILABLE:
        print("Error: tiktoken is required for tokenization")
        print("Install with: pip install tiktoken")
        return

    enc = tiktoken.get_encoding("gpt2")

    def generate_text(prompt, max_tokens, temperature, top_k):
        """Generate text from a prompt."""
        tokens = enc.encode(prompt)
        tokens = torch.tensor([tokens], dtype=torch.long, device=device)

        print(f"\nPrompt: {prompt}")
        print(f"Generating {max_tokens} tokens...")
        print("-" * 50)

        output_tokens = model.generate(
            tokens,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k if top_k > 0 else None
        )

        generated = enc.decode(output_tokens[0].tolist())
        print(generated)
        print("-" * 50)
        return generated

    if args.interactive:
        print("\nInteractive mode. Type 'quit' to exit.")
        print("Commands: /temp <value>, /topk <value>, /tokens <value>")
        temperature = args.temperature
        top_k = args.top_k
        max_tokens = args.max_tokens

        while True:
            try:
                prompt = input("\nEnter prompt: ").strip()
                if prompt.lower() == 'quit':
                    break
                if prompt.startswith('/temp '):
                    temperature = float(prompt.split()[1])
                    print(f"Temperature set to {temperature}")
                    continue
                if prompt.startswith('/topk '):
                    top_k = int(prompt.split()[1])
                    print(f"Top-k set to {top_k}")
                    continue
                if prompt.startswith('/tokens '):
                    max_tokens = int(prompt.split()[1])
                    print(f"Max tokens set to {max_tokens}")
                    continue
                if prompt:
                    generate_text(prompt, max_tokens, temperature, top_k)
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    else:
        generate_text(args.prompt, args.max_tokens, args.temperature, args.top_k)


if __name__ == "__main__":
    main()
