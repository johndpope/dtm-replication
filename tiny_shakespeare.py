"""
tiny_shakespeare.py
Downloads tiny Shakespeare + causal Transformer with positional encoding.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import requests
from pathlib import Path


def get_shakespeare_data():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    path = Path("tinyshakespeare.txt")
    if not path.exists():
        print("Downloading tiny Shakespeare...")
        r = requests.get(url)
        path.write_text(r.text)
    return path.read_text()


class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s): return [self.stoi[c] for c in s]
    def decode(self, l): return ''.join([self.itos[i] for i in l])


class TransformerBlock(eqx.Module):
    attn: eqx.nn.MultiheadAttention
    mlp: eqx.nn.MLP
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm

    def __init__(self, d_model: int, n_heads: int, key: jax.Array):
        k1, k2 = jr.split(key, 2)
        self.attn = eqx.nn.MultiheadAttention(
            num_heads=n_heads,
            query_size=d_model,
            key=k1,
        )
        self.mlp = eqx.nn.MLP(d_model, d_model, d_model * 4, depth=2, key=k2)
        self.norm1 = eqx.nn.LayerNorm(d_model)
        self.norm2 = eqx.nn.LayerNorm(d_model)

    def __call__(self, x, mask=None):
        nx = jax.vmap(self.norm1)(x)
        x = x + self.attn(nx, nx, nx, mask=mask)
        x = x + jax.vmap(self.mlp)(jax.vmap(self.norm2)(x))
        return x


class TinyTransformer(eqx.Module):
    embed: eqx.nn.Embedding       # token embedding (kept for FTA compatibility)
    pos_embed: eqx.nn.Embedding   # positional embedding
    blocks: list[TransformerBlock]
    head: eqx.nn.Linear
    final_norm: eqx.nn.LayerNorm
    max_seq_len: int = eqx.field(static=True)

    def __init__(self, vocab_size: int, d_model: int = 256, n_heads: int = 8,
                 n_layers: int = 6, max_seq_len: int = 512, key=jr.PRNGKey(0)):
        # keys: embed, pos_embed, n_layers blocks, head  (n_layers + 3 total)
        keys = jr.split(key, n_layers + 3)
        self.embed = eqx.nn.Embedding(vocab_size, d_model, key=keys[0])
        self.pos_embed = eqx.nn.Embedding(max_seq_len, d_model, key=keys[1])
        self.blocks = [TransformerBlock(d_model, n_heads, key=keys[i + 2]) for i in range(n_layers)]
        self.final_norm = eqx.nn.LayerNorm(d_model)
        self.head = eqx.nn.Linear(d_model, vocab_size, key=keys[-1])
        self.max_seq_len = max_seq_len

    def __call__(self, tokens):
        """tokens: (seq_len,) → logits: (seq_len, vocab)"""
        seq_len = tokens.shape[0]
        x = jax.vmap(self.embed)(tokens) + jax.vmap(self.pos_embed)(jnp.arange(seq_len))
        # causal mask: position i may only attend to positions 0..i
        mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        for block in self.blocks:
            x = block(x, mask=mask)
        x = jax.vmap(self.final_norm)(x)
        return jax.vmap(self.head)(x)

    def get_logits(self, tokens):
        return self(tokens)


def compute_ppl(logits, targets):
    """logits: (seq_len, vocab), targets: (seq_len,) → scalar PPL"""
    loss = jax.nn.logsumexp(logits, axis=-1) - logits[jnp.arange(logits.shape[0]), targets]
    return jnp.exp(jnp.mean(loss))
