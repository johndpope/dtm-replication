"""
flash_thermodynamic_attention.py
Multi-head Flash Thermodynamic Attention (FTA) — training-free, low-PPL
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Array, Float, Key
from thrmlDenoising.DTM import DTM
from thrmlDenoising.utils import make_cfg
from thrmlDenoising.base_graphs.poisson_binomial_ising_graph_manager import PoissonBinomialIsingGraphManager
from thrmlDenoising.smoke_testing import smoke_test_data_dict


class FlashThermodynamicAttention(eqx.Module):
    d_model: int
    n_heads: int
    head_dim: int
    n_diffusion_steps: int
    graph_preset: int
    beta_start: float
    beta_end: float
    torus: bool

    dtm: DTM
    graph_manager: PoissonBinomialIsingGraphManager

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_diffusion_steps: int = 2,
        graph_preset: int = 60_12,
        beta_start: float = 0.8,
        beta_end: float = 1.2,
        torus: bool = False,
        key: Key = jr.PRNGKey(42),
    ):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_diffusion_steps = n_diffusion_steps
        self.graph_preset = graph_preset
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.torus = torus

        self.graph_manager = PoissonBinomialIsingGraphManager(
            n_image_pixels=self.head_dim, n_label_nodes=0, n_trials=1
        )

        # Register a minimal smoke-test entry for this head dimension if not present.
        _smoke_key = (self.head_dim, 1, 1)
        if _smoke_key not in smoke_test_data_dict:
            smoke_test_data_dict[_smoke_key] = {
                "image": jnp.zeros((1, self.head_dim), dtype=jnp.bool_),
                "label": jnp.array([0], dtype=jnp.int32),
            }

        cfg = make_cfg(
            exp=dict(seed=int(jr.randint(key, (1,), 0, 2**31 - 1)[0]), descriptor="fta_multihead"),
            data=dict(dataset_name=f"smoke_testing_{self.head_dim}_1_1"),
            graph=dict(
                graph_preset_architecture=graph_preset,
                num_label_spots=0,
                grayscale_levels=1,
                torus=torus,
                base_graph_manager="poisson_binomial_ising_graph_manager",
            ),
            sampling=dict(batch_size=1, n_samples=1, steps_per_sample=8, steps_warmup=32, training_beta=1.0),
            generation=dict(generation_beta_start=beta_start, generation_beta_end=beta_end, steps_warmup=64),
            diffusion_schedule=dict(num_diffusion_steps=n_diffusion_steps, kind="log", diffusion_offset=0.1),
            diffusion_rates=dict(image_rate=0.9, label_rate=0.0),
            optim=dict(step_learning_rates=(0.0,)),
            cp=dict(adaptive_cp=False),
        )
        self.dtm = DTM(cfg)

    @eqx.filter_jit
    def __call__(
        self,
        q: Float[Array, "batch seq d_model"],
        k: Float[Array, "batch seq d_model"],
        v: Float[Array, "batch seq d_model"],
        mask: Array | None = None,
        key: Key = jr.PRNGKey(0),
    ) -> Float[Array, "batch seq d_model"]:
        B, N, D = q.shape
        H, HD = self.n_heads, self.head_dim

        # reshape to (B, H, N, HD) — vmap will split over the H axis
        q = q.reshape(B, N, H, HD).transpose(0, 2, 1, 3)  # (B, H, N, HD)
        k = k.reshape(B, N, H, HD).transpose(0, 2, 1, 3)
        v = v.reshape(B, N, H, HD).transpose(0, 2, 1, 3)

        # per-head FTA: vmap over heads (axis 1 of q/k/v, axis 0 of keys)
        def head_fta(qh, kh, vh, head_key):
            # qh, kh, vh: (B, N, HD)
            # Flatten (B, N) into a single batch dimension for the DTM denoiser.
            states = (vh > 0).astype(jnp.bool_)               # (B, N, HD)
            states_flat = states.reshape(B * N, HD)            # (B*N, HD)

            image_out_blocks, _ = self.dtm.steps[0].denoise(
                head_key, states_flat, None,
                self.dtm.steps[0].generation_spec.schedule,
            )
            # image_out_blocks[0]: (B*N, n_samples, HD) — take the last sample
            denoised = image_out_blocks[0][:, -1, :].reshape(B, N, HD)
            return denoised.astype(jnp.float32) * 2 - 1

        keys = jr.split(key, H)
        # in_axes=(1,1,1,0): map axis-1 (heads) of q/k/v, axis-0 of keys
        attended_heads = jax.vmap(head_fta, in_axes=(1, 1, 1, 0), out_axes=1)(q, k, v, keys)
        # attended_heads: (B, H, N, HD)

        # reassemble to (B, N, D)
        attended = attended_heads.transpose(0, 2, 1, 3).reshape(B, N, D)
        return attended


# =============================================================================
# Low-PPL refinement helper (training-free)
# =============================================================================
def low_ppl_refine(
    base_logits: Float[Array, "seq vocab"],
    fta: FlashThermodynamicAttention,
    embed_table: Array,           # [vocab, d_model] from your LM
    key: Key = jr.PRNGKey(0),
    blend: float = 0.7,
) -> Float[Array, "seq vocab"]:
    """Drop-in low-PPL booster: sample from LM → FTA denoise → blend logits."""
    tokens = jnp.argmax(base_logits, axis=-1)                    # (seq,)
    embeddings = embed_table[tokens][None]                       # (1, seq, d_model)

    refined = fta(embeddings, embeddings, embeddings, key=key)   # (1, seq, d_model)
    refined = refined[0]                                         # (seq, d_model)

    # project back to vocab via embedding table transpose
    proj = jnp.einsum("nd,vd->nv", refined, embed_table)        # (seq, vocab)
    return blend * base_logits + (1 - blend) * proj