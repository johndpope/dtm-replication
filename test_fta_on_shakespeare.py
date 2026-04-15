"""
test_fta_on_shakespeare.py
Trains a causal Transformer on Shakespeare to PPL < 10, then applies multi-head FTA.
"""

import time
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from tiny_shakespeare import get_shakespeare_data, CharTokenizer, TinyTransformer, compute_ppl
from flash_thermodynamic_attention import FlashThermodynamicAttention, low_ppl_refine

# ── Hyperparameters ────────────────────────────────────────────────────────────
D_MODEL    = 256
N_HEADS    = 8
N_LAYERS   = 6
SEQ_LEN    = 256
BATCH_SIZE = 4
N_STEPS    = 5000
LR_MAX     = 3e-4
LR_MIN     = 1e-5
WARMUP     = 200
LOG_EVERY  = 250
TARGET_PPL = 10.0
# ──────────────────────────────────────────────────────────────────────────────


def get_batch(key, data, batch_size, seq_len):
    starts = jr.randint(key, (batch_size,), 0, len(data) - seq_len - 1)
    tokens  = jnp.stack([data[int(s):int(s) + seq_len]     for s in starts])
    targets = jnp.stack([data[int(s) + 1:int(s) + seq_len + 1] for s in starts])
    return tokens, targets


if __name__ == "__main__":
    key = jr.PRNGKey(42)

    # ── Data ──────────────────────────────────────────────────────────────────
    text = get_shakespeare_data()
    tokenizer = CharTokenizer(text)
    print(f"Vocab size: {tokenizer.vocab_size} | Text length: {len(text)}")

    data = jnp.array(tokenizer.encode(text))
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    # fixed validation window for consistent logging
    val_tokens  = val_data[:SEQ_LEN]
    val_targets = val_data[1:SEQ_LEN + 1]

    # ── Model ─────────────────────────────────────────────────────────────────
    key, model_key = jr.split(key)
    model = TinyTransformer(
        tokenizer.vocab_size,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        key=model_key,
    )
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))
    print(f"Model: {n_params/1e6:.2f}M params | "
          f"d_model={D_MODEL} n_heads={N_HEADS} n_layers={N_LAYERS}")

    # ── Optimiser: linear warmup then cosine decay ─────────────────────────────
    schedule = optax.join_schedules(
        [optax.linear_schedule(0.0, LR_MAX, WARMUP),
         optax.cosine_decay_schedule(LR_MAX, N_STEPS - WARMUP, alpha=LR_MIN / LR_MAX)],
        boundaries=[WARMUP],
    )
    optimizer = optax.adamw(schedule, weight_decay=1e-2)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # ── Loss (batched) ─────────────────────────────────────────────────────────
    @eqx.filter_jit
    def train_step(model, opt_state, batch_tokens, batch_targets):
        def loss_fn(m):
            def single(tok, tgt):
                logits = m.get_logits(tok)
                return jnp.mean(
                    jax.nn.logsumexp(logits, -1) - logits[jnp.arange(len(tgt)), tgt]
                )
            return jnp.mean(jax.vmap(single)(batch_tokens, batch_targets))

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        return eqx.apply_updates(model, updates), opt_state, loss

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\nTraining {N_STEPS} steps (batch={BATCH_SIZE}, seq={SEQ_LEN}) — "
          f"target PPL < {TARGET_PPL} ...\n")

    t_train_start = time.time()
    tokens_seen   = 0
    final_step    = N_STEPS

    for step in range(N_STEPS):
        key, batch_key = jr.split(key)
        bt, tgt = get_batch(batch_key, train_data, BATCH_SIZE, SEQ_LEN)

        t0 = time.time()
        model, opt_state, loss = train_step(model, opt_state, bt, tgt)
        jax.block_until_ready(loss)
        step_ms = (time.time() - t0) * 1000

        tokens_seen += BATCH_SIZE * SEQ_LEN

        if step % LOG_EVERY == 0 or step == N_STEPS - 1:
            val_logits = model.get_logits(val_tokens)
            val_ppl    = float(compute_ppl(val_logits, val_targets))
            elapsed    = time.time() - t_train_start
            tok_per_s  = tokens_seen / elapsed
            print(f"step {step:5d} | loss {float(loss):.3f} | val PPL {val_ppl:7.2f} | "
                  f"{step_ms:5.0f} ms/step | {tok_per_s:,.0f} tok/s")

            if val_ppl < TARGET_PPL and step > 0:
                print(f"\n✓ Reached PPL {val_ppl:.2f} < {TARGET_PPL} at step {step}.")
                final_step = step
                break

    total_time = time.time() - t_train_start
    print(f"\nTraining done in {total_time/60:.1f} min | "
          f"{tokens_seen / total_time:,.0f} tok/s overall\n")

    # ── Final baseline PPL ────────────────────────────────────────────────────
    baseline_logits = model.get_logits(val_tokens)
    baseline_ppl    = float(compute_ppl(baseline_logits, val_targets))
    print(f"Baseline PPL (before FTA): {baseline_ppl:.2f}")

    # ── FTA refinement ────────────────────────────────────────────────────────
    key, fta_key = jr.split(key)
    fta = FlashThermodynamicAttention(
        d_model=D_MODEL, n_heads=N_HEADS, n_diffusion_steps=2, key=fta_key
    )
    embed_table = model.embed.weight

    refined_logits = low_ppl_refine(baseline_logits, fta, embed_table, key=key, blend=0.7)
    fta_ppl = float(compute_ppl(refined_logits, val_targets))

    print(f"FTA Multi-Head PPL (after refinement): {fta_ppl:.2f}")
    delta = baseline_ppl - fta_ppl
    print(f"PPL change: {delta:+.2f}  ({delta / baseline_ppl * 100:+.1f}%)")
