"""Tests for realtabformer2.py's `shared_numeric_vocab` feature (beta).

realtabformer2.py has no prior test coverage in this repo -- these are new
tests, not a regression net for existing behavior. See
/Users/avsolatorio/.claude/plans/snappy-swimming-hickey.md for the design.
"""
import numpy as np
import pandas as pd
import pytest

from realtabformer.realtabformer2 import REaLTabFormer2


def _tiny_df(n_rows: int = 40, seed: int = 1029) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "price": rng.normal(100, 20, size=n_rows).round(2),
        "age": rng.integers(18, 90, size=n_rows).astype(float),
        "gender": rng.choice(["m", "f"], size=n_rows),
    })


def test_shared_numeric_vocab_requires_tabular_model_type():
    with pytest.raises(ValueError):
        REaLTabFormer2(
            model_type="relational",
            shared_numeric_vocab=True,
        )


def test_shared_numeric_vocab_rejects_incompatible_backbone():
    # GPT-NeoX-family models don't accept `token_type_ids` in forward() --
    # confirmed directly against the installed transformers version.
    from transformers import GPTNeoXConfig

    cfg = GPTNeoXConfig(
        vocab_size=100, hidden_size=32, num_hidden_layers=2, num_attention_heads=2
    )
    with pytest.raises(ValueError, match="token_type_ids"):
        REaLTabFormer2(
            model_type="tabular",
            tabular_config=cfg,
            shared_numeric_vocab=True,
        )


def test_shared_numeric_vocab_accepts_gpt2_family_backbone():
    # Should not raise -- distilgpt2 (the default backbone) is GPT2-family.
    model = REaLTabFormer2(
        model_type="tabular",
        tabular_backbone="distilgpt2",
        shared_numeric_vocab=True,
    )
    assert model.shared_numeric_vocab is True


def test_shared_numeric_vocab_default_is_off():
    model = REaLTabFormer2(model_type="tabular", tabular_backbone="distilgpt2")
    assert model.shared_numeric_vocab is False


def test_shared_numeric_vocab_end_to_end_fit_and_sample():
    # The real integration test: a tiny model must fit and sample without
    # error, and the pooled vocab must actually be smaller than the
    # unpooled one would be for the same data (proving pooling took
    # effect, not just that nothing crashed).
    df = _tiny_df()

    pooled_model = REaLTabFormer2(
        model_type="tabular",
        epochs=1,
        batch_size=8,
        tabular_backbone="distilgpt2",
        shared_numeric_vocab=True,
    )
    pooled_model.fit(df, device="cpu", n_critic=0)

    assert hasattr(pooled_model, "col_type_ids_seq")
    assert pooled_model.col_type_ids_seq[0] == pooled_model.col_type_ids_seq[-1]
    assert len(pooled_model.col_type_ids_seq) == pooled_model.tabular_max_length

    samples = pooled_model.sample(n_samples=5, device="cpu", gen_batch=5)
    assert len(samples) == 5
    assert list(samples.columns) == list(df.columns)

    unpooled_model = REaLTabFormer2(
        model_type="tabular",
        epochs=1,
        batch_size=8,
        tabular_backbone="distilgpt2",
    )
    unpooled_model.fit(df, device="cpu", n_critic=0)

    # Pooling numeric columns' vocab must strictly shrink vocab size
    # relative to the unpooled (per-column) baseline on the same data --
    # this is the entire point of the feature, not just "didn't crash".
    assert pooled_model.tabular_config.vocab_size < unpooled_model.tabular_config.vocab_size
    assert not hasattr(unpooled_model, "col_type_ids_seq")


def test_shared_numeric_vocab_seed_input_preserves_numeric_value():
    # Regression test: `_process_seed_input` (rtf_sampler.py) used to
    # unconditionally tokenize seed values with `make_dataset`, which
    # looks up the still column-prefixed value in a vocab keyed on the
    # prefix-*stripped* value for pooled numeric/datetime columns (see
    # `build_pooled_vocab`). That lookup always missed for
    # shared_numeric_vocab=True models, silently falling back to a
    # *random* token from the column's own vocabulary -- so every
    # numeric/datetime seed value was replaced with noise, no matter how
    # common that exact value was in training. Fixed by routing pooled-
    # vocab models through `make_dataset_with_column_types` instead,
    # which applies the same prefix-stripping vocab lookup used at
    # training time.
    df = _tiny_df(n_rows=60, seed=7)
    # Use a distinct, easy-to-verify integer price so any corruption in
    # the digit-chunk tokens is unambiguous (not just a rounding quirk).
    df["price"] = np.random.default_rng(7).integers(1000, 9999, size=len(df)).astype(float)

    model = REaLTabFormer2(
        model_type="tabular",
        epochs=1,
        batch_size=8,
        tabular_backbone="distilgpt2",
        shared_numeric_vocab=True,
    )
    model.fit(df, device="cpu", n_critic=0)

    # Seed with an *actual* training-set value -- guaranteed to be
    # in-vocabulary, so any mismatch can only be a tokenization bug, not
    # legitimate OOV handling for an unseen value.
    seed_price = float(df["price"].iloc[0])
    seed_input = pd.DataFrame({"price": [seed_price]})

    samples = model.sample(n_samples=5, gen_batch=5, device="cpu", seed_input=seed_input)

    assert (samples["price"] == seed_price).all(), (
        f"seed_input price={seed_price} was not preserved in generated "
        f"samples: {samples['price'].tolist()}"
    )
