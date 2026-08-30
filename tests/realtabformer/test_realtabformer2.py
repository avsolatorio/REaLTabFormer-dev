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


# --- any_order (beta): arbitrary-subset conditioning -----------------------


def test_any_order_requires_shared_numeric_vocab():
    with pytest.raises(ValueError, match="shared_numeric_vocab"):
        REaLTabFormer2(
            model_type="tabular",
            tabular_backbone="distilgpt2",
            any_order=True,
            shared_numeric_vocab=False,
        )


def test_any_order_default_is_off():
    model = REaLTabFormer2(
        model_type="tabular", tabular_backbone="distilgpt2", shared_numeric_vocab=True
    )
    assert model.any_order is False
    assert model.column_blocks is None


def _fit_any_order_model(df: pd.DataFrame) -> REaLTabFormer2:
    model = REaLTabFormer2(
        model_type="tabular",
        epochs=1,
        batch_size=8,
        tabular_backbone="distilgpt2",
        shared_numeric_vocab=True,
        any_order=True,
    )
    model.fit(df, device="cpu", n_critic=0)
    return model


def test_any_order_end_to_end_fit_and_unconditional_sample():
    df = _tiny_df(n_rows=60, seed=13)
    model = _fit_any_order_model(df)

    block_names = [name for name, _ in model.column_blocks]
    assert block_names == ["price", "age", "gender"]

    samples = model.sample(n_samples=5, device="cpu", gen_batch=5)
    assert len(samples) == 5
    assert list(samples.columns) == list(df.columns)


def test_any_order_seed_on_last_column_only():
    # Under fixed-order training this would be impossible: "gender" is the
    # *last* column in canonical order, not a prefix, so a fixed-order
    # model can never condition on it alone. any_order=True is trained to
    # be robust to any column order specifically so this works.
    #
    # Checks *both* categorical values, not just one: a binary column has
    # a 50% chance of "passing" by luck alone via the (unrelated)
    # OOV-fallback random-value path if the seed encoding were broken --
    # exactly how a real bug here (process_data's fresh-vs-fit-time index
    # mismatch corrupting the seeded *value*, not just observed for a
    # reordered/gapped subset) initially slipped past a single-value
    # version of this test during development, caught only by the
    # save/load test's independently-chosen seed value.
    df = _tiny_df(n_rows=60, seed=17)
    model = _fit_any_order_model(df)

    for seed_val in ["m", "f"]:
        seed_input = pd.DataFrame({"gender": [seed_val]})
        samples = model.sample(
            n_samples=5, gen_batch=5, device="cpu", seed_input=seed_input
        )
        assert (samples["gender"] == seed_val).all(), (
            f"seed_input gender={seed_val!r} not preserved: "
            f"{samples['gender'].tolist()}"
        )


def test_any_order_seed_on_middle_column_skipping_earlier_column():
    # Seeds on "age" (the middle column) while skipping "price" (the
    # first column) entirely -- an arbitrary subset *with a gap*, not
    # just a suffix. This is the case that specifically exercises the
    # process_data re-indexing realignment (_realign_seed_columns):
    # process_data assigns fresh 0-based indices to whatever subset/order
    # of columns it's given, which don't match the fit-time indices
    # unless corrected.
    df = _tiny_df(n_rows=60, seed=19)
    model = _fit_any_order_model(df)

    seed_age = float(df["age"].iloc[0])
    seed_input = pd.DataFrame({"age": [seed_age]})
    samples = model.sample(n_samples=5, gen_batch=5, device="cpu", seed_input=seed_input)

    assert (samples["age"] == seed_age).all(), (
        f"seed_input age={seed_age} was not preserved: {samples['age'].tolist()}"
    )


def test_any_order_save_load_roundtrip_preserves_non_prefix_seeding(tmp_path):
    df = _tiny_df(n_rows=60, seed=23)
    model = _fit_any_order_model(df)

    save_dir = tmp_path / "any_order_model"
    model.save(save_dir)

    # save() creates save_dir/<experiment_id>/ and stores artefacts there.
    reloaded = REaLTabFormer2.load_from_dir(save_dir / model.experiment_id)
    assert reloaded.any_order is True
    assert reloaded.column_blocks == model.column_blocks

    seed_input = pd.DataFrame({"gender": ["f"]})
    samples = reloaded.sample(n_samples=5, gen_batch=5, device="cpu", seed_input=seed_input)
    assert (samples["gender"] == "f").all()


def test_any_order_conditioning_shifts_dependent_column_distribution():
    # Statistical sanity check beyond "didn't crash": construct data with
    # a strong, *coarse* dependency: "bucket" is a categorical column
    # fully determined by which half of "a"'s range a falls in. A coarse
    # threshold signal like this is far easier for a tiny model to pick
    # up in a handful of epochs on a small dataset than an exact
    # digit-level numeric relationship would be -- the point of this test
    # is to check that conditioning propagates through the any-order
    # machinery at all, not to benchmark fidelity on a hard regression.
    rng = np.random.default_rng(29)
    n = 300
    a = rng.integers(1000, 1999, size=n).astype(float)
    bucket = np.where(a < 1500, "low", "high")
    df = pd.DataFrame({
        "a": a, "bucket": bucket, "gender": rng.choice(["m", "f"], size=n),
    })

    model = REaLTabFormer2(
        model_type="tabular",
        epochs=20,
        batch_size=16,
        tabular_backbone="distilgpt2",
        shared_numeric_vocab=True,
        any_order=True,
    )
    model.fit(df, device="cpu", n_critic=0)

    seeded_low = model.sample(
        n_samples=40, gen_batch=40, device="cpu",
        seed_input=pd.DataFrame({"bucket": ["low"]}),
    )
    seeded_high = model.sample(
        n_samples=40, gen_batch=40, device="cpu",
        seed_input=pd.DataFrame({"bucket": ["high"]}),
    )

    assert (seeded_low["bucket"] == "low").all()
    assert (seeded_high["bucket"] == "high").all()

    assert seeded_low["a"].mean() < seeded_high["a"].mean(), (
        "Conditioning on bucket='low' vs 'high' (a non-prefix column) "
        "should shift generated 'a' accordingly: "
        f"low-seeded mean={seeded_low['a'].mean()}, "
        f"high-seeded mean={seeded_high['a'].mean()}"
    )
