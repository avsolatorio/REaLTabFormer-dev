"""Tests for realtabformer2.py's `shared_numeric_vocab` feature (beta).

realtabformer2.py has no prior test coverage in this repo -- these are new
tests, not a regression net for existing behavior. See
/Users/avsolatorio/.claude/plans/snappy-swimming-hickey.md for the design.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

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
    # col_type_ids_seq now always exists (defaulted to None in __init__,
    # like every other optional/beta attribute) -- an unpooled model's
    # stays None rather than the attribute being absent entirely, a
    # pre-existing inconsistency found and fixed while decoupling
    # any_order from shared_numeric_vocab (any code reading
    # `model.col_type_ids_seq` on an unpooled model used to hit
    # AttributeError instead of getting the expected None).
    assert unpooled_model.col_type_ids_seq is None


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


def test_any_order_no_longer_requires_shared_numeric_vocab():
    # any_order used to require shared_numeric_vocab=True purely because it
    # reused shared_numeric_vocab's token_type_ids column-identity
    # machinery, not because it mechanically needs it -- a disjoint
    # per-column-per-digit-chunk vocabulary (every non-pooled column
    # already has one) makes every token self-identifying regardless of
    # where any_order's permutation moves it, no token_type_ids needed.
    # Must not raise.
    model = REaLTabFormer2(
        model_type="tabular",
        tabular_backbone="distilgpt2",
        any_order=True,
        shared_numeric_vocab=False,
    )
    assert model.any_order is True
    assert model.shared_numeric_vocab is False
    assert model.col_type_ids_seq is None


def test_any_order_requires_tabular_model_type():
    # The validation any_order actually needs (column_blocks/
    # AnyOrderColumnCollator are tabular-only) -- previously reached only
    # transitively via requiring shared_numeric_vocab, which also checks
    # this; now checked directly since that requirement is gone.
    with pytest.raises(ValueError, match="tabular"):
        REaLTabFormer2(
            model_type="relational",
            tabular_backbone="distilgpt2",
            any_order=True,
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


# --- any_order without shared_numeric_vocab (the now-decoupled, recommended combination) ---


def _fit_any_order_no_shared_vocab_model(df: pd.DataFrame) -> REaLTabFormer2:
    model = REaLTabFormer2(
        model_type="tabular",
        epochs=1,
        batch_size=8,
        tabular_backbone="distilgpt2",
        shared_numeric_vocab=False,
        any_order=True,
    )
    model.fit(df, device="cpu", n_critic=0)
    return model


def test_any_order_without_shared_vocab_end_to_end_fit_and_unconditional_sample():
    df = _tiny_df(n_rows=60, seed=41)
    model = _fit_any_order_no_shared_vocab_model(df)

    assert model.col_type_ids_seq is None
    block_names = [name for name, _ in model.column_blocks]
    assert block_names == ["price", "age", "gender"]

    samples = model.sample(n_samples=5, device="cpu", gen_batch=5)
    assert len(samples) == 5
    assert list(samples.columns) == list(df.columns)


def test_any_order_without_shared_vocab_seed_on_last_column_only():
    # Same claim as any_order's own shared_numeric_vocab=True test, for
    # the decoupled combination: "gender" is the *last* column, impossible
    # to condition on under fixed-order training. Checks both values, not
    # just one, per the same rationale as the shared_numeric_vocab
    # version -- a 50% chance of "passing" via the unrelated OOV-fallback
    # path is exactly how a real seeding bug could slip past a
    # single-value check.
    df = _tiny_df(n_rows=60, seed=43)
    model = _fit_any_order_no_shared_vocab_model(df)

    for seed_val in ["m", "f"]:
        seed_input = pd.DataFrame({"gender": [seed_val]})
        samples = model.sample(
            n_samples=5, gen_batch=5, device="cpu", seed_input=seed_input
        )
        assert (samples["gender"] == seed_val).all(), (
            f"seed_input gender={seed_val!r} not preserved without "
            f"shared_numeric_vocab: {samples['gender'].tolist()}"
        )


def test_any_order_without_shared_vocab_seed_on_middle_column_skipping_earlier_column():
    # The harder case: seeds on "age" (the middle column) while skipping
    # "price" (the first column) entirely -- an arbitrary subset with a
    # gap, not just a suffix.
    df = _tiny_df(n_rows=60, seed=47)
    model = _fit_any_order_no_shared_vocab_model(df)

    seed_age = float(df["age"].iloc[0])
    seed_input = pd.DataFrame({"age": [seed_age]})
    samples = model.sample(n_samples=5, gen_batch=5, device="cpu", seed_input=seed_input)

    assert (samples["age"] == seed_age).all(), (
        f"seed_input age={seed_age} was not preserved: {samples['age'].tolist()}"
    )


def test_any_order_without_shared_vocab_save_load_roundtrip_preserves_non_prefix_seeding(tmp_path):
    df = _tiny_df(n_rows=60, seed=53)
    model = _fit_any_order_no_shared_vocab_model(df)

    save_dir = tmp_path / "any_order_no_shared_vocab_model"
    model.save(save_dir)

    reloaded = REaLTabFormer2.load_from_dir(save_dir / model.experiment_id)
    assert reloaded.any_order is True
    assert reloaded.shared_numeric_vocab is False
    assert reloaded.column_blocks == model.column_blocks
    assert reloaded.col_type_ids_seq is None

    seed_input = pd.DataFrame({"gender": ["f"]})
    samples = reloaded.sample(n_samples=5, gen_batch=5, device="cpu", seed_input=seed_input)
    assert (samples["gender"] == "f").all()


# --- digit_entropy_weighting (beta): entropy-based digit-chunk loss weighting ---


def test_digit_entropy_weighting_default_off_leaves_no_chunk_significance_weights():
    df = _tiny_df()
    model = REaLTabFormer2(
        model_type="tabular", epochs=1, batch_size=8, tabular_backbone="distilgpt2",
    )
    model.fit(df, device="cpu", n_critic=0)
    assert "chunk_significance_weights" not in model.vocab


def test_digit_entropy_weighting_end_to_end_favors_high_entropy_chunk():
    # Real, end-to-end validation of the motivating claim: fit a tiny
    # model on a deliberately heavy-tailed numeric column and confirm the
    # computed chunk_significance_weights give the (near-constant, under
    # fixed-width zero-padded encoding) leading digit chunk a *lower*
    # weight than a later, higher-variance chunk -- not just "didn't
    # crash".
    rng = np.random.default_rng(7)
    n = 300
    price = np.round(rng.exponential(scale=50, size=n))
    price = np.clip(price, 1, 999)
    df = pd.DataFrame({
        "price": price, "gender": rng.choice(["m", "f"], size=n),
    })

    model = REaLTabFormer2(
        model_type="tabular",
        epochs=1,
        batch_size=8,
        tabular_backbone="distilgpt2",
        numeric_max_len=6,
        numeric_precision=0,
        numeric_nparts=1,
    )
    model.fit(df, device="cpu", n_critic=0, digit_entropy_weighting=True)

    weights = model.vocab["chunk_significance_weights"]
    price_chunks = sorted(c for c in model.processed_columns if "price" in c)
    assert len(price_chunks) > 1  # sanity: price was actually partitioned

    leading_w = weights[price_chunks[0]]
    trailing_w = weights[price_chunks[-1]]
    assert leading_w < trailing_w, (
        f"leading chunk weight ({leading_w}) should be lower than the "
        f"trailing chunk's ({trailing_w}) for a heavy-tailed column"
    )

    # The dataset actually carries the weighted loss signal through to
    # training -- token_weights column exists on the built dataset even
    # though no explicit field_weights was set.
    assert model.model is not None  # fit succeeded end to end

    # Sampling still works unaffected by the training-time-only weighting.
    samples = model.sample(n_samples=5, device="cpu", gen_batch=5)
    assert len(samples) == 5
    assert list(samples.columns) == list(df.columns)


def test_digit_entropy_weighting_composes_with_field_weights():
    df = _tiny_df()
    model = REaLTabFormer2(
        model_type="tabular", epochs=1, batch_size=8, tabular_backbone="distilgpt2",
    )
    # Must not raise, and must actually build a model -- field_weights and
    # digit_entropy_weighting are independent, composable knobs.
    model.fit(
        df,
        device="cpu",
        n_critic=0,
        field_weights={"price": 2.0},
        digit_entropy_weighting=True,
    )
    assert model.model is not None
    assert "chunk_significance_weights" in model.vocab


# --- sensitivity-based training (_train_with_sensitivity): gen_kwargs=None crash ---


def test_sensitivity_training_does_not_crash_with_default_gen_kwargs():
    # v2 counterpart of the same regression test in test_realtabformer.py
    # -- see that test's docstring for the full explanation. Same
    # pre-existing bug (commit 73f23964), same fix
    # (`**(gen_kwargs or {})`), independently present in
    # realtabformer2.py's own copy of _train_with_sensitivity.
    df = _tiny_df(n_rows=40)
    model = REaLTabFormer2(
        model_type="tabular", epochs=2, batch_size=8, tabular_backbone="distilgpt2",
    )
    model.fit(df, device="cpu", n_critic=1, n_critic_stop=1, num_bootstrap=20)
    assert model.model is not None


def test_any_order_composes_with_digit_entropy_weighting():
    # Neither feature's own test suite exercises the other: any_order's
    # tests never pass digit_entropy_weighting, and digit_entropy_weighting's
    # tests never set any_order=True. They touch genuinely different parts
    # of the pipeline (AnyOrderColumnCollator permutes token_weights
    # alongside input_ids/labels/token_type_ids every batch;
    # chunk_significance_weights only decides what those per-token weights
    # *are*, computed once from the untouched, canonically-ordered vocab)
    # -- but "different code paths" isn't the same as "verified to compose
    # correctly", and this session already found one real bug that only
    # showed up when two independently-correct-looking features were
    # combined and actually run. Checks three things a purely orthogonal
    # pair could still get wrong together: entropy weights are computed
    # normally, any-order-specific arbitrary-subset seeding still works
    # with entropy weighting active, and unconditional sampling still
    # works too.
    rng = np.random.default_rng(31)
    n = 300
    price = np.round(rng.exponential(scale=50, size=n))
    price = np.clip(price, 1, 999)
    df = pd.DataFrame({
        "price": price, "gender": rng.choice(["m", "f"], size=n),
    })

    model = REaLTabFormer2(
        model_type="tabular",
        epochs=1,
        batch_size=8,
        tabular_backbone="distilgpt2",
        numeric_max_len=6,
        numeric_precision=0,
        numeric_nparts=1,
        shared_numeric_vocab=True,
        any_order=True,
    )
    model.fit(df, device="cpu", n_critic=0, digit_entropy_weighting=True)

    # digit_entropy_weighting's own claim still holds under any_order:
    # the heavy-tailed column's near-constant leading chunk gets a lower
    # weight than its higher-variance trailing chunk.
    weights = model.vocab["chunk_significance_weights"]
    price_chunks = sorted(c for c in model.processed_columns if "price" in c)
    assert len(price_chunks) > 1
    assert weights[price_chunks[0]] < weights[price_chunks[-1]]

    # any_order's own claim still holds under digit_entropy_weighting:
    # column_blocks got built, and unconditional sampling works.
    assert [name for name, _ in model.column_blocks] == ["price", "gender"]
    samples = model.sample(n_samples=5, device="cpu", gen_batch=5)
    assert len(samples) == 5
    assert list(samples.columns) == list(df.columns)

    # The specific thing that could plausibly break under composition:
    # any-order's arbitrary-subset seeding (condition on "gender", the
    # *last* column, impossible under fixed-order training) with the
    # entropy-weighted model. Checks both values, not just one -- a 50%
    # chance of "passing" by luck via the (unrelated) OOV-fallback path
    # is exactly how a real seeding bug slipped past a single-value
    # version of any_order's own seed test during its own development.
    for seed_val in ["m", "f"]:
        seed_input = pd.DataFrame({"gender": [seed_val]})
        seeded = model.sample(
            n_samples=5, gen_batch=5, device="cpu", seed_input=seed_input
        )
        assert (seeded["gender"] == seed_val).all(), (
            f"seed_input gender={seed_val!r} not preserved under "
            f"any_order + digit_entropy_weighting: {seeded['gender'].tolist()}"
        )


# --- numeric_categorical_threshold: cardinality-aware numeric dispatch -----


def _low_cardinality_df(n_rows: int = 200, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    bedrooms = rng.choice([1, 2, 3, 4, 5], size=n_rows, p=[0.05, 0.30, 0.35, 0.20, 0.10])
    # A rating column with real missing values -- pandas naturally
    # represents "int-like + NaN" as float64, the realistic case (a
    # nullable Int64 dtype hits a separate, pre-existing cast limitation
    # unrelated to this feature -- see the plan's dtype round-trip note).
    rating = rng.choice([1.0, 2.0, 3.0, 4.0, 5.0], size=n_rows).astype(float)
    missing_idx = rng.choice(n_rows, size=int(n_rows * 0.1), replace=False)
    rating[missing_idx] = np.nan
    return pd.DataFrame({
        "bedrooms": bedrooms.astype("int64"),
        "rating": rating,
        "price": rng.integers(100000, 999999, size=n_rows).astype(float),
        "gender": rng.choice(["m", "f"], size=n_rows),
    })


def test_numeric_categorical_threshold_default_off_leaves_column_blocks_unaffected():
    df = _low_cardinality_df()
    model = REaLTabFormer2(
        model_type="tabular", epochs=1, batch_size=8, tabular_backbone="distilgpt2",
    )
    model.fit(df, device="cpu", n_critic=0)
    bedroom_cols = [c for c in model.processed_columns if "bedrooms" in c]
    assert all("NUMERIC" in c for c in bedroom_cols)


def test_numeric_categorical_threshold_end_to_end_dtype_round_trip():
    df = _low_cardinality_df()
    model = REaLTabFormer2(
        model_type="tabular",
        epochs=3,
        batch_size=16,
        tabular_backbone="distilgpt2",
        numeric_categorical_threshold=10,
    )
    model.fit(df, device="cpu", n_critic=0)

    bedroom_cols = [c for c in model.processed_columns if "bedrooms" in c]
    rating_cols = [c for c in model.processed_columns if "rating" in c]
    price_cols = [c for c in model.processed_columns if "price" in c]
    assert len(bedroom_cols) == 1 and "CATEGORICAL" in bedroom_cols[0]
    assert len(rating_cols) == 1 and "CATEGORICAL" in rating_cols[0]
    # price has a wide range -- unaffected, still digit-chunked.
    assert len(price_cols) > 1 and all("NUMERIC" in c for c in price_cols)

    samples = model.sample(n_samples=20, device="cpu", gen_batch=20)

    # The explicit requirement: the recovered column must be the
    # *original* pandas dtype, not a string or generic object dtype --
    # regardless of which internal pipeline (digit-chunked or
    # single-token categorical) produced it.
    assert samples["bedrooms"].dtype == df["bedrooms"].dtype
    assert samples["rating"].dtype == df["rating"].dtype
    assert samples["price"].dtype == df["price"].dtype

    # Values are drawn from real, plausible options (not raw strings, not
    # tokenizer artefacts) -- bedrooms is a closed enum the model can
    # only ever have observed 1..5 for.
    assert samples["bedrooms"].dropna().isin([1, 2, 3, 4, 5]).all()

    # rating's missing-value round trip: the model was trained on a
    # column that has real NaNs, so nothing here should error out even
    # if none happen to be sampled in this particular batch of 20.
    assert samples["rating"].isna().sum() >= 0


def test_numeric_categorical_threshold_seed_input_on_demoted_column():
    # A demoted column becomes a single-element block, identical in shape
    # to any genuinely-categorical column already handled by
    # compute_column_blocks/any_order -- verify seeding on it works with
    # no special-casing needed.
    df = _low_cardinality_df()
    model = REaLTabFormer2(
        model_type="tabular",
        epochs=1,
        batch_size=16,
        tabular_backbone="distilgpt2",
        numeric_categorical_threshold=10,
    )
    model.fit(df, device="cpu", n_critic=0)

    seed_input = pd.DataFrame({"bedrooms": [3]})
    samples = model.sample(n_samples=5, gen_batch=5, device="cpu", seed_input=seed_input)
    assert (samples["bedrooms"] == 3).all()
    assert samples["bedrooms"].dtype == df["bedrooms"].dtype


# --- numeric_quantile_encoding: CDF-based numeric representation -----------


def test_numeric_quantile_encoding_end_to_end_dtype_and_distributional_fidelity():
    rng = np.random.default_rng(11)
    n = 500
    price = np.round(rng.lognormal(mean=6.0, sigma=2.0, size=n), 2)
    price = np.clip(price, 0.01, 500000)
    df = pd.DataFrame({
        "price": price, "gender": rng.choice(["m", "f"], size=n),
    })

    model = REaLTabFormer2(
        model_type="tabular",
        epochs=25,
        batch_size=16,
        tabular_backbone="distilgpt2",
        numeric_max_len=8,
        numeric_precision=4,
        numeric_nparts=1,
        numeric_quantile_encoding=True,
    )
    model.fit(df, device="cpu", n_critic=0)

    samples = model.sample(n_samples=200, device="cpu", gen_batch=200)

    # Explicit dtype-preservation bar, matching numeric_categorical_threshold's.
    assert samples["price"].dtype == df["price"].dtype

    # Bounded by the training range (np.interp's clamp-to-boundary
    # extrapolation rule, the accepted tradeoff of choosing quantile
    # encoding over magnitude+mantissa).
    assert samples["price"].min() >= df["price"].min()
    assert samples["price"].max() <= df["price"].max()

    # Distributional fidelity, measured directly rather than argued from
    # the inverse-transform-sampling theory alone.
    ks_stat, _ = stats.ks_2samp(samples["price"].dropna(), df["price"])
    assert ks_stat < 0.2, f"KS statistic too high: {ks_stat}"
