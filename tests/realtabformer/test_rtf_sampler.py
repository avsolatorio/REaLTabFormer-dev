"""This suite tests for the rtf_sampler.py module."""
import torch
import fixtures as fx

from realtabformer.rtf_sampler import TabularSampler


def test_TabularSampler():
    tab_fx = fx.tabular_fixtures(fit=True)
    rtf_model = tab_fx["rtf_model"]
    df = tab_fx["df"]
    n_samples = 100
    gen_batch = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    constrain_tokens_gen = True
    continuous_empty_limit = 10
    suppress_tokens = None

    tabular_sampler = TabularSampler(
        model_type=rtf_model.model_type,
        model=rtf_model.model,
        vocab=rtf_model.vocab,
        processed_columns=rtf_model.processed_columns,
        max_length=rtf_model.tabular_max_length,
        col_size=rtf_model.tabular_col_size,
        col_idx_ids=rtf_model.col_idx_ids,
        columns=rtf_model.columns,
        datetime_columns=rtf_model.datetime_columns,
        column_dtypes=rtf_model.column_dtypes,
        drop_na_cols=rtf_model.drop_na_cols,
        col_transform_data=rtf_model.col_transform_data,
        random_state=rtf_model.random_state,
        device=device,
    )

    pred_df = tabular_sampler.sample_tabular(
        n_samples=n_samples,
        gen_batch=gen_batch,
        device=device,
        constrain_tokens_gen=constrain_tokens_gen,
        continuous_empty_limit=continuous_empty_limit,
        suppress_tokens=suppress_tokens,
    )

    # Check that the columns are ordered in the
    # same way as the input data.
    assert all(df.columns == pred_df.columns)

    # Check that the data types are similar to
    # the training data.
    dtypes = df.dtypes
    pred_dtypes = pred_df.dtypes

    for col in dtypes.index:
        assert dtypes[col] == pred_dtypes[col]


# --- Vectorised constrained decoding (ColumnMaskLogitsProcessor) ---------


def _tiny_fitted_model(tmp_path):
    import numpy as np
    import pandas as pd
    from transformers import GPT2Config

    from realtabformer import REaLTabFormer

    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "a": rng.choice(list("xyz"), 120),
            "b": rng.integers(0, 50, 120),
            "c": rng.choice(["p", "q"], 120),
        }
    )
    model = REaLTabFormer(
        model_type="tabular",
        epochs=1,
        batch_size=16,
        checkpoints_dir=str(tmp_path / "ckpt"),
        tabular_config=GPT2Config(n_layer=1, n_embd=32, n_head=2),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.fit(df, device=device, n_critic=0)
    return model, device


def _generate(sampler, model, device, seed, n=64):
    import numpy as np

    from realtabformer.data_utils import SpecialTokens

    t2i = model.vocab["token2id"]
    torch.manual_seed(seed)
    return sampler._generate(
        device=torch.device(device),
        as_numpy=True,
        constrain_tokens_gen=True,
        inputs=torch.tensor([[t2i[SpecialTokens.BOS]]], device=device),
        do_sample=True,
        max_length=model.tabular_max_length,
        num_return_sequences=n,
        bos_token_id=t2i[SpecialTokens.BOS],
        pad_token_id=t2i[SpecialTokens.PAD],
        eos_token_id=t2i[SpecialTokens.EOS],
        suppress_tokens=None,
        forced_decoder_ids=None,
    )


def test_vectorized_constraint_matches_callback_and_respects_columns(tmp_path):
    import numpy as np

    model, device = _tiny_fitted_model(tmp_path)
    sampler = TabularSampler.sampler_from_model(model, device=device)

    try:
        TabularSampler.vectorized_constraint = True
        fast = _generate(sampler, model, device, seed=3)
        TabularSampler.vectorized_constraint = False
        slow = _generate(sampler, model, device, seed=3)
    finally:
        TabularSampler.vectorized_constraint = True

    # Same rule, same seed => identical token sequences.
    assert np.array_equal(fast, slow)

    # And every generated token is valid for its column.
    for step, allowed in model.col_idx_ids.items():
        assert np.isin(fast[:, step + 1], allowed).all()


def test_ColumnMaskLogitsProcessor_masks_disallowed_tokens():
    from realtabformer.rtf_sampler import ColumnMaskLogitsProcessor

    proc = ColumnMaskLogitsProcessor(
        col_idx_ids={0: [2, 3], 1: [4]},
        eos_token_id=1,
        vocab_size=6,
        max_steps=2,
        device=torch.device("cpu"),
    )
    scores = torch.zeros(2, 6)

    step0 = proc(torch.zeros(2, 1, dtype=torch.long), scores)  # 1 token so far
    assert torch.isfinite(step0[:, [2, 3]]).all()
    assert torch.isinf(step0[:, [0, 1, 4, 5]]).all()

    step1 = proc(torch.zeros(2, 2, dtype=torch.long), scores)
    assert torch.isfinite(step1[:, 4]).all()
    assert torch.isinf(step1[:, [0, 1, 2, 3, 5]]).all()

    # Past the last column only EOS is allowed, however long the sequence.
    late = proc(torch.zeros(2, 50, dtype=torch.long), scores)
    assert torch.isfinite(late[:, 1]).all()
    assert torch.isinf(late[:, [0, 2, 3, 4, 5]]).all()
