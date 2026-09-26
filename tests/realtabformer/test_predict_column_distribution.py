"""Tests for TabularSampler.predict_column_distribution -- extracting a REaLTabFormer2(any_order=True)
model's own calibrated distribution over ONE target column given an arbitrary-subset seed, from a
single forward pass, rather than sampling.

Built for a downstream task (SimulacraBench-style calibrated per-cell prediction, see the sibling
`simulcrabench` repo's own lab notebook) that needs the model's own belief over a specific blank
column given whatever else is visible, not a realization of it.

`log_prob_of_sequences` (rtf_rl.py) is flagged in its own docstring as the single easiest place for
a silent, invisible bug -- a mask that doesn't exactly match the sampler's own constrained-decoding
distribution. This carries the same risk (it reuses the exact same masking/step-index machinery),
so it gets the same discipline: enumerate a tiny toy schema completely, check the extracted
distribution sums to 1, and cross-check it against a large empirical resample under the same seed,
rather than trust it because the code runs.
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import GPT2Config

from realtabformer.realtabformer2 import REaLTabFormer2
from realtabformer.rtf_sampler import TabularSampler


def _fit_toy_model(tmp_path, seed=0):
    """Three columns, deliberately with a real (not independent) relationship between "a" and "c":
    c is EXACTLY determined by a in this toy data (a="x" -> c is always "p", a="y" -> always "q"),
    so a correctly-conditioned P(c | a) should collapse almost entirely onto one value -- a
    falsifiable, hand-known target the enumeration test below can check against, not just "sums
    to 1". "b" is independent noise, present so `target_col="c"` genuinely has to skip over it in
    the generation order (exercising the actual `target_col` insertion this test is for) rather
    than happening to be the only remaining column."""
    rng = np.random.default_rng(seed)
    n = 600
    a = rng.choice(list("xy"), n)
    b = rng.choice(list("pqr"), n)
    c = np.where(a == "x", "p", "q")
    df = pd.DataFrame({"a": a, "b": b, "c": c})
    model = REaLTabFormer2(
        model_type="tabular", any_order=True, shared_numeric_vocab=False,
        tabular_config=GPT2Config(n_layer=2, n_embd=32, n_head=2),
        # any_order trains a strictly harder task than a fixed column order (every column must be
        # predictable from every subset of the others, not just a left-to-right prefix) and was
        # found (this library's own synthetic-gmd investigation, lab notebook G64/G66) to need
        # substantially more epochs to converge on real data. Confirmed directly here too, not
        # assumed: 15 epochs left this toy a=>c relationship at an uninformative ~50/50 (the
        # extraction mechanism was fine; the model just hadn't learned anything conditional yet);
        # 300 gives a clearly directionally-correct, if not fully confident, distribution.
        epochs=300, batch_size=32, checkpoints_dir=str(tmp_path / "ckpt"), random_state=seed,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.fit(df, device=device, n_critic=0)
    return model, device


def test_predict_column_distribution_sums_to_one_and_recovers_a_known_deterministic_relationship():
    with tempfile.TemporaryDirectory() as tmp:
        model, device = _fit_toy_model(Path(tmp))
        sampler = TabularSampler.sampler_from_model(model, device=device)

        probs, values = sampler.predict_column_distribution(
            pd.DataFrame({"a": ["x", "y"]}), target_col="c", device=device
        )
        assert probs.shape == (2, len(values))
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

        p_idx, q_idx = values.index("p"), values.index("q")
        # a="x" -> c is ALWAYS "p" in training data; a="y" -> ALWAYS "q". A correctly-conditioned
        # model should clearly favor the right answer -- not necessarily be fully confident at a
        # toy-scale, moderate-epoch fit (empirically ~0.72/0.82 here, comfortably above chance/0.5
        # and clearly directionally correct both ways; not tightened further to avoid a flaky test
        # chasing full convergence, which isn't what this test is checking).
        assert probs[0, p_idx] > 0.65, f"P(c='p' | a='x') should clearly favor 'p', got {probs[0]} over {values}"
        assert probs[1, q_idx] > 0.65, f"P(c='q' | a='y') should clearly favor 'q', got {probs[1]} over {values}"


def test_predict_column_distribution_agrees_with_empirical_resampling_frequency():
    """Cross-checks the extracted distribution against a large resample generated the ORDINARY
    way (`.sample(seed_input=...)`, sampling "b" and "c" given a fixed "a"), the same
    single-forward-pass-vs-many-samples cross-check `rtf_rl.log_prob_of_sequences`'s own test
    uses -- confirms this reads off the SAME distribution the sampler actually samples from, not
    just a plausible-looking one.
    """
    with tempfile.TemporaryDirectory() as tmp:
        model, device = _fit_toy_model(Path(tmp), seed=1)
        sampler = TabularSampler.sampler_from_model(model, device=device)

        probs, values = sampler.predict_column_distribution(
            pd.DataFrame({"a": ["x"]}), target_col="c", device=device
        )
        exact = dict(zip(values, probs[0]))

        torch.manual_seed(0)
        n_samples = 4000
        # gen_batch=1, NOT n_samples: sample()'s internal num_return_sequences multiplies by
        # len(seed_input) -- gen_batch=1 means one completion per seed row (this project's own
        # documented OOM-shaped pitfall when seed_input already has one row per desired sample).
        samples = model.sample(
            n_samples=n_samples, gen_batch=1, device=device,
            seed_input=pd.DataFrame({"a": ["x"] * n_samples}),
        )
        assert (samples["a"] == "x").all()
        empirical = samples["c"].value_counts(normalize=True).to_dict()

        max_diff = max(abs(exact[v] - empirical.get(v, 0.0)) for v in values)
        assert max_diff < 0.03, (
            f"exact P(c|a='x')={exact} disagrees with empirical resampling frequency {empirical} "
            f"by {max_diff:.4f}"
        )


def test_predict_column_distribution_batches_multiple_seed_rows_independently():
    with tempfile.TemporaryDirectory() as tmp:
        model, device = _fit_toy_model(Path(tmp), seed=2)
        sampler = TabularSampler.sampler_from_model(model, device=device)

        probs, values = sampler.predict_column_distribution(
            pd.DataFrame({"a": ["x", "y", "x", "y"]}), target_col="c", device=device
        )
        p_idx, q_idx = values.index("p"), values.index("q")
        assert probs.shape[0] == 4
        # Rows with the SAME seed value must get IDENTICAL distributions (a pure function of the
        # seed, no row-to-row leakage), and different seeds must correctly diverge.
        assert np.allclose(probs[0], probs[2], atol=1e-6)
        assert np.allclose(probs[1], probs[3], atol=1e-6)
        assert probs[0, p_idx] > probs[1, p_idx]
        assert probs[1, q_idx] > probs[0, q_idx]


def test_predict_column_distribution_rejects_target_col_already_in_seed():
    with tempfile.TemporaryDirectory() as tmp:
        model, device = _fit_toy_model(Path(tmp), seed=3)
        sampler = TabularSampler.sampler_from_model(model, device=device)
        try:
            sampler.predict_column_distribution(
                pd.DataFrame({"a": ["x"], "c": ["p"]}), target_col="c", device=device
            )
            assert False, "expected an AssertionError"
        except AssertionError as e:
            assert "must not be one of seed_input" in str(e)
