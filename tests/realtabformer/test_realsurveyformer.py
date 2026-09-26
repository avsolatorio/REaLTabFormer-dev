"""Tests for realsurveyformer.py's REaLSurveyFormer -- a thin survey-specific orchestration layer
on top of REaLTabFormer2 (design agreed 2026-09-26; see the class docstring for the rationale).
Mirrors the toy schema and checks already validated ad hoc in the synthetic-gmd project's own
gmd_v2_any_order_selftest.py before this was promoted into the library.
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import GPT2Config

from realtabformer.realsurveyformer import REaLSurveyFormer
from realtabformer.realtabformer2 import REaLTabFormer2


def _toy_parent_schema(n=400, seed=0):
    """A strata-like categorical (12 values, NOT first in column order) that is a deterministic
    function of (region, urban), mirroring a real survey's strata->region/urban determinism."""
    rng = np.random.default_rng(seed)
    urban = rng.integers(0, 2, n)
    region = rng.integers(0, 3, n)
    substratum = rng.integers(0, 2, n)
    strata = (region * 4 + urban * 2 + substratum).astype(str)
    hsize = rng.integers(1, 8, n)
    weight = rng.uniform(0.5, 2.0, n)
    return pd.DataFrame({
        "urban": urban.astype(str), "region": region.astype(str),
        "hsize": hsize, "strata": strata, "weight": weight,
    })


def _tiny_model(device, **kwargs):
    cfg = dict(
        model_type="tabular", tabular_config=GPT2Config(n_layer=1, n_embd=16, n_head=2),
        epochs=5, batch_size=32,
    )
    cfg.update(kwargs)
    return REaLSurveyFormer(**cfg)


def test_defaults_any_order_on_shared_numeric_vocab_off():
    model = REaLSurveyFormer(model_type="tabular", tabular_backbone="distilgpt2")
    assert isinstance(model, REaLTabFormer2)
    assert model.any_order is True
    assert model.shared_numeric_vocab is False


def test_defaults_can_be_overridden():
    model = REaLSurveyFormer(model_type="tabular", tabular_backbone="distilgpt2", any_order=False)
    assert model.any_order is False


def test_build_pair_validator_accepts_only_known_pairs():
    validator = REaLSurveyFormer.build_pair_validator({
        ("strata", "urban"): {("0", "u0"), ("1", "u1")},
    })
    assert validator.validate(pd.Series({"strata": "0", "urban": "u0"})) is True
    assert validator.validate(pd.Series({"strata": "0", "urban": "u1"})) is False


def test_fit_weighted_assigns_token_weights_through_a_real_fit(monkeypatch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = _toy_parent_schema(n=60, seed=1)

    seen = {}
    import realtabformer.realsurveyformer as rsf_mod
    orig_add = rsf_mod._add_token_weights

    def spy_add(ds, per_row_weight):
        seen["weight"] = per_row_weight.copy()
        return orig_add(ds, per_row_weight)

    monkeypatch.setattr(rsf_mod, "_add_token_weights", spy_add)

    with tempfile.TemporaryDirectory() as d:
        model = _tiny_model(device, checkpoints_dir=str(Path(d) / "ckpt"))
        model.fit_weighted(df, weight_col="weight", device=device, n_critic=0)

    assert "weight" in seen
    expected = (df["weight"] / df["weight"].mean()).to_numpy()
    assert np.allclose(np.sort(seen["weight"]), np.sort(expected)), (
        "fit_weighted's injected per-row weights (mean-normalised) should match df['weight'], "
        "up to row-order shuffling done by the training pipeline itself"
    )


def test_fit_weighted_keeps_the_weight_column_in_the_schema_by_default():
    """A survey weight is often a real column worth synthesizing too (e.g. needed to reproduce
    population shares from the SYNTHETIC table later), so fit_weighted must not silently drop it
    from what the model learns, unless the caller explicitly asks via drop_weight_col=True."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = _toy_parent_schema(n=60, seed=4)
    with tempfile.TemporaryDirectory() as d:
        model = _tiny_model(device, checkpoints_dir=str(Path(d) / "ckpt"))
        model.fit_weighted(df, weight_col="weight", device=device, n_critic=0)
        samples = model.sample(n_samples=5, gen_batch=5, device=device)
    assert "weight" in samples.columns


def test_fit_weighted_drop_weight_col_excludes_it_from_the_schema():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = _toy_parent_schema(n=60, seed=5)
    with tempfile.TemporaryDirectory() as d:
        model = _tiny_model(device, checkpoints_dir=str(Path(d) / "ckpt"))
        model.fit_weighted(df, weight_col="weight", drop_weight_col=True, device=device, n_critic=0)
        samples = model.sample(n_samples=5, gen_batch=5, device=device)
    assert "weight" not in samples.columns


def test_fit_weighted_restores_the_patched_functions_even_if_fit_raises():
    import realtabformer.realtabformer2 as rtf2_mod
    before = (rtf2_mod.make_dataset, rtf2_mod.make_relational_dataset,
              rtf2_mod.make_dataset_with_column_types)

    df = _toy_parent_schema(n=10, seed=2)
    model = REaLSurveyFormer(model_type="tabular", tabular_backbone="distilgpt2")

    class Boom(Exception):
        pass

    def raise_fit(*a, **kw):
        raise Boom("simulated failure")

    model.fit = raise_fit
    with pytest.raises(Boom):
        model.fit_weighted(df, weight_col="weight")

    after = (rtf2_mod.make_dataset, rtf2_mod.make_relational_dataset,
              rtf2_mod.make_dataset_with_column_types)
    assert before == after, "the monkeypatched functions must be restored even when .fit() raises"


def test_sample_representative_hits_exact_apportioned_counts():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = _toy_parent_schema(n=400, seed=0)
    with tempfile.TemporaryDirectory() as d:
        model = _tiny_model(device, any_order=True, shared_numeric_vocab=False,
                             checkpoints_dir=str(Path(d) / "ckpt"))
        model.fit(df.drop(columns=["weight"]), device=device, n_critic=0)

        share = df["strata"].value_counts(normalize=True)
        n_samples = 300
        out = model.sample_representative(
            n_samples=n_samples, strata_col="strata", strata_share=share,
            device=device, gen_batch=50, seed=7,
        )
        assert len(out) == n_samples
        got_counts = out["strata"].astype(str).value_counts()
        share = share / share.sum()
        raw = share * n_samples
        counts = np.floor(raw).astype(int)
        remainder = int(n_samples - counts.sum())
        if remainder > 0:
            frac_order = (raw - counts).sort_values(ascending=False).index
            for s in frac_order[:remainder]:
                counts[s] += 1
        for stratum, expected in counts.items():
            assert got_counts.get(stratum, 0) == expected, (
                f"stratum {stratum}: expected exactly {expected}, got {got_counts.get(stratum, 0)}"
            )


def test_sample_representative_joint_seed_values_forces_consistency():
    """Mirrors the synthetic-gmd project's own G56 finding: seeding a stratum-like column ALONE
    lets the model generate its jointly-determined columns (region, urban) independently, which
    can violate their real deterministic relationship; seeding them TOGETHER fixes this by
    construction, guaranteed, not just improved."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = _toy_parent_schema(n=400, seed=0)
    with tempfile.TemporaryDirectory() as d:
        model = _tiny_model(device, any_order=True, shared_numeric_vocab=False,
                             checkpoints_dir=str(Path(d) / "ckpt"))
        model.fit(df.drop(columns=["weight"]), device=device, n_critic=0)

        share = df["strata"].value_counts(normalize=True)
        lut = df.groupby("strata")[["region", "urban"]].first()

        out = model.sample_representative(
            n_samples=200, strata_col="strata", strata_share=share,
            device=device, gen_batch=50, seed=11, joint_seed_values=lut,
        )
        merged = out.merge(lut, on=None, left_on="strata", right_index=True,
                            suffixes=("", "_expected"))
        assert (merged["region"] == merged["region_expected"]).all()
        assert (merged["urban"] == merged["urban_expected"]).all()


def test_sample_representative_raises_on_incomplete_joint_seed_values():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = _toy_parent_schema(n=100, seed=3)
    with tempfile.TemporaryDirectory() as d:
        model = _tiny_model(device, checkpoints_dir=str(Path(d) / "ckpt"))
        model.fit(df.drop(columns=["weight"]), device=device, n_critic=0)

        share = df["strata"].value_counts(normalize=True)
        lut = df.groupby("strata")[["region", "urban"]].first().iloc[:1]  # deliberately incomplete
        with pytest.raises(AssertionError, match="missing strata"):
            model.sample_representative(
                n_samples=50, strata_col="strata", strata_share=share,
                device=device, joint_seed_values=lut,
            )
