"""REaLSurveyFormer: a thin, survey-specific orchestration layer on top of REaLTabFormer2.

Holds ONLY features that are meaningless outside survey semantics: fitting with a per-row
design/population weight, and guaranteeing exact stratum representation given known population
shares. Both are implemented by calling through to mechanisms that already exist in
REaLTabFormer2/SharedModelMixin for reasons that have nothing to do with surveys --
`token_weights`/`WeightedLabelSmoother` (rtf_trainer.py), `any_order`'s arbitrary-subset
`seed_input`, and `ObservationValidator`'s reject-and-retry sampling loop (rtf_validators.py) --
never re-implemented or hidden here.

Design agreed with the author on 2026-09-26 after the GMD (Vietnam VHLSS) household-survey
synthesis investigation (see the independent `synthetic-gmd` repo's own lab notebook, entries
G17-G59): a feature belongs in this subclass only if a non-survey tabular-data user could not
plausibly want it for an unrelated reason. `fit_weighted` and `sample_representative` below are
promoted, generalized versions of that investigation's own `gmd_survey_weight_patch.py` and
`gmd_pipeline.py`'s `_stratified_parent_sample`/`_build_parent_pair_validator` -- generalized by
taking the survey design (weights, strata shares, pair-validity sets) as plain arguments instead
of GMD's own column names and metric module.
"""
from __future__ import annotations

import contextlib
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from . import realtabformer2 as _rtf2_mod
from .realtabformer2 import REaLTabFormer2
from .rtf_validators import ObservationValidator, ValidatorBase


def _add_token_weights(ds, per_row_weight: np.ndarray):
    """Broadcasts each row's own weight uniformly across all of that row's tokens. Asserts the
    row count matches -- a silent length mismatch would misalign weights to the wrong rows, which
    is a far worse failure than not weighting at all, so this fails loudly instead.

    Uses a BATCHED `.map()` (Arrow-batch granularity, not one Python call per row) -- a per-row
    `.map()` here was measured to cost real, avoidable wall-clock time on realistic dataset sizes
    (thousands of rows) purely from HuggingFace `Dataset.map`'s per-call Python overhead, with
    nothing about the actual computation (a length-matched list broadcast) requiring row-at-a-time
    processing.
    """
    n = len(ds)
    assert n == len(per_row_weight), (
        f"row-count mismatch: dataset has {n} rows, weight array has {len(per_row_weight)} -- "
        "per-row weights would silently misalign to the wrong rows if this proceeded."
    )
    w = np.asarray(per_row_weight, dtype=float)

    def _assign_batch(batch, indices):
        batch["token_weights"] = [[float(w[i])] * len(ids) for i, ids in zip(indices, batch["input_ids"])]
        return batch

    return ds.map(_assign_batch, with_indices=True, batched=True)


@contextlib.contextmanager
def _weighted_fit(weight: pd.Series):
    """Patches `realtabformer2`'s own `make_dataset`/`make_relational_dataset`/
    `make_dataset_with_column_types` bindings for the scope of this `with` block, so any `.fit()`
    call made inside it gets a `token_weights` column reflecting `weight` (normalised to mean 1;
    the trainer's weighted-loss denominator is scale-invariant, this just keeps values in a
    familiar range). Safe because `_fit_tabular`/`_fit_relational` (rtf_shared.py/realtabformer2.py)
    call these as bare names, resolved from `realtabformer2`'s module globals AT CALL TIME, not
    import time -- reassigning the name here transparently redirects every `.fit()` call made
    while this context is active, including on subclass instances, since the inherited methods'
    `__globals__` is still `realtabformer2`'s module dict.

    `weight` must be indexed 0..N-1 in the SAME row order as the DataFrame passed to `.fit()`
    inside this block -- the caller's responsibility, since this has no way to see that DataFrame
    until `.fit()` is actually called. The row-count assertion in `_add_token_weights` catches a
    LENGTH mismatch; it cannot catch a same-length but wrongly-ordered `weight`.
    """
    w = weight.to_numpy(dtype=float)
    w = w / w.mean()

    def _make_wrapper(orig):
        def wrapped(*args, **kwargs):
            return _add_token_weights(orig(*args, **kwargs), w)
        return wrapped

    targets = [
        (_rtf2_mod, "make_dataset"),
        (_rtf2_mod, "make_relational_dataset"),
        (_rtf2_mod, "make_dataset_with_column_types"),
    ]
    originals = [(mod, name, getattr(mod, name)) for mod, name in targets]
    for mod, name, orig in originals:
        setattr(mod, name, _make_wrapper(orig))
    try:
        yield
    finally:
        for mod, name, orig in originals:
            setattr(mod, name, orig)


class REaLSurveyFormer(REaLTabFormer2):
    """A `REaLTabFormer2` for survey microdata: adds design-weighted fitting and
    exact-stratum-representation sampling, both meaningless outside survey semantics. Everything
    else -- the tabular/relational model itself, `any_order`, `seed_input`, validators, sampling --
    is unchanged `REaLTabFormer2` behavior; use those methods directly for anything general.

    Defaults `any_order=True`, `shared_numeric_vocab=False` (the library's own recommended
    combination, see `REaLTabFormer2`'s docstring): `sample_representative`'s stratified
    `seed_input` requires `any_order=True`, since seeding a non-training-order-prefix column
    otherwise doesn't work.
    """

    def __init__(self, model_type: str = "tabular", any_order: bool = True,
                 shared_numeric_vocab: bool = False, **kwargs):
        super().__init__(model_type=model_type, any_order=any_order,
                          shared_numeric_vocab=shared_numeric_vocab, **kwargs)

    def fit_weighted(self, df: pd.DataFrame, weight_col: str, drop_weight_col: bool = False, **fit_kwargs):
        """Fits with each row's contribution to the MLE loss scaled by `df[weight_col]` (e.g. a
        survey design or population weight), via the library's existing `token_weights`/
        `WeightedLabelSmoother` mechanism (rtf_trainer.py) -- this method only builds and injects
        the per-row weight column; the weighted-loss math itself is unchanged, already-tested
        library code used identically for any other reason to weight rows.

        `weight_col` stays in the data passed to `.fit()` by default (`drop_weight_col=False`): a
        survey weight is typically a real, meaningful column in its own right (e.g. needed later to
        reproduce population shares from a SYNTHETIC table, not just the real one), so this method
        does not assume it should be excluded from what the model learns to synthesize -- it is
        used BOTH as the loss weight AND as an ordinary column, unless `drop_weight_col=True`.
        """
        weight = df[weight_col].reset_index(drop=True)
        fit_df = df.drop(columns=[weight_col]) if drop_weight_col else df
        fit_df = fit_df.reset_index(drop=True)
        with _weighted_fit(weight):
            return self.fit(fit_df, **fit_kwargs)

    @staticmethod
    def build_pair_validator(pairs: Dict[Tuple[str, str], set]) -> ObservationValidator:
        """An `ObservationValidator` enforcing a set of known-deterministic `(a, b)` column pairs.

        `pairs`: `{(col_a, col_b): {(valid_a_value, valid_b_value), ...}}` -- e.g. built by
        grouping a real training set by `col_a` and collecting the (normally singleton) set of
        `col_b` values seen with it. `sample()`'s own generation loop rejects any generated row
        whose `(col_a, col_b)` is not in the corresponding pair's valid set, and retries with the
        SAME seed_input until enough valid rows are collected -- so a rejected row regenerates for
        the same seeded target, not a fresh draw from the whole population, which is what makes
        this safe to combine with `sample_representative`'s exact-count apportionment.
        """
        class _PairValidator(ValidatorBase):
            def __init__(self, valid_pairs):
                self.valid_pairs = valid_pairs

            def validate(self, a, b):
                return (str(a), str(b)) in self.valid_pairs

        ov = ObservationValidator()
        for (a, b), valid in pairs.items():
            ov.add_validator(f"{a}>{b}", _PairValidator(valid), (a, b))
        return ov

    def sample_representative(self, n_samples: int, strata_col: str, strata_share: pd.Series,
                               device: str, gen_batch: int = 1, seed: int = 1029,
                               joint_seed_values: Optional[pd.DataFrame] = None,
                               validator: Optional[ObservationValidator] = None) -> pd.DataFrame:
        """Samples `n_samples` rows with each `strata_col` value's COUNT fixed to its population
        share, rather than left to the model's own learned marginal.

        `strata_share`: a `Series` indexed by stratum value, giving each stratum's population
        share (need not already sum to 1 -- renormalised here). Per-stratum counts are apportioned
        by the largest-remainder method (Hamilton apportionment) so they sum to EXACTLY
        `n_samples`, not just approximately.

        `joint_seed_values` (default None): an optional `DataFrame` indexed by stratum value,
        giving additional column(s) to seed ALONGSIDE `strata_col` for every row of that stratum
        (e.g. columns that are deterministic functions of the stratum in the training data).
        `any_order`'s arbitrary-subset `seed_input` lets these be seeded TOGETHER with
        `strata_col`, fixing their relationship to it by construction rather than leaving the
        model to infer it from an out-of-context seed. Every stratum in `strata_share` must have a
        row in `joint_seed_values` -- callers should verify beforehand (e.g. groupby-nunique on
        their own training data) that each such column is genuinely a deterministic function of
        the stratum; this method does not check that for them, since it only sees the lookup
        table, not the original per-row data it was built from.

        `validator` (default None): see `build_pair_validator`. `sample()`'s own generation loop
        retries with the SAME seed_input when a row is rejected, so a rejected row regenerates for
        the same target stratum, not the whole population -- trading a small amount of exact-count
        precision (the final internal sampling step draws a random subsample from however many
        valid rows accumulated across retries) for eliminating validator violations.

        Builds ONE combined `seed_input` (one row per target sample, stratum values repeated per
        their apportioned count), chunked into `gen_batch`-sized calls with `gen_batch=1` passed to
        `.sample()` itself -- NOT one `.sample()` call per stratum. `sample()`'s internal
        `num_return_sequences` scales with `len(seed_input) * gen_batch`, so `gen_batch=1` here
        means one completion per seed row, and the actual per-call batch size is controlled by the
        chunk size instead of a second, multiplicative factor.
        """
        share = strata_share / strata_share.sum()
        raw = share * n_samples
        counts = np.floor(raw).astype(int)
        remainder = int(n_samples - counts.sum())
        if remainder > 0:
            frac_order = (raw - counts).sort_values(ascending=False).index
            for s in frac_order[:remainder]:
                counts[s] += 1
        assert counts.sum() == n_samples, (
            f"apportionment bug: counts sum to {counts.sum()}, expected {n_samples}"
        )

        if joint_seed_values is not None:
            missing = set(counts.index) - set(joint_seed_values.index)
            assert not missing, f"joint_seed_values is missing strata: {sorted(missing)}"

        seed_values = np.repeat(counts.index.to_numpy(), counts.to_numpy())
        np.random.default_rng(seed).shuffle(seed_values)  # avoid gen_batch-sized chunks of one stratum

        parts = []
        for start in range(0, n_samples, gen_batch):
            chunk = seed_values[start:start + gen_batch]
            seed_df = pd.DataFrame({strata_col: chunk})
            if joint_seed_values is not None:
                seed_df = pd.concat(
                    [seed_df, joint_seed_values.loc[chunk].reset_index(drop=True)], axis=1
                )
            sp = self.sample(n_samples=len(chunk), gen_batch=1, device=device,
                              seed_input=seed_df, validator=validator)
            parts.append(sp)
        out = pd.concat(parts, ignore_index=True)
        return out.sample(frac=1, random_state=seed).reset_index(drop=True)  # un-block the chunk ordering
