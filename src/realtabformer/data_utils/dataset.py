import hashlib
import logging
import random
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from datasets import Dataset

from .columns import (
    decode_column_values,
    decode_processed_column,
    is_datetime_col,
    is_numeric_col,
)
from .constants import SpecialTokens


def get_token_id(
    token: str,
    vocab_token2id: Dict[str, int],
    oov_options: List[int],
    mask_rate: float = 0,
) -> int:
    if token in vocab_token2id:
        token_id = vocab_token2id[token]
    elif oov_options:
        token_id = random.choice(oov_options)
    else:
        token_id = vocab_token2id[SpecialTokens.UNK]

    if mask_rate > 0:
        token_id = (
            vocab_token2id[SpecialTokens.RMASK]
            if random.random() < mask_rate
            else token_id
        )

    return token_id


def _field_weight(col_name: str, field_weights: Optional[Dict[str, float]]) -> float:
    if field_weights is None:
        return 1.0
    for wk, wv in field_weights.items():
        if col_name.startswith(wk):
            return float(wv)
    return 1.0


def _combined_token_weight(
    col_name: str,
    field_weights: Optional[Dict[str, float]],
    chunk_significance_weights: Optional[Dict[str, float]],
) -> float:
    """`field_weights` (user-set, per-*original*-column importance) and
    `chunk_significance_weights` (data_utils.vocab, auto-computed,
    per-*processed*-column digit-chunk reallocation) are orthogonal and
    compose multiplicatively -- see `compute_chunk_significance_weights`'s
    docstring for why they don't conflict: the latter only reallocates
    weight *within* one original column's own chunks, it never changes
    that column's total budget.
    """
    weight = _field_weight(col_name, field_weights)
    if chunk_significance_weights is not None:
        weight *= chunk_significance_weights.get(col_name, 1.0)
    return weight


def _is_predict_field(col_name: str, predict_fields: Optional[List[str]]) -> bool:
    if predict_fields is None:
        # If no predict fields are specified, all fields are considered as predict fields for prediction.
        return True
    for pf in predict_fields:
        if col_name.startswith(pf):
            return True
    return False


def _build_one_row(
    i: int,
    example: Dict[str, Any],
    columns: List[str],
    token2id: Dict[str, int],
    col_oov: Dict[str, List[int]],
    bos_id: int,
    eos_id: int,
    sptype_id: int,
    mask_rate: float,
    return_label_ids: bool,
    return_token_type_ids: bool,
    affix_bos: bool,
    affix_eos: bool,
    field_weights: Optional[Dict[str, float]],
    predict_fields: Optional[List[str]],
    batched: bool,
    chunk_significance_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Build features for a single row at index i (works for batched inputs),
    or for the non-batched case i is ignored and values are scalars.
    """
    input_ids: List[int] = []
    token_weights: List[float] = []
    token_type_ids: List[int] = []
    label_ids: List[int] = []

    has_weights = field_weights is not None or chunk_significance_weights is not None

    if affix_bos:
        input_ids.append(bos_id)
        label_ids.append(bos_id)
        if return_token_type_ids:
            token_type_ids.append(sptype_id)
        if has_weights:
            token_weights.append(1.0)

    for k in columns:
        val = example[k][i] if batched else example[k]
        tid = get_token_id(
            val,
            token2id,
            oov_options=col_oov[k],
            mask_rate=mask_rate,
        )
        input_ids.append(tid)

        if _is_predict_field(k, predict_fields):
            label_ids.append(tid)
        else:
            label_ids.append(-100)

        if return_token_type_ids:
            col_name = decode_processed_column(k)
            token_type_ids.append(token2id[col_name])
        if has_weights:
            token_weights.append(
                _combined_token_weight(k, field_weights, chunk_significance_weights)
            )

    if affix_eos:
        input_ids.append(eos_id)
        label_ids.append(eos_id)
        if return_token_type_ids:
            token_type_ids.append(sptype_id)
        if has_weights:
            token_weights.append(1.0)

    out: Dict[str, Any] = {"input_ids": input_ids}

    if return_label_ids:
        # copy so labels can't be mutated if input_ids changes later
        out["label_ids"] = label_ids

    if return_token_type_ids:
        out["token_type_ids"] = token_type_ids

    if has_weights:
        out["token_weights"] = token_weights

    return out


def _vectorized_column_token_ids(
    values: List[Any],
    token2id: Dict[str, int],
    oov_options: List[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Map one column's batch of raw values to token ids, vectorized.

    Equivalent to calling `get_token_id(v, token2id, oov_options,
    mask_rate=0)` for each `v` in `values`, but `pandas.Series.map` with a
    dict argument dispatches to a C-level hashtable lookup instead of a
    Python function call per element, and the OOV fallback draws one
    batched `rng.choice` instead of one `random.choice` per OOV cell.
    (`mask_rate` masking is applied afterward, across the whole batch at
    once -- see `_build_batch`.)
    """
    mapped = pd.Series(values).map(token2id)
    na_mask = mapped.isna()

    if not na_mask.any():
        return mapped.to_numpy(dtype=np.int64)

    if oov_options:
        fallback = rng.choice(oov_options, size=int(na_mask.sum()))
        filled = mapped.to_numpy(dtype="float64", copy=True)
        filled[na_mask.to_numpy()] = fallback
        return filled.astype(np.int64)

    return mapped.fillna(token2id[SpecialTokens.UNK]).to_numpy(dtype=np.int64)


def _build_batch(
    example: Dict[str, Any],
    columns: List[str],
    token2id: Dict[str, int],
    col_oov: Dict[str, List[int]],
    bos_id: int,
    eos_id: int,
    mask_rate: float,
    return_label_ids: bool,
    affix_bos: bool,
    affix_eos: bool,
    field_weights: Optional[Dict[str, float]],
    predict_fields: Optional[List[str]],
    rng: np.random.Generator,
    chunk_significance_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Vectorized equivalent of calling `_build_one_row` once per row in
    the batch. Builds the whole batch's `input_ids`/`label_ids`/
    `token_weights` as 2D numpy arrays via per-column vectorized
    operations, instead of a Python loop over rows x columns.
    """
    n_cols = len(columns)
    batch_size = len(example[columns[0]])

    id_cols = np.empty((batch_size, n_cols), dtype=np.int64)
    for j, k in enumerate(columns):
        id_cols[:, j] = _vectorized_column_token_ids(
            example[k], token2id, col_oov[k], rng
        )

    if mask_rate > 0:
        rmask_id = token2id[SpecialTokens.RMASK]
        mask_draw = rng.random(size=(batch_size, n_cols)) < mask_rate
        id_cols = np.where(mask_draw, rmask_id, id_cols)

    def _affix(cols: np.ndarray, bos_val, eos_val) -> np.ndarray:
        parts = [cols]
        if affix_bos:
            parts.insert(0, np.full((batch_size, 1), bos_val, dtype=cols.dtype))
        if affix_eos:
            parts.append(np.full((batch_size, 1), eos_val, dtype=cols.dtype))
        return np.hstack(parts)

    out: Dict[str, Any] = {"input_ids": _affix(id_cols, bos_id, eos_id).tolist()}

    if return_label_ids:
        predict_mask = np.array(
            [_is_predict_field(k, predict_fields) for k in columns]
        )
        label_cols = np.where(predict_mask[None, :], id_cols, -100)
        out["label_ids"] = _affix(label_cols, bos_id, eos_id).tolist()

    if field_weights is not None or chunk_significance_weights is not None:
        weights = np.array(
            [
                _combined_token_weight(k, field_weights, chunk_significance_weights)
                for k in columns
            ],
            dtype=np.float64,
        )
        weight_cols = np.broadcast_to(weights, (batch_size, n_cols))
        out["token_weights"] = _affix(weight_cols, 1.0, 1.0).tolist()

    return out


def _build_batch_with_column_types(
    example: Dict[str, Any],
    columns: List[str],
    token2id: Dict[str, int],
    col_oov: Dict[str, List[int]],
    column_type_ids: Dict[str, int],
    numeric_like_columns: set,
    bos_id: int,
    eos_id: int,
    sptype_id: int,
    mask_rate: float,
    return_label_ids: bool,
    affix_bos: bool,
    affix_eos: bool,
    field_weights: Optional[Dict[str, float]],
    predict_fields: Optional[List[str]],
    rng: np.random.Generator,
    chunk_significance_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """REaLTabFormerV2-only variant of `_build_batch` that additionally
    emits `token_type_ids`, one id per column representing that column's
    *original* (pre-partition) identity (see `vocab.build_pooled_vocab`),
    broadcast across rows the same way `token_weights` already is --
    O(columns) work, not O(rows x columns), so this doesn't reintroduce
    the per-row Python loop the vectorization work removed.

    Reuses `_vectorized_column_token_ids`/`_is_predict_field`/
    `_combined_token_weight` unchanged. `_build_batch` (v1's function, used by
    `get_input_ids`/`make_dataset`) is not modified or called here --
    this is a parallel, standalone implementation so v1's tested path
    stays exactly as it is.

    `column_type_ids`'s vocab comes from `build_pooled_vocab`, whose
    numeric/datetime-like columns share one *pooled* value range keyed on
    the column-prefix-*stripped* raw value (see that function's
    docstring) -- so, unlike `_build_batch`, the raw values for those
    columns must be de-prefixed (`decode_column_values`) before the
    token2id lookup, or every value would still be a distinct string per
    column by construction and pooling would be a no-op.
    """
    n_cols = len(columns)
    batch_size = len(example[columns[0]])

    id_cols = np.empty((batch_size, n_cols), dtype=np.int64)
    type_cols = np.empty((batch_size, n_cols), dtype=np.int64)
    for j, k in enumerate(columns):
        raw_values = example[k]
        if k in numeric_like_columns:
            # Left as a Series (not `.tolist()`'d back to a plain list):
            # `_vectorized_column_token_ids` just wraps its `values` arg in
            # `pd.Series(values)` right away, so handing it an existing
            # Series skips a redundant list-materialize + Series-rebuild
            # round trip.
            raw_values = decode_column_values(pd.Series(raw_values))
        id_cols[:, j] = _vectorized_column_token_ids(
            raw_values, token2id, col_oov[k], rng
        )
        type_cols[:, j] = column_type_ids[k]

    if mask_rate > 0:
        rmask_id = token2id[SpecialTokens.RMASK]
        mask_draw = rng.random(size=(batch_size, n_cols)) < mask_rate
        id_cols = np.where(mask_draw, rmask_id, id_cols)
        # `type_cols` is untouched by masking: a masked position still
        # belongs to its column, it just doesn't reveal its value.

    def _affix(cols: np.ndarray, bos_val, eos_val) -> np.ndarray:
        parts = [cols]
        if affix_bos:
            parts.insert(0, np.full((batch_size, 1), bos_val, dtype=cols.dtype))
        if affix_eos:
            parts.append(np.full((batch_size, 1), eos_val, dtype=cols.dtype))
        return np.hstack(parts)

    out: Dict[str, Any] = {"input_ids": _affix(id_cols, bos_id, eos_id).tolist()}
    # BOS/EOS get the dedicated [SPTYPE] structural marker as their type
    # id, matching the convention already present (but previously unused)
    # in `_build_one_row`.
    out["token_type_ids"] = _affix(type_cols, sptype_id, sptype_id).tolist()

    if return_label_ids:
        predict_mask = np.array(
            [_is_predict_field(k, predict_fields) for k in columns]
        )
        label_cols = np.where(predict_mask[None, :], id_cols, -100)
        out["label_ids"] = _affix(label_cols, bos_id, eos_id).tolist()

    if field_weights is not None or chunk_significance_weights is not None:
        weights = np.array(
            [
                _combined_token_weight(k, field_weights, chunk_significance_weights)
                for k in columns
            ],
            dtype=np.float64,
        )
        weight_cols = np.broadcast_to(weights, (batch_size, n_cols))
        out["token_weights"] = _affix(weight_cols, 1.0, 1.0).tolist()

    return out


def make_dataset_with_column_types(
    df: pd.DataFrame,
    vocab: Dict,
    mask_rate: float = 0,
    affix_eos: bool = True,
    field_weights: Optional[Dict[str, float]] = None,
    batch_size: int = 32768,
    num_proc: Optional[int] = None,
    predict_fields: Optional[List[str]] = None,
    seed: Optional[int] = None,
    keep_in_memory: bool = True,
    chunk_significance_weights: Optional[Dict[str, float]] = None,
) -> Dataset:
    """REaLTabFormerV2-only counterpart of `make_dataset`: builds a
    dataset with an additional `token_type_ids` column carrying each
    position's column-identity id (see `_build_batch_with_column_types`
    and `vocab.build_pooled_vocab`). `vocab` must come from
    `build_pooled_vocab`, not `build_vocab` -- it needs the
    `column_type_ids` key this function reads below.

    `make_dataset`/`get_input_ids`/`_build_batch` (v1's functions) are
    not modified or called here.
    """
    training_dataset = Dataset.from_pandas(df, preserve_index=False)

    columns = list(df.columns)
    token2id = vocab["token2id"]
    col_oov = vocab["column_token_ids"]
    column_type_ids = vocab["column_type_ids"]
    numeric_like_columns = {
        c for c in columns if is_numeric_col(c) or is_datetime_col(c)
    }
    bos_id = token2id[SpecialTokens.BOS]
    eos_id = token2id[SpecialTokens.EOS]
    sptype_id = token2id[SpecialTokens.SPTYPE]

    # Same rationale as `make_dataset`: a local Generator, independent of
    # the legacy global `random`/`np.random` state, created once and
    # shared (via closure) across every batch for one deterministic
    # sequence over the whole dataset given a `seed`.
    rng = np.random.default_rng(seed)

    new_fingerprint = _cheap_map_fingerprint(
        training_dataset,
        mask_rate=mask_rate,
        affix_eos=affix_eos,
        field_weights=field_weights,
        predict_fields=predict_fields,
        seed=seed,
        variant="column_types",
        chunk_significance_weights=chunk_significance_weights,
    )

    logging.info("Creating the input_ids/label_ids/token_type_ids columns...")

    return training_dataset.map(
        lambda example: _build_batch_with_column_types(
            example,
            columns,
            token2id,
            col_oov,
            column_type_ids,
            numeric_like_columns,
            bos_id,
            eos_id,
            sptype_id,
            mask_rate,
            True,  # return_label_ids
            True,  # affix_bos
            affix_eos,
            field_weights,
            predict_fields,
            rng,
            chunk_significance_weights,
        ),
        remove_columns=training_dataset.column_names,
        num_proc=num_proc,
        batch_size=batch_size,
        batched=True,
        new_fingerprint=new_fingerprint,
        keep_in_memory=keep_in_memory,
    )


def get_input_ids(
    example: Dict[str, Any],
    vocab: Dict,
    columns: List[str],
    mask_rate: float = 0.0,
    return_label_ids: bool = True,
    return_token_type_ids: bool = False,
    affix_bos: bool = True,
    affix_eos: bool = True,
    field_weights: Optional[Dict[str, float]] = None,
    predict_fields: Optional[List[str]] = None,
    batched: bool = False,
    rng: Optional[np.random.Generator] = None,
    chunk_significance_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    assert return_token_type_ids is False, (
        "token_type_ids not implemented in this refactor yet."
    )

    token2id = vocab["token2id"]
    col_oov = vocab["column_token_ids"]
    bos_id = token2id[SpecialTokens.BOS]
    eos_id = token2id[SpecialTokens.EOS]

    # --- Non-batched path: single row, vectorization buys nothing here ---
    if not batched:
        return _build_one_row(
            i=0,
            example=example,
            columns=columns,
            token2id=token2id,
            col_oov=col_oov,
            bos_id=bos_id,
            eos_id=eos_id,
            sptype_id=token2id[SpecialTokens.SPTYPE],
            mask_rate=mask_rate,
            return_label_ids=return_label_ids,
            return_token_type_ids=return_token_type_ids,
            affix_bos=affix_bos,
            affix_eos=affix_eos,
            field_weights=field_weights,
            predict_fields=predict_fields,
            batched=batched,
            chunk_significance_weights=chunk_significance_weights,
        )

    # --- Batched path: vectorized, see _build_batch ---
    return _build_batch(
        example=example,
        columns=columns,
        token2id=token2id,
        col_oov=col_oov,
        bos_id=bos_id,
        eos_id=eos_id,
        mask_rate=mask_rate,
        return_label_ids=return_label_ids,
        affix_bos=affix_bos,
        affix_eos=affix_eos,
        field_weights=field_weights,
        predict_fields=predict_fields,
        rng=rng if rng is not None else np.random.default_rng(),
        chunk_significance_weights=chunk_significance_weights,
    )


def _cheap_map_fingerprint(training_dataset: Dataset, **transform_args: Any) -> str:
    # `Dataset.map()` otherwise computes its own cache fingerprint by
    # dill-pickling the *entire* transform closure (here: `vocab`, `rng`,
    # etc.) -- for closures like this one that turned out to be
    # pathologically slow (confirmed: dominated wall-clock time, worse
    # than the actual vectorized work, and scaling with row count since
    # it's invoked once per internal batch/writer-flush, not once per
    # `.map()` call). We don't rely on `datasets`' on-disk cache here, so
    # compute a cheap fingerprint ourselves from just the actual varying
    # inputs (the input dataset's own fingerprint plus the transform
    # arguments) instead of letting `datasets` hash the whole closure.
    payload = repr((training_dataset._fingerprint, transform_args))
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def make_dataset(
    df: pd.DataFrame,
    vocab: Dict,
    mask_rate: float = 0,
    affix_eos: bool = True,
    return_token_type_ids: bool = False,
    field_weights: Optional[Dict[str, float]] = None,
    batched: bool = True,
    batch_size: int = 32768,
    num_proc: Optional[int] = None,
    predict_fields: Optional[List[str]] = None,
    seed: Optional[int] = None,
    keep_in_memory: bool = True,
    chunk_significance_weights: Optional[Dict[str, float]] = None,
) -> Dataset:
    # Load the dataframe into a HuggingFace Dataset
    training_dataset = Dataset.from_pandas(df, preserve_index=False)
    num_proc = num_proc

    # A single Generator, created once and shared across every batch (via
    # closure) rather than re-seeded per batch, so the draws across the
    # whole dataset form one deterministic sequence for a given `seed`
    # instead of different batches repeating the same values. This is a
    # local object, independent of the legacy global `random`/`np.random`
    # state the rest of training uses (sensitivity computation, model
    # init, etc. -- see realtabformer.py's `random.seed`/`np.random.seed`
    # calls) -- it neither reads nor perturbs that state.
    rng = np.random.default_rng(seed)

    new_fingerprint = _cheap_map_fingerprint(
        training_dataset,
        mask_rate=mask_rate,
        affix_eos=affix_eos,
        return_token_type_ids=return_token_type_ids,
        field_weights=field_weights,
        batched=batched,
        predict_fields=predict_fields,
        seed=seed,
        chunk_significance_weights=chunk_significance_weights,
    )

    # Create the input_ids and label_ids columns
    logging.info("Creating the input_ids and label_ids columns...")

    return training_dataset.map(
        lambda example: get_input_ids(
            example,
            vocab,
            df.columns,
            mask_rate=mask_rate,
            affix_eos=affix_eos,
            return_token_type_ids=return_token_type_ids,
            field_weights=field_weights,
            batched=batched,
            predict_fields=predict_fields,
            rng=rng,
            chunk_significance_weights=chunk_significance_weights,
        ),
        remove_columns=training_dataset.column_names,
        num_proc=num_proc,
        batch_size=batch_size,
        batched=batched,
        new_fingerprint=new_fingerprint,
        # By the time execution reaches here, `df` is already a fully
        # in-memory pandas DataFrame (process_data eagerly copies/
        # transforms it upstream) -- there's no larger-than-memory/
        # out-of-core scenario for this library to preserve, so building
        # the mapped output in memory instead of a memory-mapped disk
        # cache file avoids real disk I/O that otherwise dominates
        # make_dataset's wall-clock time (confirmed via profiling).
        keep_in_memory=keep_in_memory,
    )


def get_relational_input_ids(
    example,
    input_idx,
    vocab,
    columns,
    output_dataset,
    in_out_idx,
    output_max_length: Optional[int] = None,
    return_token_type_ids: bool = False,
) -> dict:
    # Start with 2 to take into account the [BOS] and [EOS] tokens
    sequence_len = 2

    # Build the input_ids for the encoder
    input_payload = get_input_ids(
        example,
        vocab["encoder"],
        columns,
        return_label_ids=False,
        return_token_type_ids=return_token_type_ids,
        affix_bos=True,
        affix_eos=True,
    )
    input_ids = input_payload["input_ids"]
    token_type_ids = input_payload.get("token_type_ids")

    # Build the label_ids for the decoder
    output_idx = in_out_idx[input_idx]

    valid = True

    label_ids = [vocab["decoder"]["token2id"][SpecialTokens.BOS]]
    if len(output_idx) > 0:
        for ids in output_dataset.select(output_idx)["input_ids"]:
            # Pad each observation with the [BMEM] and [EMEM] tokens

            tmp_label_ids = [vocab["decoder"]["token2id"][SpecialTokens.BMEM]]
            tmp_label_ids.extend(ids)
            tmp_label_ids.append(vocab["decoder"]["token2id"][SpecialTokens.EMEM])

            if output_max_length:
                if (sequence_len + len(tmp_label_ids)) > output_max_length:
                    # This exceeds the expected limit.
                    # Drop this observation.
                    valid = False
                    break

            label_ids.extend(tmp_label_ids)
            sequence_len += len(tmp_label_ids)

    label_ids.append(vocab["decoder"]["token2id"][SpecialTokens.EOS])

    payload = dict(
        input_ids=input_ids,
        # The variable `labels` is used in the EncoderDecoder model
        # instead of `label_ids`.
        labels=label_ids if valid else None,
    )

    if token_type_ids is not None:
        payload["token_type_ids"] = token_type_ids

    return payload


def make_relational_dataset(
    in_df: pd.DataFrame,
    out_df: pd.DataFrame,
    vocab: dict,
    in_out_idx: dict,
    mask_rate=0,
    output_max_length: Optional[int] = None,
    return_token_type_ids: bool = False,
) -> Dataset:
    # Relational data
    # Load the dataframe into a HuggingFace Dataset
    encoder_dataset = Dataset.from_pandas(in_df, preserve_index=False)

    # Load the dataframe into a HuggingFace Dataset
    decoder_dataset = Dataset.from_pandas(out_df, preserve_index=False)
    # Do not add [BOS] and [EOS] here. This will be handled
    # in the creation of the training_dataset in `get_relational_input_ids`.
    decoder_dataset = decoder_dataset.map(
        lambda example: get_input_ids(
            example,
            vocab["decoder"],
            out_df.columns,
            mask_rate=mask_rate,
            return_label_ids=False,
            return_token_type_ids=return_token_type_ids,
            affix_bos=False,
            affix_eos=False,
        ),
        remove_columns=decoder_dataset.column_names,
    )

    training_dataset = encoder_dataset.map(
        lambda example, idx: get_relational_input_ids(
            example,
            idx,
            vocab,
            in_df.columns,
            decoder_dataset,
            in_out_idx,
            output_max_length,
        ),
        remove_columns=encoder_dataset.column_names,
        with_indices=True,
    )

    # If the output_max_length variable is specified, filter
    # observations that exceed this length. The
    # `get_relational_input_ids` should have set the
    # `labels` to None if the output exceeds `output_max_length`.
    if output_max_length:
        init_data_length = training_dataset.shape[0]

        training_dataset = training_dataset.filter(
            lambda example: example["labels"] is not None
        )

        removed_count = init_data_length - training_dataset.shape[0]
        if removed_count > 0:
            warnings.warn(
                f"A total of {removed_count} out of {init_data_length} has been removed from the training data because they exceeded the `output_max_length` of {output_max_length}."
            )

    return training_dataset
