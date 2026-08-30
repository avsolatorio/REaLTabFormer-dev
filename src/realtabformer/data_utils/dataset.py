import logging
import random
import warnings
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import Dataset

from .columns import decode_processed_column
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
) -> Dict[str, Any]:
    """
    Build features for a single row at index i (works for batched inputs),
    or for the non-batched case i is ignored and values are scalars.
    """
    input_ids: List[int] = []
    token_weights: List[float] = []
    token_type_ids: List[int] = []
    label_ids: List[int] = []

    if affix_bos:
        input_ids.append(bos_id)
        label_ids.append(bos_id)
        if return_token_type_ids:
            token_type_ids.append(sptype_id)
        if field_weights is not None:
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
        if field_weights is not None:
            token_weights.append(_field_weight(k, field_weights))

    if affix_eos:
        input_ids.append(eos_id)
        label_ids.append(eos_id)
        if return_token_type_ids:
            token_type_ids.append(sptype_id)
        if field_weights is not None:
            token_weights.append(1.0)

    out: Dict[str, Any] = {"input_ids": input_ids}

    if return_label_ids:
        # copy so labels can't be mutated if input_ids changes later
        out["label_ids"] = label_ids

    if return_token_type_ids:
        out["token_type_ids"] = token_type_ids

    if field_weights is not None:
        out["token_weights"] = token_weights

    return out


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
) -> Dict[str, Any]:
    assert return_token_type_ids is False, (
        "token_type_ids not implemented in this refactor yet."
    )

    token2id = vocab["token2id"]
    col_oov = vocab["column_token_ids"]

    row_kwargs = dict(
        example=example,
        columns=columns,
        token2id=token2id,
        col_oov=col_oov,
        bos_id=token2id[SpecialTokens.BOS],
        eos_id=token2id[SpecialTokens.EOS],
        sptype_id=token2id[SpecialTokens.SPTYPE],
        mask_rate=mask_rate,
        return_label_ids=return_label_ids,
        return_token_type_ids=return_token_type_ids,
        affix_bos=affix_bos,
        affix_eos=affix_eos,
        field_weights=field_weights,
        predict_fields=predict_fields,
        batched=batched,
    )

    # --- Non-batched path: return single example dict with flat lists ---
    if not batched:
        return _build_one_row(i=0, **row_kwargs)

    # --- Batched path: example[k] is a list; return list-of-list per key ---
    # Infer batch size from the first column
    first_col = columns[0]
    B = len(example[first_col])

    rows = [_build_one_row(i, **row_kwargs) for i in range(B)]

    batched_out: Dict[str, Any] = {
        "input_ids": [r["input_ids"] for r in rows],
    }
    if return_label_ids:
        batched_out["label_ids"] = [r["label_ids"] for r in rows]
    if return_token_type_ids:
        batched_out["token_type_ids"] = [r["token_type_ids"] for r in rows]
    if field_weights is not None:
        batched_out["token_weights"] = [r["token_weights"] for r in rows]

    return batched_out


def make_dataset(
    df: pd.DataFrame,
    vocab: Dict,
    mask_rate: float = 0,
    affix_eos: bool = True,
    return_token_type_ids: bool = False,
    field_weights: Optional[Dict[str, float]] = None,
    batched: bool = True,
    batch_size: int = 2048,
    num_proc: Optional[int] = None,
    predict_fields: Optional[List[str]] = None,
) -> Dataset:
    # Load the dataframe into a HuggingFace Dataset
    training_dataset = Dataset.from_pandas(df, preserve_index=False)
    num_proc = num_proc

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
        ),
        remove_columns=training_dataset.column_names,
        num_proc=num_proc,
        batch_size=batch_size,
        batched=batched,
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
