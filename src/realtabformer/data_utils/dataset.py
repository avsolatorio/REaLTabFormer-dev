import logging
import random
import warnings
from typing import Dict, List, Optional

import pandas as pd
from datasets import Dataset

from .columns import extract_processed_column
from .constants import SpecialTokens


def get_token_id(
    token: str, vocab_token2id: Dict[str, int], mask_rate: float = 0
) -> int:
    token_id = vocab_token2id.get(token, vocab_token2id[SpecialTokens.UNK])
    if mask_rate > 0:
        token_id = (
            vocab_token2id[SpecialTokens.RMASK]
            if random.random() < mask_rate
            else token_id
        )

    return token_id


def get_input_ids(
    example,
    vocab: Dict,
    columns: List,
    mask_rate: float = 0,
    return_label_ids: Optional[bool] = True,
    return_token_type_ids: Optional[bool] = False,
    affix_bos: Optional[bool] = True,
    affix_eos: Optional[bool] = True,
) -> Dict:
    # Raise an assertion error while the implementation
    # is not yet ready.
    assert return_token_type_ids is False
    input_ids: List[int] = []
    token_type_ids: List[int] = []

    if affix_bos:
        input_ids.append(vocab["token2id"][SpecialTokens.BOS])
        if return_token_type_ids:
            token_type_ids.append(vocab["token2id"][SpecialTokens.SPTYPE])

    for k in columns:
        input_ids.append(get_token_id(example[k], vocab["token2id"], mask_rate))
        if return_token_type_ids:
            col_name = extract_processed_column(k)
            token_type_ids.append(vocab["token2id"][col_name])

    if affix_eos:
        input_ids.append(vocab["token2id"][SpecialTokens.EOS])
        if return_token_type_ids:
            token_type_ids.append(vocab["token2id"][SpecialTokens.SPTYPE])

    data = dict(input_ids=input_ids)

    if return_label_ids:
        data["label_ids"] = input_ids

    if return_token_type_ids:
        data["token_type_ids"] = token_type_ids

    return data


def make_dataset(
    df: pd.DataFrame,
    vocab: Dict,
    mask_rate: float = 0,
    affix_eos: bool = True,
    return_token_type_ids: bool = False,
) -> Dataset:
    # Load the dataframe into a HuggingFace Dataset
    training_dataset = Dataset.from_pandas(df, preserve_index=False)

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
        ),
        remove_columns=training_dataset.column_names,
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
