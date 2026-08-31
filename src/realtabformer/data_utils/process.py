from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from .columns import encode_column_values, encode_processed_column
from .constants import TEACHER_FORCING_PRE, ColDataType
from .transform import (
    process_categorical_data,
    process_datetime_data,
    process_numeric_data,
    tokenize_numeric_col,
)


def _coerce_numeric_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    # Force cast integral values to Int64Dtype dtype
    # to save precision if they are represented as float.
    for c in df:
        try:
            if pd.api.types.is_datetime64_any_dtype(df[c].dtype):
                # Don't cast datetime types.
                continue

            if pd.api.types.is_numeric_dtype(df[c].dtype):
                # Only cast if the column is explicitly numeric type.
                df[c] = df[c].astype(pd.Int64Dtype())
        except TypeError:
            pass
        except ValueError:
            pass

    return df


def _inject_teacher_forcing_column(
    df: pd.DataFrame, target_col: str, first_col_type
) -> pd.DataFrame:
    assert first_col_type is None, (
        "Implicit ordering of columns when teacher-forcing of target is used is not supported yet!"
    )
    tf_col_name = f"{TEACHER_FORCING_PRE}_{target_col}"
    assert tf_col_name not in df.columns, (
        f"The column name ({tf_col_name}) must not be in the raw data. Found instead..."
    )

    target_ser = df[target_col].copy()
    target_ser.name = tf_col_name
    df = pd.concat([target_ser, df], axis=1)

    return df


def _compute_column_index_prefixes(
    df: pd.DataFrame, col_transform_data: Dict
) -> Dict[str, str]:
    # Rename the columns to encode the original order by adding a suffix of
    # increasing integer values. The zero-pad width (`num_cols`) is
    # persisted in `col_transform_data` under `num_col_key` so it stays
    # stable across a later call on the same fitted transform data with
    # fewer columns (e.g. partial/seed-input processing), rather than being
    # recomputed from whatever subset of columns is present this time.
    num_col_key = "$%NUM_COLS%$"

    assert num_col_key not in df.columns, (
        f"The column name ({num_col_key}) must not be in the raw data. Found instead..."
    )
    col_transform_data[num_col_key] = num_cols = col_transform_data.get(
        num_col_key, len(str(len(df.columns)))
    )

    return {col: f"{str(i).zfill(num_cols)}" for i, col in enumerate(df.columns)}


def _split_numeric_categorical_cols(
    df: pd.DataFrame,
    numeric_cols: pd.Index,
    col_transform_data: Dict,
    numeric_categorical_threshold: int,
) -> pd.Index:
    """Of `numeric_cols` (dtype-numeric columns), demote the ones that
    should be tokenized as a single categorical value instead of
    digit-chunked -- returns the subset that stays on the numeric
    pipeline. A numeric column with only `K` distinct values has a hard
    information ceiling of `log2(K)` bits; digit-chunking it still spends
    several autoregressive generation steps (and gives the
    entropy-weighting heuristic several *marginally* plausible-looking
    but actually redundant positions -- see compute_chunk_significance_weights)
    to represent something a single categorical token already captures
    exactly, in one generation step.

    The decision is made *once*, at first encounter, and frozen into
    `col_transform_data[c]["treated_as_categorical"]` -- mirroring the
    same "does `col_transform_data.get(c)` already exist?" first-time
    check `_process_typed_columns` uses for its own transform data -- so
    that later calls on a different (often much smaller -- a seed_input
    can be a single row) slice of the same column always replay the
    frozen decision instead of recomputing cardinality on data too small
    for that to mean anything.
    """
    kept = []
    for c in numeric_cols:
        existing = col_transform_data.get(c)
        if existing is not None:
            if existing.get("treated_as_categorical", False):
                continue
            kept.append(c)
            continue

        if (
            numeric_categorical_threshold is not None
            and df[c].nunique() <= numeric_categorical_threshold
        ):
            col_transform_data[c] = {"treated_as_categorical": True}
            continue

        kept.append(c)

    return pd.Index(kept)


def _process_typed_columns(
    df: pd.DataFrame,
    cols,
    col_idx: Dict[str, str],
    dtype_tag: str,
    processor: Callable[[pd.Series, Dict], Tuple[pd.Series, Dict]],
    col_transform_data: Dict,
    numeric_nparts: int,
    col_name_to_transform_data: Dict[str, Dict],
    processed_series: List[pd.Series],
    orig_to_processed_col_map: Dict[str, str],
) -> None:
    # Shared by the numeric and datetime column loops in `process_data`: both
    # do the same steps below, differing only in `dtype_tag` and which
    # processor is called. Mutates `col_transform_data`,
    # `col_name_to_transform_data`, `processed_series`, and
    # `orig_to_processed_col_map` in place, matching the original inline
    # loops.
    for c in cols:
        col_name = encode_processed_column(col_idx[c], dtype_tag, c)
        orig_to_processed_col_map[c] = col_name

        _col_transform_data = col_transform_data.get(c)
        series, transform_data = processor(df[c], _col_transform_data)
        if _col_transform_data is None:
            # This means that no transform data is available
            # before the processing.
            transform_data["numeric_nparts"] = numeric_nparts
            col_transform_data[c] = transform_data
        series.name = col_name
        col_name_to_transform_data[col_name] = transform_data
        processed_series.append(series)


def _tokenize_numeric_like_columns(
    processed_df: pd.DataFrame,
    col_name_to_transform_data: Dict[str, Dict],
    numeric_nparts: int,
) -> pd.DataFrame:
    if processed_df.empty:
        return processed_df

    # Combine the processed numeric and datetime data.
    return pd.concat(
        [
            tokenize_numeric_col(
                processed_df[col],
                # During inference this deliberately reads the nparts value
                # saved at training time for this column, not the outer
                # `numeric_nparts` argument.
                nparts=col_name_to_transform_data[col].get(
                    "numeric_nparts", numeric_nparts
                ),
            )
            for col in processed_df.columns
        ],
        axis=1,
    )


def _process_categorical_columns(
    df: pd.DataFrame,
    processed_df: pd.DataFrame,
    categorical_cols,
    col_idx: Dict[str, str],
    orig_to_processed_col_map: Dict[str, str],
) -> pd.DataFrame:
    if categorical_cols.empty:
        return processed_df

    # Process the rest of the data, assumed to be categorical values.
    for c in categorical_cols:
        orig_to_processed_col_map[c] = encode_processed_column(
            col_idx[c], ColDataType.CATEGORICAL, c
        )

    return pd.concat(
        [
            processed_df,
            *(
                process_categorical_data(df[c]).rename(orig_to_processed_col_map[c])
                for c in categorical_cols
            ),
        ],
        axis=1,
    )


def _reorder_columns_by_first_col_type(
    processed_df: pd.DataFrame, first_col_type
) -> pd.DataFrame:
    # Get the different sets of column types
    cat_cols = processed_df.columns[
        processed_df.columns.str.contains(ColDataType.CATEGORICAL)
    ]
    numeric_like_cols = processed_df.columns[
        ~processed_df.columns.str.contains(ColDataType.CATEGORICAL)
    ]

    if first_col_type == ColDataType.CATEGORICAL:
        return processed_df[cat_cols.union(numeric_like_cols, sort=False)]
    elif first_col_type == ColDataType.NUMERIC:
        return processed_df[numeric_like_cols.union(cat_cols, sort=False)]
    else:
        # Reorder columns to the original order
        return processed_df[sorted(processed_df.columns)]


def _encode_all_column_values(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        # Add the column name as part of the value.
        df[c] = encode_column_values(df[c])

    return df


def process_data(
    df: pd.DataFrame,
    numeric_max_len=10,
    numeric_precision=4,
    numeric_nparts=2,
    first_col_type=None,
    col_transform_data: Dict = None,
    target_col: str = None,
    numeric_categorical_threshold: int = None,
    numeric_quantile_encoding: bool = False,
    numeric_quantile_bins: int = 1000,
) -> Tuple[pd.DataFrame, Dict, Dict[str, str]]:
    # This should receive a dataframe with dtypes that have already been
    # properly categorized between numeric and categorical.
    # Date type can be converted as UNIX timestamps.
    assert first_col_type in [None, ColDataType.CATEGORICAL, ColDataType.NUMERIC]

    df = df.copy()
    orig_to_processed_col_map: Dict[str, str] = {}

    # Unify the variable for missing data
    df = df.fillna(pd.NA)

    df = _coerce_numeric_dtypes(df)

    if target_col is not None:
        df = _inject_teacher_forcing_column(df, target_col, first_col_type)

    if col_transform_data is None:
        col_transform_data = dict()

    col_idx = _compute_column_index_prefixes(df, col_transform_data)

    # Create a dataframe that will hold the processed data
    processed_series: List[pd.Series] = []

    # Process numerical data
    numeric_cols = df.select_dtypes(include=np.number).columns
    numeric_cols = _split_numeric_categorical_cols(
        df, numeric_cols, col_transform_data, numeric_categorical_threshold
    )

    col_name_to_transform_data: Dict[str, Dict] = dict()

    _process_typed_columns(
        df,
        numeric_cols,
        col_idx,
        ColDataType.NUMERIC,
        lambda s, td: process_numeric_data(
            s,
            max_len=numeric_max_len,
            numeric_precision=numeric_precision,
            transform_data=td,
            quantile_encoding=numeric_quantile_encoding,
            quantile_n_bins=numeric_quantile_bins,
        ),
        col_transform_data,
        numeric_nparts,
        col_name_to_transform_data,
        processed_series,
        orig_to_processed_col_map,
    )

    # Process datetime data
    datetime_cols = df.select_dtypes(include="datetime").columns

    _process_typed_columns(
        df,
        datetime_cols,
        col_idx,
        ColDataType.DATETIME,
        lambda s, td: process_datetime_data(s, transform_data=td),
        col_transform_data,
        numeric_nparts,
        col_name_to_transform_data,
        processed_series,
        orig_to_processed_col_map,
    )

    processed_df = pd.concat([pd.DataFrame()] + processed_series, axis=1)
    processed_df = _tokenize_numeric_like_columns(
        processed_df, col_name_to_transform_data, numeric_nparts
    )

    # NOTE: The categorical data should be the last to be processed!
    categorical_cols = df.columns.difference(numeric_cols).difference(datetime_cols)
    processed_df = _process_categorical_columns(
        df, processed_df, categorical_cols, col_idx, orig_to_processed_col_map
    )

    df = _reorder_columns_by_first_col_type(processed_df, first_col_type)
    df = _encode_all_column_values(df)

    return df, col_transform_data, orig_to_processed_col_map
