import pandas as pd

from .columns import extract_processed_column


def build_vocab(df: pd.DataFrame = None, special_tokens=None, add_columns: bool = True):
    assert (df is not None) or special_tokens, (
        "At least one of `df` or `special_tokens` must not be None."
    )

    if add_columns and df is not None:
        # We limit this feature to data that are likely
        # to have been processed using the convention imposed
        # where we keep track of the column index in the processed data.
        assert df.columns.str[0].str.isdigit().all()

    id2token = {}
    # `curr_id` is tracked as a running counter incremented by exactly the
    # number of ids assigned at each step, rather than recomputed via
    # `max(id2token) + 1` (which rescans the whole, monotonically-growing
    # dict every time -- O(vocab size) per call instead of O(1)). Since ids
    # are always assigned contiguously starting from `curr_id` and never
    # reused or removed, the two are algebraically identical.
    curr_id = 0
    if special_tokens:
        id2token.update(dict(enumerate(special_tokens)))
        curr_id = len(special_tokens)
    column_token_ids = {}

    if df is not None:
        for col in df.columns:
            unique_vals = sorted(df[col].unique())
            id2token.update(dict(enumerate(unique_vals, curr_id)))
            column_token_ids[col] = list(range(curr_id, curr_id + len(unique_vals)))
            curr_id += len(unique_vals)

        if add_columns:
            col_labels = [extract_processed_column(col) for col in df.columns]
            id2token.update(dict(enumerate(col_labels, curr_id)))
            curr_id += len(col_labels)

    token2id = {v: k for k, v in id2token.items()}

    return dict(
        id2token=id2token,
        token2id=token2id,
        column_token_ids=column_token_ids,
    )
