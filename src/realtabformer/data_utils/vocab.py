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
    curr_id = 0
    if special_tokens:
        id2token.update(dict(enumerate(special_tokens)))
        curr_id = max(id2token) + 1
    column_token_ids = {}

    if df is not None:
        for col in df.columns:
            id2token.update(dict(enumerate(sorted(df[col].unique()), curr_id)))
            column_token_ids[col] = list(range(curr_id, max(id2token) + 1))
            curr_id = max(id2token) + 1

        if add_columns:
            id2token.update(
                dict(
                    enumerate(
                        [extract_processed_column(col) for col in df.columns], curr_id
                    )
                )
            )

            # Add this here to prevent a semantic error in case we add another
            # set of tokens later.
            curr_id = max(id2token) + 1

    token2id = {v: k for k, v in id2token.items()}

    return dict(
        id2token=id2token,
        token2id=token2id,
        column_token_ids=column_token_ids,
    )
