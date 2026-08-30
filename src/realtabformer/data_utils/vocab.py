import pandas as pd

from .columns import (
    decode_column_values,
    decode_partition_numeric_col,
    decode_processed_column,
    extract_processed_column,
    is_datetime_col,
    is_numeric_col,
)


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


def _original_column_name(processed_col: str) -> str:
    """Recover the pre-partition, pre-index/dtype-prefix column name from a
    processed column name -- e.g. "0___NUMERIC___price_00" -> "price",
    "2___CATEGORICAL___gender" -> "gender". Numeric/datetime partition
    sub-columns of the same original column (price_00, price_01, ...) all
    map to the same name; `decode_processed_column` alone is not enough
    for those since it leaves the "_00"/"_01" partition suffix attached.
    """
    base = decode_processed_column(processed_col)
    if is_numeric_col(processed_col) or is_datetime_col(processed_col):
        return decode_partition_numeric_col(base)
    return base


def compute_column_blocks(processed_columns) -> list:
    """Group `processed_columns` (e.g. `model.processed_columns`) into
    per-*original*-column blocks, each an `[original_name, [indices]]` pair,
    in original left-to-right order. Every numeric/datetime partition
    sub-column of the same original column (`price_00`, `price_01`, ...)
    lands in one block, in their existing relative order, via the same
    `_original_column_name` grouping `build_pooled_vocab`'s
    `column_type_ids` already uses.

    Built for REaLTabFormerV2's any-order training (`AnyOrderColumnCollator`,
    rtf_datacollator.py) and order-aware sampling (`rtf_sampler.py`): both
    need to permute/query *original* columns as indivisible units without
    ever splitting one column's digit chunks apart from each other.

    Returned as `[[name, [indices]], ...]` (lists, not tuples) so it
    round-trips as-is through `REaLTabFormer2.save()`'s generic
    `json.dumps(self.__dict__)` / `load_from_dir()`'s generic `setattr`
    restore loop -- a tuple would silently become a list on that round trip
    anyway (JSON has no tuple type), so storing it as a list from the start
    avoids a save/load type mismatch instead of masking one.
    """
    blocks: dict = {}
    order: list = []
    for i, col in enumerate(processed_columns):
        name = _original_column_name(col)
        if name not in blocks:
            blocks[name] = []
            order.append(name)
        blocks[name].append(i)

    return [[name, blocks[name]] for name in order]


def build_pooled_vocab(df: pd.DataFrame = None, special_tokens=None):
    """Like `build_vocab`, but the *embedding space* (`id2token`/`token2id`)
    for numeric/datetime partition columns is pooled -- shared across all
    of them, instead of every processed column getting its own -- so the
    same digit chunk in two different numeric columns maps to the same
    token id, and the underlying `wte` row (and therefore anything learned
    about that digit chunk) is shared and updated by every column that
    uses it. Additionally returns `column_type_ids`, mapping each
    processed column to a token id representing its *original*
    (pre-partition) column identity, with every partition sub-column of
    the same original column sharing one id. Intended for
    REaLTabFormerV2's `token_type_ids`-based column-identity embedding
    (added on top of the value token's own embedding, reusing the same
    embedding table -- GPT2-family models compute `token_type_embeds =
    self.wte(token_type_ids)`, the same `wte` used for `input_ids`, so
    these ids must be valid vocab indices, not a separate small id space).

    Crucially, `column_token_ids[col]` -- the set `_prefix_allowed_tokens_fn`
    (rtf_sampler.py) masks generation down to at each column's position,
    and the OOV-fallback pool `get_token_id`/`_vectorized_column_token_ids`
    draw from at training time -- is **not** the full pooled/shared range.
    It's narrowed back down to only the ids for values that specific
    column actually observed, looked up in the shared `token2id`. Sharing
    the embedding *space* and constraining the *allowed set per position*
    are independent concerns: the former is what gives the transfer-
    learning/smaller-vocab benefit (training-time), the latter is what
    gives the hard per-column range guarantee constrained decoding relies
    on (generation-time) -- pooling both together would let `price`'s
    position emit any digit chunk ever seen by *any* numeric/datetime
    column (including a different partition position of `price` itself,
    e.g. its own trailing-digit chunk leaking into its leading-digit
    chunk's allowed set), silently widening every numeric column's
    generation range to the union of all of them. Narrowing
    `column_token_ids` back to per-column keeps the range exactly as
    strict as `build_vocab`'s, with no other change needed anywhere else
    in the pipeline -- everything downstream (constrained decoding, OOV
    fallback, `realtabformer2.py`'s `col_idx_ids`) already reads
    `column_token_ids` generically, unaware of how it was built.

    Values are pooled on their *raw*, column-prefix-stripped form
    (`decode_column_values`): `process_data`'s final step
    (`encode_column_values`) bakes the owning column's name onto the
    front of every cell value, so pooling the as-is (prefixed) strings
    would never actually collide across columns and the shared embedding
    space would be a no-op.

    Categorical columns are intentionally NOT pooled (kept one token
    range per column, same as `build_vocab`) -- a coincidental string
    match across two categorical columns is more likely accidental than
    meaningful, unlike a digit chunk, which is unambiguous regardless of
    which column it came from.

    Unlike `build_vocab`, there is no `add_columns` flag: column-identity
    markers are always produced (that's the point of this function), via
    a dedicated, collision-free mechanism -- not `build_vocab`'s
    `add_columns=True`/`extract_processed_column` path, which mints one
    label per (index, dtype) prefix and therefore collides across
    partition sub-columns of the same original column (confirmed:
    `extract_processed_column("0___NUMERIC___price_00")` and
    `extract_processed_column("0___NUMERIC___price_01")` both return
    `"0___NUMERIC"`). That path is dormant in practice --
    `add_columns=True` is only ever exercised in `build_vocab`'s own
    tests today, never in `realtabformer.py`'s or `realtabformer2.py`'s
    real call sites, both of which pass `add_columns=False` -- so it's a
    latent, out-of-scope bug in `build_vocab`, not something this
    function inherits or fixes.
    """
    assert (df is not None) or special_tokens, (
        "At least one of `df` or `special_tokens` must not be None."
    )

    if df is not None:
        # Same convention assumption as `build_vocab`.
        assert df.columns.str[0].str.isdigit().all()

    id2token = {}
    curr_id = 0
    if special_tokens:
        id2token.update(dict(enumerate(special_tokens)))
        curr_id = len(special_tokens)

    column_token_ids = {}
    column_type_ids = {}

    if df is not None:
        numeric_like_cols = [
            c for c in df.columns if is_numeric_col(c) or is_datetime_col(c)
        ]
        categorical_cols = [c for c in df.columns if c not in numeric_like_cols]

        # Pool numeric/datetime partition-column values into one shared
        # embedding range, but narrow `column_token_ids[col]` back down to
        # each column's own observed values -- see docstring. Each
        # column's decoded (prefix-stripped) values are computed once and
        # reused for both the pooled concat and the per-column narrowing
        # below, instead of decoding every column twice.
        if numeric_like_cols:
            decoded_cols = {c: decode_column_values(df[c]) for c in numeric_like_cols}
            pooled_values = pd.concat(decoded_cols.values(), ignore_index=True)
            unique_vals = sorted(pooled_values.unique())
            id2token.update(dict(enumerate(unique_vals, curr_id)))
            shared_token2id = {v: k for k, v in enumerate(unique_vals, curr_id)}
            curr_id += len(unique_vals)

            for c in numeric_like_cols:
                own_vals = sorted(decoded_cols[c].unique())
                column_token_ids[c] = [shared_token2id[v] for v in own_vals]

        # Categorical columns: unchanged from build_vocab, one range each.
        for c in categorical_cols:
            unique_vals = sorted(df[c].unique())
            id2token.update(dict(enumerate(unique_vals, curr_id)))
            column_token_ids[c] = list(range(curr_id, curr_id + len(unique_vals)))
            curr_id += len(unique_vals)

        # Column-identity markers: one fresh token per *original* column,
        # grouping numeric/datetime partitions together. `__COLTYPE__`
        # prefix guarantees no collision with a real value token.
        original_names = []
        seen = set()
        for c in df.columns:
            name = _original_column_name(c)
            if name not in seen:
                seen.add(name)
                original_names.append(name)

        col_type_tokens = [f"__COLTYPE__{name}" for name in original_names]
        id2token.update(dict(enumerate(col_type_tokens, curr_id)))
        name_to_type_id = {name: curr_id + i for i, name in enumerate(original_names)}
        curr_id += len(col_type_tokens)

        for c in df.columns:
            column_type_ids[c] = name_to_type_id[_original_column_name(c)]

    token2id = {v: k for k, v in id2token.items()}

    return dict(
        id2token=id2token,
        token2id=token2id,
        column_token_ids=column_token_ids,
        column_type_ids=column_type_ids,
    )
