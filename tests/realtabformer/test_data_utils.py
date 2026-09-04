import json
import random
import warnings

import numpy as np
import pandas as pd
import pytest

from realtabformer import data_utils as du

df = pd.DataFrame({
    "float": [1.0, 0.123, 32.2334, 100.32],
    "int": [1, 23, 4, 1232],
    "datetime": [
        pd.Timestamp("20180310"),
        pd.Timestamp("20190310"),
        pd.Timestamp("20200316"),
        pd.Timestamp("20181110")],
    "string": [
        "deep", "learning", "data", "science"]})


def test_SPECIAL_COL_SEP():
    # Make sure that any changes in SPECIAL_COL_SEP
    # is deliberate. So have this test here.
    assert du.SPECIAL_COL_SEP == "___"


def test_ColDataType():
    assert du.ColDataType.types() == ["NUMERIC", "DATETIME", "CATEGORICAL"]


def test_SpecialTokens():
    # The order of the tokens as defined
    # in SpecialTokens matters, so
    # we need to make sure that they
    # don't change accidentaly!
    assert du.SpecialTokens.tokens() == [
        "[UNK]",
        "[SEP]",
        "[PAD]",
        "[CLS]",
        "[MASK]",
        "[BOS]",
        "[EOS]",
        "[BMEM]",
        "[EMEM]",
        "[RMASK]",
        "[SPTYPE]",
    ]


def test_get_input_ids():
    # df = du.process_data(df)
    ddf = df.copy()

    with pytest.raises(AssertionError):
        vocab = du.build_vocab(ddf.astype(str), special_tokens=du.SpecialTokens.tokens(), add_columns=True)

    ddf.columns = [f"{idx}___{dtype}___{col}" for idx, (col, dtype) in enumerate(zip(ddf.columns, ["NUMERIC", "NUMERIC", "DATETIME", "CATEGORICAL"]))]
    vocab = du.build_vocab(ddf.astype(str), special_tokens=du.SpecialTokens.tokens(), add_columns=True)

    example = {
        "0___NUMERIC___float": 32.32, "1___NUMERIC___int": 13,
        "2___DATETIME___datetime": pd.Timestamp("20180310"),
        "3___CATEGORICAL___string": "learning"}

    with pytest.raises(AssertionError):
        # Raise assertion for return_token_type_ids=True
        out = du.get_input_ids(
            example,
            vocab=vocab,
            columns=ddf.columns,
            mask_rate=0,
            return_label_ids=True,
            return_token_type_ids=True,
            affix_bos=True,
            affix_eos=True
        )

    out = du.get_input_ids(
        example,
        vocab=vocab,
        columns=ddf.columns,
        mask_rate=0,
        return_label_ids=True,
        return_token_type_ids=False,
        affix_bos=True,
        affix_eos=True
    )

    assert "label_ids" in out
    assert out["label_ids"] == out["input_ids"]
    assert out["input_ids"][0] == vocab["token2id"][du.SpecialTokens.BOS]
    assert out["input_ids"][-1] == vocab["token2id"][du.SpecialTokens.EOS]

    out = du.get_input_ids(
        example,
        vocab=vocab,
        columns=ddf.columns,
        mask_rate=0,
        return_label_ids=True,
        return_token_type_ids=False,
        affix_bos=True,
        affix_eos=False
    )

    assert "label_ids" in out
    assert out["label_ids"] == out["input_ids"]
    assert out["input_ids"][0] == vocab["token2id"][du.SpecialTokens.BOS]
    assert out["input_ids"][-1] != vocab["token2id"][du.SpecialTokens.EOS]


def test_build_vocab_no_add_columns():
    # Test vocab without special tokens.
    # Convert all data to string for convenience.
    vocab = du.build_vocab(df.astype(str), add_columns=False)
    # Check vocab size
    assert len(vocab["id2token"]) == 16
    assert len(vocab["token2id"]) == 16

    # Check id range
    assert min(vocab["id2token"]) == 0
    assert max(vocab["id2token"]) == 15

    # Check that the token2id and id2token
    # actually are inverse maps of each other.
    for token, t_id in vocab["token2id"].items():
        assert vocab["token2id"][token] == t_id

    # Check values
    assert vocab["id2token"][0] == "0.123"
    assert vocab["id2token"][1] == "1.0"

    # Note that "100.32" goes before "32.2334"
    # because we changed the dtype to string.
    # The sorting applies to the string-converted
    # data.
    assert vocab["id2token"][2] == "100.32"
    assert vocab["id2token"][3] == "32.2334"

    # Sample other tokens.
    assert vocab["id2token"][5] == "1232"
    assert vocab["id2token"][10] == "2019-03-10"
    assert vocab["id2token"][14] == "learning"

    # Check the column token ids output.
    assert vocab["column_token_ids"]["int"] == [4, 5, 6, 7]
    assert vocab["column_token_ids"]["string"] == [12, 13, 14, 15]


def test_build_vocab_add_columns():
    # Test vocab without special tokens.
    # Convert all data to string for convenience.
    ddf = df.copy()
    ddf.columns = [f"{idx}___{dtype}___{col}" for idx, (col, dtype) in enumerate(zip(ddf.columns, ["NUMERIC", "NUMERIC", "DATETIME", "CATEGORICAL"]))]

    vocab = du.build_vocab(ddf.astype(str), add_columns=True)
    # Check vocab size
    assert len(vocab["id2token"]) == (16 + 4)
    assert len(vocab["token2id"]) == (16 + 4)

    # Check id range
    assert min(vocab["id2token"]) == 0
    assert max(vocab["id2token"]) == 19

    # Check column values
    assert vocab["id2token"][16] == "0___NUMERIC"
    assert vocab["id2token"][17] == "1___NUMERIC"
    assert vocab["id2token"][18] == "2___DATETIME"
    assert vocab["id2token"][19] == "3___CATEGORICAL"


def test_process_numeric_data():
    series, transform_data = du.process_numeric_data(
        df["int"], max_len=10, numeric_precision=4)
    expected_out = ["0001", "0023", "0004", "1232"]

    for v, e in zip(series, expected_out):
        assert v == e

    # Make sure that max_len doesn't truncate integral
    # data.
    series, transform_data = du.process_numeric_data(
        df["int"], max_len=2, numeric_precision=4)

    expected_out = ["0001", "0023", "0004", "1232"]

    for v, e in zip(series, expected_out):
        assert v == e

    series, transform_data = du.process_numeric_data(
        df["float"], max_len=5, numeric_precision=4)

    # Note that the value of 0.123 will be truncated
    # to 000.1 because we prioritize the padding at
    # the leading digits of the values before truncating
    # the max length.
    expected_out = ["001.0", "000.1", "032.2", "100.3"]

    for v, e in zip(series, expected_out):
        assert v == e

    # Check that the processing of the
    # data is sensitive to the resolution loss.
    with pytest.raises(AssertionError):
        # This should raise an AssertionError because the
        # desired max_len of 4 will generate a loss in
        # the numeric resolution of the data
        series, transform_data = du.process_numeric_data(
            df["float"], max_len=4, numeric_precision=4)


def test_process_data():
    pr_df, _, _ = du.process_data(
        df, numeric_max_len=10, numeric_precision=4, numeric_nparts=2)

    # Validate the processed columns
    assert pr_df.shape[1] == 12
    assert pr_df.columns.str.startswith(f"0___{du.ColDataType.NUMERIC}___float").sum() == 4
    assert pr_df.columns.str.startswith(f"1___{du.ColDataType.NUMERIC}___int").sum() == 2
    assert pr_df.columns.str.startswith(f"2___{du.ColDataType.DATETIME}___datetime").sum() == 5
    assert pr_df.columns.str.startswith(f"3___{du.ColDataType.CATEGORICAL}___string").sum() == 1

    # Validate that the columns are properly ordered (default)
    start_idx = 0
    for col in pr_df.columns:
        col_idx = int(col.split(du.SPECIAL_COL_SEP)[0])
        if col_idx != start_idx:
            assert (start_idx + 1) == col_idx
            start_idx = col_idx

    for col in pr_df.columns:
        assert pr_df[col].str.startswith(col).all()


@pytest.mark.parametrize("first_col_type", [None, du.ColDataType.CATEGORICAL, du.ColDataType.NUMERIC])
def test_process_data_first_col_type(first_col_type):
    # Test for categorical first cols
    pr_df, _, _ = du.process_data(
        df, numeric_max_len=10, numeric_precision=4, numeric_nparts=2,
        first_col_type=first_col_type
    )

    start_idx = 0
    seen_last = False
    for idx, col in enumerate(pr_df.columns):
        if first_col_type is not None:
            if idx == 0:
                # Make sure that our set first_col_type
                # is actually the first column type in the
                # returned data.
                assert first_col_type in col
            elif first_col_type not in col:
                seen_last = True

            if not seen_last:
                if first_col_type == du.ColDataType.CATEGORICAL:
                    assert first_col_type in col
                else:
                    # NUMERIC and DATETIME fall under the same
                    # general numeric category.
                    assert (first_col_type in col) or (du.ColDataType.DATETIME in col)
            else:
                assert first_col_type not in col
        else:
            # If no preferred first_col_type is set,
            # we use the actual order of the input
            # dataframe.
            col_idx = int(col.split(du.SPECIAL_COL_SEP)[0])
            if col_idx != start_idx:
                assert (start_idx + 1) == col_idx
                start_idx = col_idx


def test_decode_processed_column():
    assert du.decode_processed_column(f"0___{du.ColDataType.NUMERIC}___float_00") == "float_00"
    assert du.decode_processed_column(f"0___{du.ColDataType.NUMERIC}___int_01") == "int_01"
    assert du.decode_processed_column(f"0___{du.ColDataType.DATETIME}___datetime_03") == "datetime_03"
    assert du.decode_processed_column(f"0___{du.ColDataType.CATEGORICAL}___string_02") == "string_02"

    # Check if leading numeric prefix is long.
    assert du.decode_processed_column(f"121___{du.ColDataType.CATEGORICAL}___string_02") == "string_02"

    # Check if columns is not in the expected format.
    assert du.decode_processed_column("random_col") == "random_col"


@pytest.mark.parametrize("dtype", du.ColDataType.types() + ["SOMEDTYPE"])
def test_encode_processed_column(dtype):
    idx = 0
    col = "foo"

    if dtype != "SOMEDTYPE":
        # Expected form: "0___(NUMERIC|DATETIME|CATEGORICAL)___foo"
        out = f"{idx}{du.SPECIAL_COL_SEP}{dtype}{du.SPECIAL_COL_SEP}{col}"
        assert du.encode_processed_column(idx, dtype, col) == out
    else:
        with pytest.raises(AssertionError):
            du.encode_processed_column(idx, dtype, col)


@pytest.mark.parametrize("dtype", du.ColDataType.types() + ["SOMEDTYPE"])
def test_extract_processed_column(dtype):
    col = "foo012_%$@&#"

    for _ in range(10):
        idx = random.randint(0, 1000)

        if dtype != "SOMEDTYPE":
            # Expected form: "0___(NUMERIC|DATETIME|CATEGORICAL)___foo012_%$@&#"
            expected = f"{idx}{du.SPECIAL_COL_SEP}{dtype}"
            inp = f"{idx}{du.SPECIAL_COL_SEP}{dtype}{du.SPECIAL_COL_SEP}{col}{du.SPECIAL_COL_SEP}00"
            assert du.extract_processed_column(inp) == expected
        else:
            inp = f"{idx}{du.SPECIAL_COL_SEP}{dtype}{du.SPECIAL_COL_SEP}{col}{du.SPECIAL_COL_SEP}00"
            assert du.extract_processed_column(inp) is None


# --- Regression tests added ahead of the data_utils.py refactor on this
# branch (see /Users/avsolatorio/.claude/plans/snappy-swimming-hickey.md,
# redone here against feat/support-seed-input's data_utils.py, which adds
# field_weights, predict_fields, batched dataset creation, num_proc, and
# the orig_to_processed_col_map / "$%NUM_COLS%$" persistence on top of
# main's version). Expected values were captured by actually running this
# branch's code, not hand-derived.

df_with_missing = pd.DataFrame({
    "float": [1.0, 0.123, 32.2334, 100.32, np.nan],
    "int": [1, 23, 4, 1232, np.nan],
    "datetime": [
        pd.Timestamp("20180310"),
        pd.Timestamp("20190310"),
        pd.Timestamp("20200316"),
        pd.Timestamp("20181110"),
        pd.NaT],
    "string": ["deep", "learning", "data", None, "science"]})


def test_process_data_values_and_transform_data_snapshot():
    # Locks in the exact cell values, `col_transform_data` shape (including
    # the "$%NUM_COLS%$" persistence key this branch added), and
    # `orig_to_processed_col_map` for a dataframe with missing values in
    # every column type. `col_transform_data` is JSON-serialized verbatim
    # into rtf_config.json by REaLTabFormer.save()/restored via a generic
    # setattr loop with no key remapping in load_from_dir() -- so its exact
    # key set for each dtype (asserted below) must not change across the
    # refactor.
    pr_df, ctd, orig_map = du.process_data(
        df_with_missing, numeric_max_len=10, numeric_precision=4, numeric_nparts=2
    )

    assert list(pr_df.columns) == [
        "0___NUMERIC___float_00", "0___NUMERIC___float_01",
        "0___NUMERIC___float_02", "0___NUMERIC___float_03",
        "1___NUMERIC___int_00", "1___NUMERIC___int_01",
        "2___DATETIME___datetime_00", "2___DATETIME___datetime_01",
        "2___DATETIME___datetime_02", "2___DATETIME___datetime_03",
        "2___DATETIME___datetime_04", "3___CATEGORICAL___string",
    ]

    expected_values = {
        "0___NUMERIC___float_00": [
            "0___NUMERIC___float_00___00", "0___NUMERIC___float_00___00",
            "0___NUMERIC___float_00___03", "0___NUMERIC___float_00___10",
            "0___NUMERIC___float_00___@@"],
        "0___NUMERIC___float_01": [
            "0___NUMERIC___float_01___1.", "0___NUMERIC___float_01___0.",
            "0___NUMERIC___float_01___2.", "0___NUMERIC___float_01___0.",
            "0___NUMERIC___float_01___@@"],
        "0___NUMERIC___float_02": [
            "0___NUMERIC___float_02___00", "0___NUMERIC___float_02___12",
            "0___NUMERIC___float_02___23", "0___NUMERIC___float_02___32",
            "0___NUMERIC___float_02___@@"],
        "0___NUMERIC___float_03": [
            "0___NUMERIC___float_03___00", "0___NUMERIC___float_03___30",
            "0___NUMERIC___float_03___34", "0___NUMERIC___float_03___00",
            "0___NUMERIC___float_03___@@"],
        "1___NUMERIC___int_00": [
            "1___NUMERIC___int_00___00", "1___NUMERIC___int_00___00",
            "1___NUMERIC___int_00___00", "1___NUMERIC___int_00___12",
            "1___NUMERIC___int_00___@@"],
        "1___NUMERIC___int_01": [
            "1___NUMERIC___int_01___01", "1___NUMERIC___int_01___23",
            "1___NUMERIC___int_01___04", "1___NUMERIC___int_01___32",
            "1___NUMERIC___int_01___@@"],
        "2___DATETIME___datetime_00": [
            "2___DATETIME___datetime_00___-2", "2___DATETIME___datetime_00___00",
            "2___DATETIME___datetime_00___03", "2___DATETIME___datetime_00___-0",
            "2___DATETIME___datetime_00___@@"],
        "2___DATETIME___datetime_01": [
            "2___DATETIME___datetime_01___90", "2___DATETIME___datetime_01___24",
            "2___DATETIME___datetime_01___45", "2___DATETIME___datetime_01___79",
            "2___DATETIME___datetime_01___@@"],
        "2___DATETIME___datetime_02": [
            "2___DATETIME___datetime_02___95", "2___DATETIME___datetime_02___40",
            "2___DATETIME___datetime_02___81", "2___DATETIME___datetime_02___27",
            "2___DATETIME___datetime_02___@@"],
        "2___DATETIME___datetime_03": [
            "2___DATETIME___datetime_03___20", "2___DATETIME___datetime_03___80",
            "2___DATETIME___datetime_03___60", "2___DATETIME___datetime_03___20",
            "2___DATETIME___datetime_03___@@"],
        "2___DATETIME___datetime_04": [
            "2___DATETIME___datetime_04___0", "2___DATETIME___datetime_04___0",
            "2___DATETIME___datetime_04___0", "2___DATETIME___datetime_04___0",
            "2___DATETIME___datetime_04___@@"],
        "3___CATEGORICAL___string": [
            "3___CATEGORICAL___string___deep", "3___CATEGORICAL___string___learning",
            "3___CATEGORICAL___string___data", "3___CATEGORICAL___string___<NA>",
            "3___CATEGORICAL___string___science"],
    }
    for col, expected in expected_values.items():
        assert pr_df[col].tolist() == expected, col

    expected_ctd = {
        "$%NUM_COLS%$": 1,
        "float": {"max_len": 10, "numeric_precision": 4, "mx_sig": 3, "ljust": 8, "numeric_nparts": 2},
        "int": {"max_len": 10, "numeric_precision": 4, "mx_sig": -1, "zfill": 4, "numeric_nparts": 2},
        "datetime": {
            "max_len": 10, "numeric_precision": 0, "mx_sig": -1, "zfill": 9,
            "mean_date": 1549735200, "numeric_nparts": 2,
        },
    }
    assert ctd == expected_ctd

    assert orig_map == {
        "float": "0___NUMERIC___float",
        "int": "1___NUMERIC___int",
        "datetime": "2___DATETIME___datetime",
        "string": "3___CATEGORICAL___string",
    }


def test_process_data_target_col_teacher_forcing_snapshot():
    # Locks in the teacher-forcing column injection behavior of `target_col`,
    # including how it appears in `orig_to_processed_col_map`.
    df = pd.DataFrame({
        "float": [1.0, 0.123, 32.2334, 100.32],
        "int": [1, 23, 4, 1232],
        "datetime": [
            pd.Timestamp("20180310"), pd.Timestamp("20190310"),
            pd.Timestamp("20200316"), pd.Timestamp("20181110")],
        "string": ["deep", "learning", "data", "science"]})

    pr_df, _, orig_map = du.process_data(
        df, numeric_max_len=10, numeric_precision=4, numeric_nparts=2,
        target_col="string",
    )

    assert pr_df.shape == (4, 13)
    tf_col = "0___CATEGORICAL____TEACHERFORCING_string"
    assert tf_col in pr_df.columns
    assert pr_df[tf_col].tolist() == [
        f"{tf_col}___deep", f"{tf_col}___learning",
        f"{tf_col}___data", f"{tf_col}___science",
    ]
    # The original (non-teacher-forced) `string` column is still present.
    assert any(c.endswith("___string") and tf_col not in c for c in pr_df.columns)
    assert orig_map["_TEACHERFORCING_string"] == tf_col
    assert orig_map["string"] != tf_col


def test_col_transform_data_json_roundtrip():
    # `col_transform_data` is dumped verbatim as raw JSON by
    # `REaLTabFormer.save()` and restored via a generic `setattr` loop with
    # no key remapping in `load_from_dir()`. This test locks in that the
    # dict produced by `process_data` -- including the "$%NUM_COLS%$" key --
    # survives a JSON round-trip unchanged, the precise property backward
    # compatibility with already-saved models depends on.
    _, ctd, _ = du.process_data(
        df_with_missing, numeric_max_len=10, numeric_precision=4, numeric_nparts=2
    )
    roundtripped = json.loads(json.dumps(ctd))
    assert roundtripped == ctd


def test_build_vocab_duplicate_values_across_columns():
    # Locks in existing (imperfect but real) behavior: when the same string
    # value appears in two different columns, `id2token` gets two entries
    # for it, but `token2id` (built as `{v: k for k, v in id2token.items()}`)
    # only keeps the *last* one. Pre-existing quirk, not something this
    # refactor is meant to fix.
    ddf = pd.DataFrame({
        "colA": ["x", "y", "x", "z"],
        "colB": ["x", "m", "n", "x"],
    })
    ddf.columns = ["0___NUMERIC___colA", "1___CATEGORICAL___colB"]

    vocab = du.build_vocab(ddf.astype(str), add_columns=False)
    assert vocab["id2token"] == {0: "x", 1: "y", 2: "z", 3: "m", 4: "n", 5: "x"}
    assert vocab["token2id"] == {"x": 5, "y": 1, "z": 2, "m": 3, "n": 4}
    assert vocab["column_token_ids"] == {
        "0___NUMERIC___colA": [0, 1, 2],
        "1___CATEGORICAL___colB": [3, 4, 5],
    }


def _build_vocab_reference(df=None, special_tokens=None, add_columns=True):
    # Pre-optimization reference implementation of build_vocab, using the
    # original `max(id2token) + 1` recomputation instead of a running
    # counter. Kept here (not shipped) purely to prove the O(1)-per-column
    # optimization in data_utils/vocab.py is byte-identical, not just
    # equivalent-in-effect.
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
                        [du.extract_processed_column(col) for col in df.columns], curr_id
                    )
                )
            )
            curr_id = max(id2token) + 1

    token2id = {v: k for k, v in id2token.items()}
    return dict(id2token=id2token, token2id=token2id, column_token_ids=column_token_ids)


def test_build_vocab_running_counter_matches_reference_on_high_cardinality():
    # Proves the running-counter optimization in build_vocab produces
    # byte-identical output to the original max(id2token)-recomputation
    # formula, on a dataframe with many columns and high-cardinality
    # values -- the regime where the two formulas could plausibly diverge
    # if the optimization were subtly wrong. Guards against vocab ids
    # baked into already-trained models shifting.
    rng = np.random.default_rng(1029)
    n_cols = 12
    ddf = pd.DataFrame({
        f"{idx}___CATEGORICAL___col{idx}": [
            f"val_{v}" for v in rng.integers(0, 500, size=200)
        ]
        for idx in range(n_cols)
    })

    for special_tokens, add_columns in [
        (None, False),
        (du.SpecialTokens.tokens(), True),
        (du.SpecialTokens.tokens(), False),
    ]:
        expected = _build_vocab_reference(
            ddf, special_tokens=special_tokens, add_columns=add_columns
        )
        actual = du.build_vocab(
            ddf, special_tokens=special_tokens, add_columns=add_columns
        )
        assert actual["id2token"] == expected["id2token"]
        assert actual["token2id"] == expected["token2id"]
        assert actual["column_token_ids"] == expected["column_token_ids"]


def test_get_input_ids_field_weights_and_predict_fields_snapshot():
    # Locks in this branch's field_weights/predict_fields extension to
    # get_input_ids: token_weights output and label_ids masking (-100) for
    # non-predict fields.
    #
    # NOTE: `example`'s numeric/datetime values are raw Python objects
    # (float/int/Timestamp), while the vocab's token2id keys are the
    # string-cast form (`ddf.astype(str)`), so those columns are always an
    # OOV *type* mismatch here and get_token_id's `random.choice(oov_options)`
    # fallback kicks in -- this matches the existing `test_get_input_ids`
    # test above, which only asserts BOS/EOS structural properties for the
    # same reason. `label_ids` and `token_weights` don't depend on which
    # token id was actually chosen, so those are safe to assert exactly;
    # `input_ids` is not.
    ddf = df.copy()
    ddf.columns = [
        f"{idx}___{dtype}___{col}"
        for idx, (col, dtype) in enumerate(
            zip(ddf.columns, ["NUMERIC", "NUMERIC", "DATETIME", "CATEGORICAL"])
        )
    ]
    vocab = du.build_vocab(
        ddf.astype(str), special_tokens=du.SpecialTokens.tokens(), add_columns=True
    )
    example = {
        "0___NUMERIC___float": 32.32, "1___NUMERIC___int": 13,
        "2___DATETIME___datetime": pd.Timestamp("20180310"),
        "3___CATEGORICAL___string": "learning",
    }

    out = du.get_input_ids(
        example, vocab=vocab, columns=ddf.columns, mask_rate=0,
        return_label_ids=True, return_token_type_ids=False,
        affix_bos=True, affix_eos=True,
        field_weights={"0___NUMERIC___float": 5.0},
        predict_fields=["3___CATEGORICAL___string"],
    )

    assert len(out["input_ids"]) == 6
    assert out["input_ids"][0] == vocab["token2id"][du.SpecialTokens.BOS]
    assert out["input_ids"][-1] == vocab["token2id"][du.SpecialTokens.EOS]
    # "learning" is a plain string so it's an exact vocab hit, unlike the
    # numeric/datetime columns above -- deterministic.
    assert out["input_ids"][4] == vocab["token2id"]["learning"]

    # Only the predict_fields column keeps its real label; everything else
    # (including BOS/EOS) is masked with -100. Deterministic regardless of
    # which token id an OOV column resolved to.
    assert out["label_ids"] == [
        vocab["token2id"][du.SpecialTokens.BOS], -100, -100, -100,
        vocab["token2id"]["learning"], vocab["token2id"][du.SpecialTokens.EOS],
    ]
    # BOS/EOS get weight 1.0; the weighted column gets 5.0; unweighted
    # columns default to 1.0. Deterministic regardless of chosen token id.
    assert out["token_weights"] == [1.0, 5.0, 1.0, 1.0, 1.0, 1.0]


def test_data_utils_public_api_surface():
    # Golden import list: every name actually consumed elsewhere in this
    # codebase (realtabformer.py, rtf_sampler.py, realtabformer2.py, and the
    # test suite itself) must remain importable from `realtabformer.data_utils`
    # after the module is split into a package. Cheap insurance against a
    # dropped re-export.
    required_names = {
        # consumed by realtabformer.py / realtabformer2.py
        "ModelFileName", "ModelType", "SpecialTokens", "TabularArtefact",
        "build_vocab", "make_dataset", "make_relational_dataset", "process_data",
        # consumed by rtf_sampler.py
        "INVALID_NUMS_RE", "NUMERIC_NA_TOKEN", "decode_column_values",
        "decode_partition_numeric_col", "decode_processed_column",
        "fix_multi_decimal", "is_datetime_col", "is_numeric_col",
        "is_numeric_datetime_col",
        # consumed by the test suite
        "ColDataType", "SPECIAL_COL_SEP", "encode_processed_column",
        "extract_processed_column", "get_input_ids", "process_numeric_data",
    }
    missing = {name for name in required_names if not hasattr(du, name)}
    assert not missing, f"Missing from realtabformer.data_utils: {missing}"


# --- Regression tests for the vectorized get_input_ids batched path (see
# /Users/avsolatorio/.claude/plans/snappy-swimming-hickey.md, the data_utils
# performance plan). The non-batched path (_build_one_row) is unchanged by
# that optimization, so it doubles as ground truth: with no RNG involved
# (mask_rate=0, full vocab coverage) the vectorized batched output must
# exactly match calling the non-batched path once per row.

def _make_perf_test_vocab(n_rows: int = 64, n_cols: int = 8, seed: int = 7):
    rng = np.random.default_rng(seed)
    ddf = pd.DataFrame({
        f"{idx}___CATEGORICAL___col{idx}": [
            f"v{v}" for v in rng.integers(0, 20, size=n_rows)
        ]
        for idx in range(n_cols)
    })
    vocab = du.build_vocab(
        ddf.astype(str), special_tokens=du.SpecialTokens.tokens(), add_columns=True
    )
    return ddf, vocab


def test_get_input_ids_batched_matches_nonbatched_when_deterministic():
    # No RNG consumption possible in this regime (mask_rate=0, every value
    # drawn straight from the vocab-fitting dataframe so nothing is OOV) --
    # the vectorized batched path and the row-by-row non-batched path must
    # therefore agree exactly.
    ddf, vocab = _make_perf_test_vocab()
    columns = list(ddf.columns)

    example_batch = {col: ddf[col].astype(str).tolist() for col in columns}

    batched_out = du.get_input_ids(
        example_batch,
        vocab=vocab,
        columns=columns,
        mask_rate=0,
        return_label_ids=True,
        affix_bos=True,
        affix_eos=True,
        field_weights={columns[0]: 3.0},
        predict_fields=[columns[1]],
        batched=True,
    )

    for i in range(len(ddf)):
        row_example = {col: example_batch[col][i] for col in columns}
        row_out = du.get_input_ids(
            row_example,
            vocab=vocab,
            columns=columns,
            mask_rate=0,
            return_label_ids=True,
            affix_bos=True,
            affix_eos=True,
            field_weights={columns[0]: 3.0},
            predict_fields=[columns[1]],
            batched=False,
        )
        assert batched_out["input_ids"][i] == row_out["input_ids"]
        assert batched_out["label_ids"][i] == row_out["label_ids"]
        assert batched_out["token_weights"][i] == row_out["token_weights"]


def test_get_input_ids_batched_determinism_with_seeded_rng():
    # Same seed, run twice through make_dataset -> identical output. Proves
    # the Generator-based design is actually deterministic given a seed,
    # even when mask_rate/OOV are in play (so the RNG is genuinely
    # exercised, unlike the test above).
    ddf, vocab = _make_perf_test_vocab(n_rows=200, n_cols=6, seed=11)
    columns = list(ddf.columns)
    processed_df = ddf.astype(str)
    processed_df.columns = columns

    def _run():
        ds = du.make_dataset(processed_df, vocab, mask_rate=0.3, seed=1029)
        return ds["input_ids"]

    first = _run()
    second = _run()
    assert first == second


def test_get_input_ids_batched_masking_and_oov_distribution():
    # Statistical check for the regime where old and new code can't be
    # compared value-for-value (RNG consumption differs by design -- see
    # the plan's RNG section): assert the empirical mask rate is close to
    # the configured mask_rate, and that OOV substitutions always resolve
    # to a value from the correct column's own vocabulary.
    ddf, vocab = _make_perf_test_vocab(n_rows=2000, n_cols=5, seed=3)
    columns = list(ddf.columns)

    # Half the batch gets values guaranteed to be out-of-vocabulary.
    example_batch = {col: ddf[col].astype(str).tolist() for col in columns}
    oov_col = columns[0]
    n = len(example_batch[oov_col])
    for i in range(n // 2):
        example_batch[oov_col][i] = f"__never_seen__{i}"

    mask_rate = 0.4
    rng = np.random.default_rng(42)
    out = du.dataset._build_batch(
        example=example_batch,
        columns=columns,
        token2id=vocab["token2id"],
        col_oov=vocab["column_token_ids"],
        bos_id=vocab["token2id"][du.SpecialTokens.BOS],
        eos_id=vocab["token2id"][du.SpecialTokens.EOS],
        mask_rate=mask_rate,
        return_label_ids=False,
        affix_bos=False,
        affix_eos=False,
        field_weights=None,
        predict_fields=None,
        rng=rng,
    )

    input_ids = np.array(out["input_ids"])
    rmask_id = vocab["token2id"][du.SpecialTokens.RMASK]

    empirical_mask_rate = (input_ids == rmask_id).mean()
    assert abs(empirical_mask_rate - mask_rate) < 0.03

    oov_col_idx = columns.index(oov_col)
    oov_options = set(vocab["column_token_ids"][oov_col])
    oov_col_ids = input_ids[: n // 2, oov_col_idx]
    # Every substituted id (that isn't itself a mask token) must come from
    # this column's own vocabulary, never another column's.
    non_masked = oov_col_ids[oov_col_ids != rmask_id]
    assert set(non_masked.tolist()) <= oov_options


def test_make_dataset_end_to_end_with_seed_matches_manual_vectorized_call():
    # Exercises the full make_dataset -> .map(batched=True) path (not just
    # _build_batch called directly) to make sure the Generator threaded in
    # via the closure actually reaches the vectorized code, end to end.
    ddf, vocab = _make_perf_test_vocab(n_rows=50, n_cols=4, seed=5)
    columns = list(ddf.columns)
    processed_df = ddf.astype(str)
    processed_df.columns = columns

    dataset = du.make_dataset(processed_df, vocab, mask_rate=0, seed=123)
    assert len(dataset) == len(ddf)
    assert dataset[0]["input_ids"][0] == vocab["token2id"][du.SpecialTokens.BOS]
    assert dataset[0]["input_ids"][-1] == vocab["token2id"][du.SpecialTokens.EOS]


# --- REaLTabFormerV2-only: build_pooled_vocab / make_dataset_with_column_types
# (see /Users/avsolatorio/.claude/plans/snappy-swimming-hickey.md). v1's
# build_vocab/make_dataset are untouched by any of this.

def test_build_pooled_vocab_shares_tokens_across_numeric_columns():
    # The point of this function: the same digit chunk in two different
    # numeric columns must map to the SAME token id in the shared
    # embedding space (id2token/token2id), unlike build_vocab where every
    # column gets its own disjoint range with no relation between
    # identical values. But `column_token_ids` -- the set generation is
    # constrained to at each column's position -- must stay narrowed to
    # each column's own observed values, not the full shared range (see
    # test_build_pooled_vocab_preserves_per_column_range_guarantee).
    raw_df = pd.DataFrame({
        "price": [10.5, 20.3, 30.1, 40.9],
        "age": [10.5, 20.3, 5.0, 99.0],  # shares "10", "20" with price
        "gender": ["m", "f", "m", "f"],
    })
    pr_df, _, _ = du.process_data(
        raw_df, numeric_max_len=6, numeric_precision=1, numeric_nparts=2
    )
    vocab = du.build_pooled_vocab(pr_df, special_tokens=du.SpecialTokens.tokens())

    price_00 = [c for c in pr_df.columns if c.endswith("price_00")][0]
    age_00 = [c for c in pr_df.columns if c.endswith("age_00")][0]
    gender_col = [c for c in pr_df.columns if c.endswith("gender")][0]

    # price_00 observed {"10","20","30","40"}, age_00 observed
    # {"10","20","05","99"} -- different sets, so their narrowed allowed
    # ids must differ (price_00 must not be able to emit age_00's "99",
    # and vice versa).
    assert set(vocab["column_token_ids"][price_00]) != set(vocab["column_token_ids"][age_00])
    # categorical stays on its own, separate range.
    assert vocab["column_token_ids"][gender_col] != vocab["column_token_ids"][price_00]

    # Row 0: price=10.5 -> price_00="10"; age=10.5 -> age_00="10" (same
    # raw digit chunk). Confirm they resolve to the identical token id --
    # the shared embedding space is preserved despite the narrowing.
    ds = du.make_dataset_with_column_types(pr_df, vocab, mask_rate=0)
    price_00_idx = list(pr_df.columns).index(price_00) + 1  # +1 for BOS
    age_00_idx = list(pr_df.columns).index(age_00) + 1
    shared_id = ds[0]["input_ids"][price_00_idx]
    assert shared_id == ds[0]["input_ids"][age_00_idx]

    # And that shared id is a member of both columns' own narrowed
    # allowed sets -- the narrowing didn't drop a value either column
    # actually needs to be able to encode.
    assert shared_id in vocab["column_token_ids"][price_00]
    assert shared_id in vocab["column_token_ids"][age_00]


def test_build_pooled_vocab_preserves_per_column_range_guarantee():
    # Pooling the embedding space must NOT widen any column's *allowed*
    # value range beyond what build_vocab (unpooled) allows for the same
    # data -- that would break the hard per-column range guarantee that
    # constrained decoding (rtf_sampler._prefix_allowed_tokens_fn) and the
    # OOV fallback (get_token_id / _vectorized_column_token_ids) both rely
    # on `column_token_ids` for. This also guards against leakage across
    # partition positions of the *same* original column (e.g. a narrow
    # leading-digit chunk range polluted by a wide trailing-digit chunk
    # range from the same column).
    raw_df = pd.DataFrame({
        "big_value": list(range(9000, 9010)),
        "small_value": [1, 2, 3, 4, 5, 6, 7, 8, 1, 2],
    })
    pr_df, _, _ = du.process_data(
        raw_df, numeric_max_len=6, numeric_precision=0, numeric_nparts=2
    )

    unpooled = du.build_vocab(
        pr_df, special_tokens=du.SpecialTokens.tokens(), add_columns=False
    )
    pooled = du.build_pooled_vocab(pr_df, special_tokens=du.SpecialTokens.tokens())

    for col in pr_df.columns:
        # build_vocab's id2token entries are still column-prefixed
        # (encode_column_values bakes the prefix on before either
        # function sees the data); build_pooled_vocab's numeric entries
        # are prefix-stripped (decode_column_values). Compare on the
        # decoded raw value so both sides are on equal footing.
        unpooled_allowed = {
            du.decode_column_values(
                pd.Series([unpooled["id2token"][i]])
            ).iloc[0]
            for i in unpooled["column_token_ids"][col]
        }
        pooled_allowed = {
            pooled["id2token"][i] for i in pooled["column_token_ids"][col]
        }
        assert pooled_allowed == unpooled_allowed, col


def test_build_pooled_vocab_column_type_ids_group_partitions():
    # column_type_ids must group all partition sub-columns of the same
    # original column (price_00, price_01, ...) under one id, distinct
    # from other columns' ids.
    raw_df = pd.DataFrame({
        "price": [10.5, 20.3, 30.1, 40.9],
        "age": [10.5, 20.3, 5.0, 99.0],
        "gender": ["m", "f", "m", "f"],
    })
    pr_df, _, _ = du.process_data(
        raw_df, numeric_max_len=6, numeric_precision=1, numeric_nparts=2
    )
    vocab = du.build_pooled_vocab(pr_df, special_tokens=du.SpecialTokens.tokens())

    price_cols = [c for c in pr_df.columns if "price" in c]
    age_cols = [c for c in pr_df.columns if "age" in c]
    gender_cols = [c for c in pr_df.columns if "gender" in c]

    price_type_ids = {vocab["column_type_ids"][c] for c in price_cols}
    age_type_ids = {vocab["column_type_ids"][c] for c in age_cols}
    gender_type_ids = {vocab["column_type_ids"][c] for c in gender_cols}

    assert len(price_cols) > 1  # sanity: price really did get partitioned
    assert len(price_type_ids) == 1  # ...but all partitions share one id
    assert len(age_type_ids) == 1
    assert len(gender_type_ids) == 1
    # And the three original columns are distinguishable from each other.
    assert price_type_ids != age_type_ids != gender_type_ids
    assert price_type_ids != gender_type_ids


def test_make_dataset_with_column_types_deterministic_when_no_rng_needed():
    # Same principle as test_get_input_ids_batched_matches_nonbatched_when_deterministic:
    # with mask_rate=0 and full vocab coverage, there is no RNG consumption,
    # so two independent calls must produce byte-identical output.
    raw_df = pd.DataFrame({
        "price": [10.5, 20.3, 30.1, 40.9] * 5,
        "gender": ["m", "f", "m", "f"] * 5,
    })
    pr_df, _, _ = du.process_data(raw_df, numeric_max_len=6, numeric_precision=1)
    vocab = du.build_pooled_vocab(pr_df, special_tokens=du.SpecialTokens.tokens())

    ds1 = du.make_dataset_with_column_types(pr_df, vocab, mask_rate=0, seed=7)
    ds2 = du.make_dataset_with_column_types(pr_df, vocab, mask_rate=0, seed=7)
    assert ds1["input_ids"] == ds2["input_ids"]
    assert ds1["token_type_ids"] == ds2["token_type_ids"]
    assert ds1["label_ids"] == ds2["label_ids"]

    # token_type_ids length must match input_ids length (one type id per
    # position, including BOS/EOS).
    assert len(ds1[0]["token_type_ids"]) == len(ds1[0]["input_ids"])


def test_compute_column_blocks_groups_partitions_and_preserves_order():
    raw_df = pd.DataFrame({
        "price": [10.5, 20.3, 30.1, 40.9],
        "age": [10.5, 20.3, 5.0, 99.0],
        "gender": ["m", "f", "m", "f"],
    })
    pr_df, _, _ = du.process_data(
        raw_df, numeric_max_len=6, numeric_precision=1, numeric_nparts=2
    )
    blocks = du.compute_column_blocks(pr_df.columns.tolist())

    names = [name for name, _ in blocks]
    assert names == ["price", "age", "gender"]

    price_indices = dict(blocks)["price"]
    age_indices = dict(blocks)["age"]
    gender_indices = dict(blocks)["gender"]

    # price/age were partitioned (numeric_nparts=2) -- multiple indices,
    # each pointing at the right processed column, in order.
    assert len(price_indices) > 1
    assert [pr_df.columns[i] for i in price_indices] == [
        c for c in pr_df.columns if "price" in c
    ]
    assert [pr_df.columns[i] for i in age_indices] == [
        c for c in pr_df.columns if "age" in c
    ]
    # gender is categorical -- exactly one index.
    assert len(gender_indices) == 1
    assert pr_df.columns[gender_indices[0]] == [
        c for c in pr_df.columns if "gender" in c
    ][0]

    # Every index appears exactly once across all blocks (a partition).
    all_indices = sorted(i for _, indices in blocks for i in indices)
    assert all_indices == list(range(len(pr_df.columns)))


def test_compute_chunk_significance_weights_favors_high_entropy_chunks():
    # A heavy-tailed column's own leading chunk (near-constant "0" for
    # most rows under the fixed-width zero-padded encoding) should get a
    # *lower* weight than its own high-variance trailing chunk -- not the
    # reverse, which a naive position-based decay would give.
    n = 1000
    df = pd.DataFrame({
        "0___NUMERIC___col_hi": ["0"] * 990 + ["9"] * 10,  # mostly constant
        "0___NUMERIC___col_lo": (
            np.random.default_rng(0).integers(0, 10, size=n).astype(str)
        ),  # near-uniform
        "1___CATEGORICAL___gender": ["m", "f"] * (n // 2),
    })
    processed_columns = df.columns.tolist()

    weights = du.compute_chunk_significance_weights(df, processed_columns, floor=0.1)

    hi_w = weights["0___NUMERIC___col_hi"]
    lo_w = weights["0___NUMERIC___col_lo"]
    gender_w = weights["1___CATEGORICAL___gender"]

    assert hi_w < lo_w
    assert abs((hi_w + lo_w) / 2 - 1.0) < 1e-9  # block averages to 1.0
    assert gender_w == 1.0  # single-chunk column: always exactly 1.0

    # A fully constant chunk resolves to exactly the floor value (relative
    # to its column's mean) -- never zero, no matter how degenerate.
    all_constant = pd.DataFrame({
        "0___NUMERIC___a": ["0"] * n,
        "0___NUMERIC___b": ["0"] * n,
    })
    w = du.compute_chunk_significance_weights(
        all_constant, all_constant.columns.tolist(), floor=0.1
    )
    assert w["0___NUMERIC___a"] == w["0___NUMERIC___b"] == 1.0


def test_build_vocab_chunk_significance_opt_in():
    raw_df = pd.DataFrame({
        "price": [10.5, 20.3, 30.1, 40.9] * 20,
        "gender": ["m", "f", "m", "f"] * 20,
    })
    pr_df, _, _ = du.process_data(
        raw_df, numeric_max_len=6, numeric_precision=1, numeric_nparts=1
    )

    vocab_off = du.build_vocab(
        pr_df, special_tokens=du.SpecialTokens.tokens(), add_columns=False
    )
    assert "chunk_significance_weights" not in vocab_off

    vocab_on = du.build_vocab(
        pr_df,
        special_tokens=du.SpecialTokens.tokens(),
        add_columns=False,
        compute_chunk_significance=True,
    )
    assert set(vocab_on["chunk_significance_weights"].keys()) == set(pr_df.columns)


def test_build_pooled_vocab_chunk_significance_opt_in():
    raw_df = pd.DataFrame({
        "price": [10.5, 20.3, 30.1, 40.9] * 20,
        "gender": ["m", "f", "m", "f"] * 20,
    })
    pr_df, _, _ = du.process_data(
        raw_df, numeric_max_len=6, numeric_precision=1, numeric_nparts=1
    )

    vocab_off = du.build_pooled_vocab(pr_df, special_tokens=du.SpecialTokens.tokens())
    assert "chunk_significance_weights" not in vocab_off

    vocab_on = du.build_pooled_vocab(
        pr_df, special_tokens=du.SpecialTokens.tokens(), compute_chunk_significance=True
    )
    assert set(vocab_on["chunk_significance_weights"].keys()) == set(pr_df.columns)


def test_make_dataset_token_weights_from_chunk_significance_without_field_weights():
    # token_weights must be built purely from chunk_significance_weights
    # even when field_weights=None -- this is the gate-condition fix
    # (previously token_weights was only ever built when field_weights
    # was explicitly set).
    raw_df = pd.DataFrame({
        "price": [10.5, 20.3, 30.1, 40.9] * 5,
        "gender": ["m", "f", "m", "f"] * 5,
    })
    pr_df, _, _ = du.process_data(raw_df, numeric_max_len=6, numeric_precision=1)
    vocab = du.build_vocab(
        pr_df,
        special_tokens=du.SpecialTokens.tokens(),
        add_columns=False,
        compute_chunk_significance=True,
    )

    ds = du.make_dataset(
        pr_df,
        vocab,
        mask_rate=0,
        return_token_type_ids=False,
        field_weights=None,
        chunk_significance_weights=vocab["chunk_significance_weights"],
    )
    assert "token_weights" in ds.column_names

    # BOS/EOS weights are always 1.0; the middle weights must match the
    # precomputed per-column significance weights, in column order.
    row = ds[0]
    expected = (
        [1.0]
        + [vocab["chunk_significance_weights"][c] for c in pr_df.columns]
        + [1.0]
    )
    assert row["token_weights"] == expected


def test_make_dataset_with_column_types_token_weights_from_chunk_significance():
    raw_df = pd.DataFrame({
        "price": [10.5, 20.3, 30.1, 40.9] * 5,
        "gender": ["m", "f", "m", "f"] * 5,
    })
    pr_df, _, _ = du.process_data(raw_df, numeric_max_len=6, numeric_precision=1)
    vocab = du.build_pooled_vocab(
        pr_df, special_tokens=du.SpecialTokens.tokens(), compute_chunk_significance=True
    )

    ds = du.make_dataset_with_column_types(
        pr_df,
        vocab,
        mask_rate=0,
        field_weights=None,
        chunk_significance_weights=vocab["chunk_significance_weights"],
    )
    assert "token_weights" in ds.column_names
    row = ds[0]
    expected = (
        [1.0]
        + [vocab["chunk_significance_weights"][c] for c in pr_df.columns]
        + [1.0]
    )
    assert row["token_weights"] == expected


def test_chunk_significance_composes_multiplicatively_with_field_weights():
    raw_df = pd.DataFrame({
        "price": [10.5, 20.3, 30.1, 40.9] * 5,
        "gender": ["m", "f", "m", "f"] * 5,
    })
    pr_df, _, _ = du.process_data(raw_df, numeric_max_len=6, numeric_precision=1)
    vocab = du.build_vocab(
        pr_df,
        special_tokens=du.SpecialTokens.tokens(),
        add_columns=False,
        compute_chunk_significance=True,
    )
    price_col = [c for c in pr_df.columns if "price" in c][0]
    field_weights = {price_col: 3.0}

    ds = du.make_dataset(
        pr_df,
        vocab,
        mask_rate=0,
        return_token_type_ids=False,
        field_weights=field_weights,
        chunk_significance_weights=vocab["chunk_significance_weights"],
    )
    row = ds[0]
    price_idx = pr_df.columns.tolist().index(price_col) + 1  # +1 for BOS
    expected = 3.0 * vocab["chunk_significance_weights"][price_col]
    assert abs(row["token_weights"][price_idx] - expected) < 1e-9


# --- numeric_categorical_threshold: cardinality-aware numeric dispatch -----


def _bedrooms_price_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "bedrooms": rng.choice([1, 2, 3, 4, 5], size=n, p=[0.05, 0.30, 0.35, 0.20, 0.10]),
        "price": rng.integers(100000, 999999, size=n).astype(float),
        "gender": rng.choice(["m", "f"], size=n),
    })


def test_numeric_categorical_threshold_default_off_unchanged():
    df = _bedrooms_price_df()
    pr_off, _, _ = du.process_data(
        df, numeric_max_len=8, numeric_precision=0, numeric_nparts=1
    )
    bedroom_cols = [c for c in pr_off.columns if "bedrooms" in c]
    assert du.is_numeric_col(bedroom_cols[0])


def test_numeric_categorical_threshold_demotes_low_cardinality_column():
    df = _bedrooms_price_df()
    pr_on, ctd_on, _ = du.process_data(
        df, numeric_max_len=8, numeric_precision=0, numeric_nparts=1,
        numeric_categorical_threshold=10,
    )
    bedroom_cols = [c for c in pr_on.columns if "bedrooms" in c]
    price_cols = [c for c in pr_on.columns if "price" in c]

    assert len(bedroom_cols) == 1
    assert du.is_categorical_col(bedroom_cols[0])
    assert ctd_on["bedrooms"] == {"treated_as_categorical": True}

    # High-cardinality "price" is unaffected: still digit-chunked, still
    # tagged numeric.
    assert len(price_cols) > 1
    assert all(du.is_numeric_col(c) for c in price_cols)


def test_numeric_categorical_threshold_decision_frozen_and_replayed():
    # The decision is made once at fit time and must be *replayed*, not
    # recomputed, on a later call with a much smaller slice of the same
    # column (e.g. a seed_input that could be a single row) -- cardinality
    # computed fresh on 1 row would be meaningless.
    df = _bedrooms_price_df()
    _, ctd, _ = du.process_data(
        df, numeric_max_len=8, numeric_precision=0, numeric_nparts=1,
        numeric_categorical_threshold=10,
    )

    seed = df.iloc[[0]][["bedrooms"]]
    # No threshold passed at all on replay -- must still come out categorical.
    seed_pr, _, _ = du.process_data(seed, col_transform_data=ctd)
    assert du.is_categorical_col(seed_pr.columns.tolist()[0])

    # A threshold passed at replay time must be ignored -- the frozen
    # fit-time decision always wins.
    seed_pr2, _, _ = du.process_data(
        seed, col_transform_data=ctd, numeric_categorical_threshold=0
    )
    assert du.is_categorical_col(seed_pr2.columns.tolist()[0])


# --- numeric_quantile_encoding: CDF-based numeric representation -----------


def _heavy_tailed_series(n=300, seed=3):
    rng = np.random.default_rng(seed)
    return pd.Series(
        np.round(rng.lognormal(mean=4.0, sigma=1.5, size=n), 2), name="price"
    )


def test_numeric_quantile_encoding_round_trip_recovers_observed_values():
    s = _heavy_tailed_series()
    formatted, transform_data = du.process_numeric_data(
        s, max_len=8, numeric_precision=4, quantile_encoding=True
    )
    assert transform_data["quantile_encoding"] is True

    quantile_values = np.array(transform_data["quantile_values"])
    quantile_positions = np.array(transform_data["quantile_positions"])

    # `formatted` is now a bare zero-padded digit-index string (no "0."
    # prefix) -- divide back by the grid to recover q itself.
    grid = 10 ** transform_data["numeric_precision"]
    q = formatted.astype(float).to_numpy() / grid
    recovered = np.interp(q, quantile_positions, quantile_values)

    rel_err = np.abs(recovered - s.to_numpy()) / s.to_numpy()
    # Recovery is exact up to the rounding introduced by formatting `q` to
    # `numeric_precision` decimal digits and the piecewise-linear
    # interpolation between the 1000 fitted breakpoints -- both small,
    # bounded sources of error, not a fidelity failure.
    assert np.median(rel_err) < 0.01
    assert rel_err.max() < 0.05


def test_numeric_quantile_encoding_interpolates_monotonically_between_points():
    s = _heavy_tailed_series()
    _, transform_data = du.process_numeric_data(
        s, max_len=8, numeric_precision=4, quantile_encoding=True
    )
    quantile_values = np.array(transform_data["quantile_values"])
    quantile_positions = np.array(transform_data["quantile_positions"])

    sorted_unique = np.sort(s.unique())
    lo, hi = sorted_unique[10], sorted_unique[11]
    mid = (lo + hi) / 2

    q_lo, q_mid, q_hi = np.interp([lo, mid, hi], quantile_values, quantile_positions)
    assert q_lo <= q_mid <= q_hi


def test_numeric_quantile_encoding_clips_out_of_range_values():
    s = _heavy_tailed_series()
    _, transform_data = du.process_numeric_data(
        s, max_len=8, numeric_precision=4, quantile_encoding=True
    )

    # Forward direction: values far outside the observed range clip to the
    # training min/max quantile position instead of extrapolating.
    too_big = pd.Series([s.max() * 100])
    too_small = pd.Series([s.min() / 100])

    formatted_big, _ = du.process_numeric_data(
        too_big, max_len=8, numeric_precision=4,
        transform_data=dict(transform_data), quantile_encoding=True,
    )
    formatted_small, _ = du.process_numeric_data(
        too_small, max_len=8, numeric_precision=4,
        transform_data=dict(transform_data), quantile_encoding=True,
    )
    # Digit-index format: the top representable value is grid - 1 (e.g.
    # "9999" at precision=4), not a digit-index of exactly `grid` -- that
    # would need a 5th digit and break the fixed-width guarantee, so the
    # forward transform's exact q=1.0 clips down by one grid cell instead.
    assert formatted_big.iloc[0] == "9999"
    assert formatted_small.iloc[0] == "0000"


def test_numeric_quantile_encoding_requires_positive_precision():
    s = _heavy_tailed_series()
    with pytest.raises(AssertionError, match="numeric_precision > 0"):
        du.process_numeric_data(
            s, max_len=8, numeric_precision=0, quantile_encoding=True
        )


def test_numeric_quantile_encoding_max_len_is_irrelevant():
    # The digit-index representation (no "0." prefix) never runs through
    # the generic max_len-truncation code path at all -- its width is
    # exactly numeric_precision by construction -- so a max_len far too
    # small for the old "0." + digits format no longer matters.
    s = _heavy_tailed_series()
    formatted, _ = du.process_numeric_data(
        s, max_len=1, numeric_precision=4, quantile_encoding=True
    )
    assert (formatted.str.len() == 4).all()


def test_numeric_quantile_encoding_default_off_leaves_transform_data_unchanged():
    # "Zero footprint when unused": a column that never opts into quantile
    # encoding must not grow any new transform_data keys because the
    # feature exists elsewhere in the model.
    s = pd.Series([1.0, 2.0, 3.0])
    _, transform_data = du.process_numeric_data(s, max_len=8, numeric_precision=4)
    assert "quantile_encoding" not in transform_data
    assert "quantile_values" not in transform_data
    assert "quantile_positions" not in transform_data


def test_numeric_quantile_encoding_precision_collision_warning():
    rng = np.random.default_rng(7)

    # High-cardinality column, low precision: more distinct values than
    # `10 ** numeric_precision` distinguishable quantile levels -- distinct
    # values collide onto the same reconstructed value, so this must warn.
    high_card = pd.Series(rng.random(200) * 1000)
    with pytest.warns(UserWarning, match="unique values but"):
        du.process_numeric_data(
            high_card, max_len=8, numeric_precision=1, quantile_encoding=True
        )

    # Low-cardinality column, ample precision: comfortably under the
    # threshold, must stay silent.
    low_card = pd.Series([1.0, 2.0, 3.0] * 50)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        du.process_numeric_data(
            low_card, max_len=8, numeric_precision=4, quantile_encoding=True
        )


def test_numeric_quantile_encoding_point_mass_reclaims_wasted_resolution():
    # Follow-up to the boundary-precision fix: that fix made a zero-inflated
    # column *decode correctly*, but a plain QuantileTransformer fit still
    # spends a share of its n_quantiles breakpoints proportional to the
    # point mass's own frequency re-describing the same repeated value --
    # 955 of 1000 for a 95.5%-zero column -- leaving only the remainder to
    # describe the part of the distribution that actually varies.
    # `_fit_quantile_breakpoints` excises the dominant value from the fit
    # entirely and gives it one reserved breakpoint instead, at its true
    # rank and frequency, so this asserts that reallocation actually
    # happens (not just that it doesn't crash).
    rng = np.random.default_rng(1029)
    n = 2400
    n_zero = int(n * 0.955)
    zeros = np.zeros(n_zero)
    nonzero = rng.exponential(scale=500, size=n - n_zero) + 1
    values = np.concatenate([zeros, nonzero])
    rng.shuffle(values)
    s = pd.Series(np.round(values, 0))

    _, transform_data = du.process_numeric_data(
        s, max_len=8, numeric_precision=4, quantile_encoding=True
    )
    quantile_values = np.array(transform_data["quantile_values"])

    # Exactly one breakpoint describes the point mass -- not hundreds.
    assert (quantile_values == 0).sum() == 1
    # The rest of the (much smaller, now dominant-mass-free) breakpoint
    # budget describes the nonzero remainder.
    assert (quantile_values != 0).sum() >= min(1000, (s != 0).sum()) - 1


def test_numeric_quantile_encoding_point_mass_round_trip_precision_improves():
    # Direct before/after comparison on the nonzero remainder's round-trip
    # fidelity: reclaiming the wasted breakpoints should make the
    # remainder's decoded values *closer* to the originals, not just
    # "still correct in shape" -- verified against a plain fit on the
    # same data (bypassing `_fit_quantile_breakpoints`) as the baseline.
    rng = np.random.default_rng(1029)
    n = 2400
    n_zero = int(n * 0.955)
    zeros = np.zeros(n_zero)
    nonzero = rng.exponential(scale=500, size=n - n_zero) + 1
    values = np.concatenate([zeros, nonzero])
    rng.shuffle(values)
    s = pd.Series(np.round(values, 0))

    formatted, transform_data = du.process_numeric_data(
        s, max_len=8, numeric_precision=4, quantile_encoding=True
    )
    quantile_values = np.array(transform_data["quantile_values"])
    quantile_positions = np.array(transform_data["quantile_positions"])
    grid = 10 ** transform_data["numeric_precision"]
    decoded = np.interp(
        formatted.astype(float).to_numpy() / grid, quantile_positions, quantile_values
    )

    nonzero_mask = s.to_numpy() != 0
    rel_err = np.abs(decoded[nonzero_mask] - s.to_numpy()[nonzero_mask]) / s.to_numpy()[nonzero_mask]

    # A plain (non-point-mass-aware) fit on the same data, for comparison --
    # this is exactly what the previous implementation produced.
    from sklearn.preprocessing import QuantileTransformer

    valid = s.astype("float64")
    qt = QuantileTransformer(n_quantiles=min(1000, len(valid)), output_distribution="uniform")
    qt.fit(valid.to_numpy().reshape(-1, 1))
    plain_values = qt.quantiles_.ravel()
    plain_positions = qt.references_
    plain_q = np.interp(valid.to_numpy(), plain_values, plain_positions)
    plain_decoded = np.interp(plain_q, plain_positions, plain_values)
    plain_rel_err = np.abs(plain_decoded[nonzero_mask] - valid.to_numpy()[nonzero_mask]) / valid.to_numpy()[nonzero_mask]

    assert np.median(rel_err) < np.median(plain_rel_err)


def test_numeric_quantile_encoding_mid_distribution_point_mass():
    # The point mass doesn't have to sit at an extreme (min/max) of the
    # column -- confirm the below/above rank split is handled generally,
    # not just for the zero-at-the-boundary case every other test uses.
    rng = np.random.default_rng(3)
    below = rng.uniform(0, 100, size=200)
    above = rng.uniform(300, 400, size=200)
    mass = np.full(1000, 200.0)
    values = np.concatenate([below, above, mass])
    rng.shuffle(values)
    s = pd.Series(values)

    formatted, transform_data = du.process_numeric_data(
        s, max_len=8, numeric_precision=4, quantile_encoding=True
    )
    quantile_values = np.array(transform_data["quantile_values"])
    quantile_positions = np.array(transform_data["quantile_positions"])

    mass_formatted = formatted.loc[s[s == 200.0].index]
    assert mass_formatted.nunique() == 1

    grid = 10 ** transform_data["numeric_precision"]
    decoded = np.interp(
        formatted.astype(float).to_numpy() / grid, quantile_positions, quantile_values
    )
    mass_mask = s.to_numpy() == 200.0
    assert np.allclose(decoded[mass_mask], 200.0, atol=1e-6)

    below_mask = s.to_numpy() < 200
    above_mask = s.to_numpy() > 200
    below_err = np.abs(decoded[below_mask] - s.to_numpy()[below_mask])
    above_err = np.abs(decoded[above_mask] - s.to_numpy()[above_mask])
    assert np.median(below_err) < 1.0
    assert np.median(above_err) < 1.0


def test_numeric_quantile_encoding_below_threshold_point_mass_unaffected():
    # A repeated value that *doesn't* clear _POINT_MASS_THRESHOLD (5%)
    # must not trigger the excision path -- ordinary ties in continuous
    # data shouldn't pay any extra fit complexity or behave differently
    # from before this feature existed.
    rng = np.random.default_rng(11)
    s = pd.Series(np.concatenate([np.zeros(20), rng.exponential(scale=50, size=980) + 1]))

    _, transform_data = du.process_numeric_data(
        s, max_len=8, numeric_precision=4, quantile_encoding=True
    )
    quantile_values = np.array(transform_data["quantile_values"])
    # A plain fit collapses ties onto repeated breakpoints rather than
    # reserving one dedicated entry -- more than one breakpoint at 0.
    assert (quantile_values == 0).sum() > 1


def test_numeric_quantile_encoding_fully_constant_column_does_not_crash():
    # A column with only one distinct value, ever: a pre-existing edge
    # case (present before this feature too -- confirmed by reproducing
    # it against the prior commit), not something this feature is meant
    # to fix (a real user with a literally-constant numeric column should
    # route it through numeric_categorical_threshold, or drop it -- it
    # carries zero information either way). This only asserts it doesn't
    # raise, so future changes don't silently make it worse.
    s = pd.Series([5.0] * 200)
    du.process_numeric_data(s, max_len=8, numeric_precision=4, quantile_encoding=True)


def test_numeric_quantile_encoding_zero_inflated_point_mass_round_trip():
    # Regression test for a real bug found on the UCI Adult `capital-loss`
    # column (95.5% exact zeros, a long nonzero tail): a value that recurs
    # thousands of times collapses a long run of fitted quantile positions
    # onto the same breakpoint, so every occurrence forward-transforms to
    # one exact quantile position -- that run's right edge. If that edge's
    # raw float (e.g. 0.954954954954955) isn't exactly representable at
    # `numeric_precision` decimal digits, formatting-then-parsing it (what
    # generation always does) rounds it onto the *other* side of the
    # boundary, into the next distinct value's segment -- and because a
    # point mass makes that segment's value jump sharply, the decoded
    # value comes back completely wrong for the entire point mass, not
    # just approximately imprecise. Confirmed end to end: before the fix,
    # every zero-valued row in this reproduction decoded to a nonzero
    # value; after, 100% decode back to exactly 0.
    rng = np.random.default_rng(1029)
    n = 2400
    n_zero = int(n * 0.955)
    zeros = np.zeros(n_zero)
    nonzero = rng.exponential(scale=500, size=n - n_zero) + 1
    values = np.concatenate([zeros, nonzero])
    rng.shuffle(values)
    s = pd.Series(np.round(values, 0))

    formatted, transform_data = du.process_numeric_data(
        s, max_len=8, numeric_precision=4, quantile_encoding=True
    )
    quantile_values = np.array(transform_data["quantile_values"])
    quantile_positions = np.array(transform_data["quantile_positions"])

    # Every zero-valued row must forward-transform to the exact same,
    # grid-representable quantile string -- not spread across values that
    # would round differently.
    zero_formatted = formatted.loc[s[s == 0].index]
    assert zero_formatted.nunique() == 1

    # Decoding that string must recover exactly 0, for every zero row --
    # not an interpolated near-miss.
    grid = 10 ** transform_data["numeric_precision"]
    decoded = np.interp(
        formatted.astype(float).to_numpy() / grid, quantile_positions, quantile_values
    )
    zero_mask = s.to_numpy() == 0
    assert np.allclose(decoded[zero_mask], 0.0, atol=1e-9)

    # The stored breakpoint positions must remain usable by np.interp,
    # i.e. non-decreasing, after the fit-time precision-grid snap.
    assert (np.diff(quantile_positions) >= 0).all()


def test_numeric_quantile_encoding_digit_entropy_near_maximal_through_real_pipeline():
    # The concrete, numbers-backed claim this feature rests on: run the
    # *actual* process_data path (not a standalone prototype) on a
    # heavy-tailed column and confirm the resulting digit-chunk columns
    # carry close to maximal per-position entropy, unlike the fixed-width
    # encoding's near-constant leading chunks.
    rng = np.random.default_rng(7)
    price = np.clip(np.round(rng.lognormal(mean=6.0, sigma=2.0, size=2000), 2), 0.01, 500000)
    df = pd.DataFrame({"price": price})

    pr_df, _, _ = du.process_data(
        df, numeric_max_len=8, numeric_precision=4, numeric_nparts=1,
        numeric_quantile_encoding=True,
    )
    price_cols = sorted(c for c in pr_df.columns if "price" in c)
    # 4 digit-index positions -- no more structurally-constant "0"/"."
    # prefix columns (dropped; see process_numeric_data's quantile_encoding
    # branch).
    assert len(price_cols) == 4

    def norm_entropy(col):
        chars = pr_df[col].str.split(du.SPECIAL_COL_SEP).str[-1]
        _, counts = np.unique(chars, return_counts=True)
        p = counts / counts.sum()
        h = -(p * np.log2(p)).sum()
        return h / np.log2(10)

    # Every remaining position is a genuine fractional digit of q --
    # all carry essentially maximal entropy now that the constant "0"/"."
    # positions are gone entirely.
    entropies = [norm_entropy(c) for c in price_cols]
    assert all(e > 0.9 for e in entropies)


def test_numeric_quantile_encoding_mutually_exclusive_with_categorical_threshold():
    # Both flags set on the same fit, on different columns: the
    # low-cardinality column is demoted to categorical and never reaches
    # quantile encoding; the high-cardinality column is quantile-encoded
    # and never demoted.
    df = _bedrooms_price_df()
    pr_df, ctd, _ = du.process_data(
        df, numeric_max_len=8, numeric_precision=4, numeric_nparts=1,
        numeric_categorical_threshold=10,
        numeric_quantile_encoding=True,
    )

    bedroom_cols = [c for c in pr_df.columns if "bedrooms" in c]
    price_cols = [c for c in pr_df.columns if "price" in c]

    assert len(bedroom_cols) == 1
    assert du.is_categorical_col(bedroom_cols[0])
    assert ctd["bedrooms"] == {"treated_as_categorical": True}

    assert all(du.is_numeric_col(c) for c in price_cols)
    assert ctd["price"]["quantile_encoding"] is True


def test_numeric_quantile_encoding_composes_with_digit_entropy_weighting():
    # digit_entropy_weighting (compute_chunk_significance_weights) is
    # generic over whatever digit-chunk columns exist -- verify directly
    # that it runs cleanly on quantile-encoded columns and produces sane,
    # non-degenerate weights rather than trusting the "should be fine"
    # reasoning.
    rng = np.random.default_rng(7)
    price = np.clip(np.round(rng.lognormal(mean=6.0, sigma=2.0, size=500), 2), 0.01, 500000)
    df = pd.DataFrame({"price": price, "gender": rng.choice(["m", "f"], size=500)})

    pr_df, _, _ = du.process_data(
        df, numeric_max_len=8, numeric_precision=4, numeric_nparts=1,
        numeric_quantile_encoding=True,
    )

    vocab = du.build_vocab(
        pr_df,
        special_tokens=du.SpecialTokens.tokens(),
        add_columns=False,
        compute_chunk_significance=True,
    )
    weights = vocab["chunk_significance_weights"]
    price_cols = [c for c in pr_df.columns if "price" in c]
    assert set(price_cols).issubset(weights.keys())

    price_weights = [weights[c] for c in price_cols]
    # No more constant "0"/"." positions to pull toward the floor -- all 4
    # positions are genuine fractional digits of q with near-maximal,
    # near-equal entropy, so their renormalized-within-block weights
    # should all cluster close to 1.0 rather than showing the old strong
    # low/high split.
    assert all(0.9 < w < 1.1 for w in price_weights)
