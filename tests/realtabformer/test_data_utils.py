import json
import random

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
