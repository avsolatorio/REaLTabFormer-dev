"""Tests for rtf_datacollator.py's AnyOrderColumnCollator -- the any-order
training collator for REaLTabFormerV2's `any_order` feature (beta). See
/Users/avsolatorio/.claude/plans/snappy-swimming-hickey.md for the design.
"""
import itertools

import torch

from realtabformer.rtf_datacollator import AnyOrderColumnCollator

COLUMN_BLOCKS = [["price", [0, 1]], ["age", [2]], ["gender", [3]]]
BOS, EOS = -1, -2


def _make_features(n: int):
    # Row layout: [BOS, price_00=10, price_01=11, age_00=20, gender=30, EOS]
    row = torch.tensor([BOS, 10, 11, 20, 30, EOS])
    type_row = torch.tensor([99, 100, 100, 200, 300, 99])
    return [
        {
            "input_ids": row.clone(),
            "labels": row.clone(),
            "token_type_ids": type_row.clone(),
        }
        for _ in range(n)
    ]


def test_any_order_collator_leaves_bos_eos_untouched():
    collator = AnyOrderColumnCollator(column_blocks=COLUMN_BLOCKS, seed=1)
    batch = collator(_make_features(20))

    assert (batch["input_ids"][:, 0] == BOS).all()
    assert (batch["input_ids"][:, -1] == EOS).all()
    assert (batch["token_type_ids"][:, 0] == 99).all()
    assert (batch["token_type_ids"][:, -1] == 99).all()


def test_any_order_collator_preserves_row_content_as_a_permutation():
    collator = AnyOrderColumnCollator(column_blocks=COLUMN_BLOCKS, seed=2)
    batch = collator(_make_features(20))

    for i in range(20):
        middle = batch["input_ids"][i, 1:-1].tolist()
        assert sorted(middle) == [10, 11, 20, 30]


def test_any_order_collator_keeps_numeric_chunks_adjacent_and_ordered():
    # price's two digit chunks (10, 11) must always move together, in that
    # relative order -- only *block* order is permuted, not chunk order.
    collator = AnyOrderColumnCollator(column_blocks=COLUMN_BLOCKS, seed=3)
    batch = collator(_make_features(30))

    for i in range(30):
        middle = batch["input_ids"][i, 1:-1].tolist()
        idx10 = middle.index(10)
        assert middle[idx10 + 1] == 11, middle


def test_any_order_collator_token_type_ids_follow_their_value():
    collator = AnyOrderColumnCollator(column_blocks=COLUMN_BLOCKS, seed=4)
    batch = collator(_make_features(30))

    val_to_type = {10: 100, 11: 100, 20: 200, 30: 300}
    for i in range(30):
        middle = batch["input_ids"][i, 1:-1].tolist()
        tmiddle = batch["token_type_ids"][i, 1:-1].tolist()
        for v, ty in zip(middle, tmiddle):
            assert val_to_type[v] == ty


def test_any_order_collator_labels_permuted_consistently_with_input_ids():
    # Use distinct label values so a mismatched permutation would be
    # detectable, not accidentally still-correct.
    row = torch.tensor([BOS, 10, 11, 20, 30, EOS])
    label_row = torch.tensor([BOS, -100, 11, -100, 30, EOS])  # some masked
    type_row = torch.tensor([99, 100, 100, 200, 300, 99])
    features = [
        {"input_ids": row.clone(), "labels": label_row.clone(), "token_type_ids": type_row.clone()}
        for _ in range(30)
    ]
    collator = AnyOrderColumnCollator(column_blocks=COLUMN_BLOCKS, seed=5)
    batch = collator(features)

    for i in range(30):
        in_mid = batch["input_ids"][i, 1:-1].tolist()
        lab_mid = batch["labels"][i, 1:-1].tolist()
        for v, lab in zip(in_mid, lab_mid):
            if v == 10:
                assert lab == -100
            elif v == 11:
                assert lab == 11
            elif v == 20:
                assert lab == -100
            elif v == 30:
                assert lab == 30


def test_any_order_collator_token_weights_permuted_when_present():
    row = torch.tensor([BOS, 10, 11, 20, 30, EOS])
    type_row = torch.tensor([99, 100, 100, 200, 300, 99])
    weight_row = torch.tensor([1.0, 2.0, 2.0, 3.0, 4.0, 1.0])
    features = [
        {
            "input_ids": row.clone(),
            "labels": row.clone(),
            "token_type_ids": type_row.clone(),
            "token_weights": weight_row.clone(),
        }
        for _ in range(30)
    ]
    collator = AnyOrderColumnCollator(column_blocks=COLUMN_BLOCKS, seed=6)
    batch = collator(features)

    val_to_weight = {10: 2.0, 11: 2.0, 20: 3.0, 30: 4.0}
    for i in range(30):
        in_mid = batch["input_ids"][i, 1:-1].tolist()
        w_mid = batch["token_weights"][i, 1:-1].tolist()
        for v, w in zip(in_mid, w_mid):
            assert val_to_weight[v] == w


def test_any_order_collator_different_rows_get_different_permutations():
    collator = AnyOrderColumnCollator(column_blocks=COLUMN_BLOCKS, seed=7)
    batch = collator(_make_features(60))

    distinct_orders = {tuple(batch["input_ids"][i, 1:-1].tolist()) for i in range(60)}
    # 3 blocks -> 3! = 6 possible orders; with 60 draws we should see more
    # than one, and ideally most/all of them.
    assert len(distinct_orders) > 1
    all_possible = {
        tuple(
            itertools.chain.from_iterable(
                dict(COLUMN_BLOCKS)[name] for name in perm
            )
        )
        for perm in itertools.permutations(["price", "age", "gender"])
    }
    # Sanity: every order we saw is one of the theoretically possible ones
    # (translated from block-index space to value space for comparison).
    value_by_index = {0: 10, 1: 11, 2: 20, 3: 30}
    all_possible_values = {tuple(value_by_index[i] for i in order) for order in all_possible}
    assert distinct_orders <= all_possible_values


def test_any_order_collator_consecutive_calls_draw_different_permutations():
    # The RNG must advance across __call__s, not reset/repeat -- that's
    # the whole point of doing this per-batch rather than baking one
    # permutation into the dataset once. With 50 identical-input rows per
    # call, exact elementwise equality across two calls would require the
    # RNG to have produced the identical sequence of 50 permutations
    # twice in a row -- astronomically unlikely if it actually advanced.
    collator = AnyOrderColumnCollator(column_blocks=COLUMN_BLOCKS, seed=8)
    batch1 = collator(_make_features(50))
    batch2 = collator(_make_features(50))

    assert not torch.equal(batch1["input_ids"], batch2["input_ids"])
