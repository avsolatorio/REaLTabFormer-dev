"""Tests for rtf_rl_relational.py -- RL fine-tuning building blocks for a REaLTabFormer RELATIONAL
model. Ported and adapted from the GMD survey-synthesis investigation's own
gmd_rl_relational_selftest.py (see the independent synthetic-gmd repo's lab notebook, G26-G60).

`relational_log_prob_of_sequences` carries the same "single easiest place for a silent bug" risk as
`rtf_rl.log_prob_of_sequences` (see that module's docstring), compounded by a real off-by-one
convention difference from the tabular case (see this module's own docstring) -- so it gets the
same enumerate-and-check-probability-mass treatment here, not just a round-trip smoke test.
"""
import itertools
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import GPT2Config

from realtabformer import REaLTabFormer
from realtabformer import rtf_rl_relational as R
from realtabformer.data_utils import SpecialTokens, process_data
from realtabformer.data_utils.dataset import make_dataset
from realtabformer.rtf_sampler import RelationalSampler


def _fit_toy_relational(seed=0):
    rng = np.random.default_rng(seed)
    n_parents = 60
    parent = pd.DataFrame({"pid": [f"h{i}" for i in range(n_parents)], "a": rng.choice(list("xy"), n_parents)})
    rows = []
    for i in range(n_parents):
        k = rng.integers(1, 3)  # 1-2 children, so the model should assign little mass to >=3
        for _ in range(k):
            rows.append(dict(pid=f"h{i}", b=rng.choice(list("pq"))))
    child = pd.DataFrame(rows)

    tmp = tempfile.mkdtemp()
    pm = REaLTabFormer(model_type="tabular", tabular_config=GPT2Config(n_layer=1, n_embd=16, n_head=2),
                        epochs=2, batch_size=8, checkpoints_dir=tmp + "/p", random_state=seed)
    pm.fit(parent.drop(columns="pid"), device="cpu")
    pm.save(tmp + "/parent_saved")
    ppath = tmp + "/parent_saved/" + os.listdir(tmp + "/parent_saved")[0]
    cm = REaLTabFormer(model_type="relational", parent_realtabformer_path=ppath, output_max_length=None,
                        train_size=1.0, epochs=3, batch_size=8, checkpoints_dir=tmp + "/c", random_state=seed)
    cm.fit(df=child, in_df=parent, join_on="pid", device="cpu")
    return cm, parent


def _enumerate_sequences(eos_id, bos_id, bmem_id, emem_id, col_values, max_members):
    out = []
    decoder_start = eos_id
    for k in range(max_members + 1):
        prefix = [decoder_start, bos_id]
        for member_values in itertools.product(col_values, repeat=k):
            seq = list(prefix)
            for v in member_values:
                seq += [bmem_id, v, emem_id]
            seq += [eos_id]
            out.append(tuple(seq))
    return out


def _relational_fixture():
    cm, parent = _fit_toy_relational(seed=0)
    sampler = RelationalSampler.sampler_from_model(cm, device="cpu")
    t2i = cm.vocab["decoder"]["token2id"]
    eos_id, bos_id = t2i[SpecialTokens.EOS], t2i[SpecialTokens.BOS]
    bmem_id, emem_id = t2i[SpecialTokens.BMEM], t2i[SpecialTokens.EMEM]
    col_values = cm.col_idx_ids[0]

    max_members = 5  # generous cap vs a model trained on data with max 2 children per parent
    seqs = _enumerate_sequences(eos_id, bos_id, bmem_id, emem_id, col_values, max_members)
    max_len = max(len(s) for s in seqs)
    padded = [list(s) + [t2i[SpecialTokens.PAD]] * (max_len - len(s)) for s in seqs]
    dec = torch.tensor(padded)

    in_df, _, _ = process_data(parent.drop(columns="pid").iloc[[0]], col_transform_data=cm.in_col_transform_data)
    enc_ds = make_dataset(in_df, cm.vocab["encoder"], seed=0)
    enc_ids = torch.tensor([enc_ds["input_ids"][0]]).repeat(dec.shape[0], 1)

    mask = R.build_relational_mask(sampler, max_steps=max_len, vocab_size=cm.model.config.decoder.vocab_size, device=torch.device("cpu"))
    logp = R.relational_log_prob_of_sequences(cm.model, enc_ids, dec, mask, eos_id)
    return cm, parent, sampler, seqs, logp


def test_relational_log_prob_sums_close_to_one():
    _, _, _, seqs, logp = _relational_fixture()
    total_mass = float(torch.exp(logp).sum())
    assert 0.95 <= total_mass <= 1.0001, (
        f"expected mass close to 1.0 (shortfall = tail beyond the enumeration cap), got {total_mass}"
    )


def test_relational_log_prob_matches_empirical_sampling():
    cm, parent, sampler, seqs, logp = _relational_fixture()
    t2i = cm.vocab["decoder"]["token2id"]
    eos_id = t2i[SpecialTokens.EOS]

    in_df, _, _ = process_data(parent.drop(columns="pid").iloc[[0]], col_transform_data=cm.in_col_transform_data)
    enc_ds = make_dataset(in_df, cm.vocab["encoder"], seed=0)
    enc_ids = torch.tensor([enc_ds["input_ids"][0]])

    n_samples = 20000
    torch.manual_seed(1)
    raw = sampler._generate(
        device=torch.device("cpu"), as_numpy=True, constrain_tokens_gen=True, inputs=enc_ids,
        do_sample=True, num_return_sequences=n_samples, suppress_tokens=None,
        bos_token_id=t2i[SpecialTokens.BOS], pad_token_id=t2i[SpecialTokens.PAD], eos_token_id=eos_id,
    )

    def trim(row):
        row = list(row)
        for i in range(1, len(row)):
            if row[i] == eos_id:
                return tuple(row[: i + 1])
        return tuple(row)

    emp_counts = Counter(trim(r) for r in raw)
    exact_probs = {s: float(torch.exp(logp[i])) for i, s in enumerate(seqs)}

    max_diff = 0.0
    for s, p in sorted(exact_probs.items(), key=lambda kv: -kv[1])[:8]:
        emp = emp_counts.get(s, 0) / n_samples
        max_diff = max(max_diff, abs(p - emp))
    assert max_diff < 0.02, f"exact and empirical probabilities should agree closely, max diff {max_diff}"


def test_encode_sample_decode_round_trip_gives_one_row_group_per_parent():
    cm, parent = _fit_toy_relational(seed=1)
    sampler = RelationalSampler.sampler_from_model(cm, device="cpu")
    t2i = cm.vocab["decoder"]["token2id"]
    eos_id, bos_id, pad_id = t2i[SpecialTokens.EOS], t2i[SpecialTokens.BOS], t2i[SpecialTokens.PAD]

    batch_parent = parent.iloc[:10].reset_index(drop=True)
    enc_ids = R.encode_parent_batch(batch_parent.drop(columns="pid"), cm.in_col_transform_data, cm.vocab["encoder"], torch.device("cpu"))
    assert enc_ids.shape[0] == len(batch_parent)

    raw = R.sample_child_rollouts(sampler, cm.model, torch.device("cpu"), enc_ids, eos_id, bos_id, pad_id, cm.relational_max_length, seed=0)
    child_batch = R.decode_child_batch(sampler, raw, cm.vocab["decoder"], batch_parent["pid"], key="pid")
    assert "pid" in child_batch.columns
    assert set(child_batch["pid"]).issubset(set(batch_parent["pid"]))


def test_decode_child_batch_default_key_is_id():
    cm, parent = _fit_toy_relational(seed=2)
    sampler = RelationalSampler.sampler_from_model(cm, device="cpu")
    t2i = cm.vocab["decoder"]["token2id"]
    eos_id, bos_id, pad_id = t2i[SpecialTokens.EOS], t2i[SpecialTokens.BOS], t2i[SpecialTokens.PAD]

    batch_parent = parent.iloc[:5].reset_index(drop=True)
    enc_ids = R.encode_parent_batch(batch_parent.drop(columns="pid"), cm.in_col_transform_data, cm.vocab["encoder"], torch.device("cpu"))
    raw = R.sample_child_rollouts(sampler, cm.model, torch.device("cpu"), enc_ids, eos_id, bos_id, pad_id, cm.relational_max_length, seed=0)
    child_batch = R.decode_child_batch(sampler, raw, cm.vocab["decoder"], batch_parent["pid"])
    assert "id" in child_batch.columns
