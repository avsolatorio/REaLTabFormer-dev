"""Tests for rtf_rl.py -- RL fine-tuning building blocks for a REaLTabFormer tabular model.
Ported and adapted from the GMD (Vietnam VHLSS household-survey synthesis) investigation's own
gmd_rl_selftest.py (see the independent synthetic-gmd repo's lab notebook, G17-G60), which is where
this module's mechanisms were originally built and validated before being promoted here.

`log_prob_of_sequences` is flagged in its own docstring as the single easiest place for a silent,
invisible bug (a policy-gradient direction that's mathematically wrong with no visible symptom) --
`test_log_prob_of_sequences_defines_a_valid_probability_distribution` is the correctness gate for
it: enumerate every possible sequence for a tiny toy schema and check the probabilities sum to
EXACTLY 1, rather than trust it because "it looks reasonable".
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import GPT2Config

from realtabformer import REaLTabFormer
from realtabformer import rtf_rl
from realtabformer.data_utils import SpecialTokens
from realtabformer.rtf_sampler import TabularSampler


def _ols_fit(X, y):
    Xc = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    return beta, y - Xc @ beta


def _build_toy_model(tmp_path, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"a": rng.choice(list("xy"), 200), "b": rng.choice(list("pqr"), 200)})
    model = REaLTabFormer(
        model_type="tabular", epochs=1, batch_size=16, checkpoints_dir=str(tmp_path / "ckpt"),
        tabular_config=GPT2Config(n_layer=1, n_embd=16, n_head=2),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.fit(df, device=device, n_critic=0)
    return model, device


def test_log_prob_of_sequences_defines_a_valid_probability_distribution():
    with tempfile.TemporaryDirectory() as tmp:
        model, device = _build_toy_model(Path(tmp))
        t2i = model.vocab["token2id"]
        bos, eos = t2i[SpecialTokens.BOS], t2i[SpecialTokens.EOS]
        col0_ids, col1_ids = model.col_idx_ids[0], model.col_idx_ids[1]
        seqs = torch.tensor([[bos, a, b, eos] for a in col0_ids for b in col1_ids]).to(device)
        vocab_size = model.model.config.vocab_size
        max_steps = max(model.tabular_max_length, len(model.col_idx_ids))

        with torch.no_grad():
            logp = rtf_rl.log_prob_of_sequences(model.model, seqs, model.col_idx_ids, eos, vocab_size, max_steps)
        probs = logp.exp().cpu().numpy()
        total = probs.sum()
        assert abs(total - 1.0) < 1e-4, f"probabilities sum to {total}, not 1.0 -- the mask does not match"
        assert (probs >= 0).all() and (probs <= 1.0 + 1e-6).all()


def test_log_prob_of_sequences_agrees_with_empirical_sampling_frequency():
    with tempfile.TemporaryDirectory() as tmp:
        model, device = _build_toy_model(Path(tmp))
        t2i = model.vocab["token2id"]
        bos, eos = t2i[SpecialTokens.BOS], t2i[SpecialTokens.EOS]
        col0_ids, col1_ids = model.col_idx_ids[0], model.col_idx_ids[1]
        seqs = torch.tensor([[bos, a, b, eos] for a in col0_ids for b in col1_ids]).to(device)
        vocab_size = model.model.config.vocab_size
        max_steps = max(model.tabular_max_length, len(model.col_idx_ids))
        with torch.no_grad():
            probs = rtf_rl.log_prob_of_sequences(
                model.model, seqs, model.col_idx_ids, eos, vocab_size, max_steps
            ).exp().cpu().numpy()

        sampler = TabularSampler.sampler_from_model(model, device=device)
        torch.manual_seed(0)
        n_samples = 20000
        raw = sampler._generate(
            device=torch.device(device), as_numpy=True, constrain_tokens_gen=True,
            inputs=torch.tensor([[bos]], device=device), do_sample=True,
            max_length=model.tabular_max_length, num_return_sequences=n_samples,
            bos_token_id=bos, pad_token_id=t2i[SpecialTokens.PAD], eos_token_id=eos,
            suppress_tokens=None, forced_decoder_ids=None,
        )
        raw_t = raw[:, :4]
        seq_key = {tuple(s.tolist()): p for s, p in zip(seqs.cpu(), probs)}
        emp = {}
        for row in raw_t.tolist():
            key = tuple(row)
            emp[key] = emp.get(key, 0) + 1
        max_diff = max(abs(exact_p - emp.get(key, 0) / n_samples) for key, exact_p in seq_key.items())
        assert max_diff < 0.02, f"exact log-prob and empirical sampling frequency disagree by {max_diff:.4f}"


def test_grouped_relative_advantage_reduces_to_pooled_at_one_group():
    torch.manual_seed(0)
    rewards = torch.randn(12)
    pooled = rtf_rl.group_relative_advantage(rewards)
    grouped = rtf_rl.grouped_relative_advantage(rewards, group_size=len(rewards))
    assert torch.allclose(pooled, grouped, atol=1e-6)


def test_grouped_relative_advantage_hand_computed_and_isolates_groups():
    rewards = torch.tensor([1.0, 2.0, 3.0, 10.0, 10.0, 10.0])
    adv = rtf_rl.grouped_relative_advantage(rewards, group_size=3, eps=1e-8)
    expected_a_std = (2.0 / 3.0) ** 0.5
    assert abs(adv[0].item() - (1.0 - 2.0) / (expected_a_std + 1e-8)) < 1e-5
    assert abs(adv[1].item() - 0.0) < 1e-5
    assert abs(adv[2].item() - (3.0 - 2.0) / (expected_a_std + 1e-8)) < 1e-5
    assert torch.allclose(adv[3:], torch.zeros(3), atol=1e-3)

    pooled = rtf_rl.group_relative_advantage(rewards)
    assert pooled[3].item() > 0.5, "pooled advantage should treat group B as a notable outlier"


def test_group_relative_advantage_is_inert_to_a_constant_added_to_every_reward():
    torch.manual_seed(0)
    r = torch.randn(32)
    adv_plain = rtf_rl.group_relative_advantage(r)
    adv_shifted = rtf_rl.group_relative_advantage(r + 5.0)
    assert torch.allclose(adv_plain, adv_shifted, atol=1e-5)


def test_grpo_loss_sample_weight_none_matches_old_behaviour_exactly():
    torch.manual_seed(0)
    new_logp = torch.randn(16, requires_grad=True)
    ref_logp = torch.randn(16)
    advantage = torch.randn(16)
    out_none = rtf_rl.grpo_loss(new_logp, ref_logp, advantage, kl_coef=0.1)
    expected_policy = -(advantage.detach() * new_logp).mean()
    expected_kl = (new_logp - ref_logp).mean()
    assert torch.allclose(out_none["policy_loss"], expected_policy.detach(), atol=1e-9)
    assert torch.allclose(out_none["kl"], expected_kl.detach(), atol=1e-9)


def test_grpo_loss_sample_weight_scale_invariant_and_hand_computed():
    new_logp = torch.tensor([1.0, 2.0], requires_grad=True)
    ref_logp = torch.tensor([0.5, 0.5])
    advantage = torch.tensor([1.0, -1.0])
    w = torch.tensor([1.0, 3.0])
    out = rtf_rl.grpo_loss(new_logp, ref_logp, advantage, kl_coef=0.0, sample_weight=w)
    assert abs(out["policy_loss"].item() - 1.25) < 1e-6
    out_scaled = rtf_rl.grpo_loss(new_logp, ref_logp, advantage, kl_coef=0.0, sample_weight=w * 1000.0)
    assert abs(out["policy_loss"].item() - out_scaled["policy_loss"].item()) < 1e-4


def test_weighted_quantile_matches_unweighted_at_uniform_weights():
    rng = np.random.default_rng(0)
    v = rng.normal(size=500)
    q = np.array([0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    got = rtf_rl._weighted_quantile(v, np.ones_like(v), q)
    want = np.quantile(v, q, method="hazen")
    assert np.allclose(got, want, atol=1e-9)
    not_default = np.quantile(v, q)
    assert not np.allclose(got, not_default, atol=1e-6)


def test_weighted_quantile_hand_computed():
    v = np.array([0.0, 1.0, 2.0])
    w = np.array([1.0, 1.0, 8.0])
    median = rtf_rl._weighted_quantile(v, w, np.array([0.5]))[0]
    assert median > 1.5
    assert np.quantile(v, 0.5) == 1.0


def test_weighted_mean_std_matches_unweighted_and_hand_computed():
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    m0, s0 = rtf_rl.weighted_mean_std(v)
    assert abs(m0 - v.mean()) < 1e-12 and abs(s0 - pd.Series(v).std()) < 1e-9
    assert abs(s0 - v.std()) > 1e-6  # ddof=1 vs ddof=0 must actually differ

    m, s = rtf_rl.weighted_mean_std(np.array([0.0, 10.0]), np.array([1.0, 9.0]))
    assert abs(m - 9.0) < 1e-9

    m_u, s_u = rtf_rl.weighted_mean_std(v, weights=np.full(len(v), 7.0))
    assert abs(m_u - m0) < 1e-9 and abs(s_u - s0) < 1e-9


def test_weighted_value_counts_matches_unweighted_at_weights_none():
    labels = pd.Series(["a", "a", "b", "a", "b", "c"])
    got = rtf_rl._weighted_value_counts(labels, weights=None)
    assert got == labels.value_counts(normalize=True).to_dict()


def test_weighted_value_counts_hand_computed():
    labels = pd.Series(["a", "a", "b"])
    weights = pd.Series([1.0, 1.0, 8.0])
    got = rtf_rl._weighted_value_counts(labels, weights)
    assert abs(got["a"] - 0.2) < 1e-9 and abs(got["b"] - 0.8) < 1e-9


def test_compute_target_stats_weighted_vs_unweighted_on_a_designed_case():
    n_lo, n_hi = 40, 10
    df = pd.DataFrame({
        "x": np.concatenate([np.zeros(n_lo), np.full(n_hi, 10.0)]),
        "cat": ["p"] * n_lo + ["q"] * n_hi,
    })
    w = pd.Series(np.concatenate([np.ones(n_lo), np.full(n_hi, 100.0)]))

    unweighted_before = rtf_rl.compute_target_stats(df, columns=["x", "cat"])
    unweighted_after = rtf_rl.compute_target_stats(df, columns=["x", "cat"], weights=None)
    assert unweighted_before == unweighted_after

    weighted = rtf_rl.compute_target_stats(df, columns=["x", "cat"], weights=w)
    assert unweighted_before["col"]["cat"]["p"] > unweighted_before["col"]["cat"]["q"]
    assert weighted["col"]["cat"]["q"] > weighted["col"]["cat"]["p"]
    assert abs(weighted["col"]["cat"]["q"] - 1000 / 1040) < 1e-6


def test_marginal_joint_reward_exact_values():
    target = dict(col={"a": {"x": 0.5, "y": 0.5}}, pair={}, edges={"a": None})
    batch = pd.DataFrame({"a": ["x"] * 8 + ["y"] * 2})
    r = rtf_rl.marginal_joint_reward(batch, target, columns=["a"], pairs=[])
    assert np.allclose(r[:8], -0.3, atol=1e-9)
    assert np.allclose(r[8:], 0.3, atol=1e-9)

    target2 = dict(
        col={"a": {"x": 0.5, "y": 0.5}, "b": {"p": 0.5, "q": 0.5}},
        pair={("a", "b"): {("x", "p"): 0.5, ("y", "q"): 0.5}},
        edges={"a": None, "b": None},
    )
    batch2 = pd.DataFrame({"a": ["x", "y", "x"], "b": ["p", "q", "q"]})  # last row: never-seen (x,q) combo
    r2 = rtf_rl.marginal_joint_reward(batch2, target2, columns=[], pairs=[("a", "b")])
    assert r2[2] < 0


def test_ema_frequency_tracker_reduces_noise_vs_raw_batch():
    rng = np.random.default_rng(7)
    true_p = np.array([0.40, 0.25, 0.15, 0.12, 0.08])
    cats = ["a", "b", "c", "d", "e"]
    target = dict(col={"x": dict(zip(cats, true_p))}, pair={}, edges={"x": None})

    def l1_to_true(freq):
        return float(sum(abs(freq.get(c, 0.0) - p) for c, p in zip(cats, true_p)))

    tracker = rtf_rl.EMAFrequencyTracker(decay=0.95)
    raw_errors, ema_errors = [], []
    for _ in range(200):
        batch = pd.DataFrame({"x": rng.choice(cats, size=8, p=true_p)})
        raw_errors.append(l1_to_true(batch["x"].value_counts(normalize=True).to_dict()))
        tracker.update(batch, columns=["x"], pairs=[], target=target)
        ema_errors.append(l1_to_true(tracker.col_freq["x"]))

    raw_median, ema_median = np.median(raw_errors[-50:]), np.median(ema_errors[-50:])
    assert ema_median < raw_median * 0.5


def test_smoothed_reward_exact_values_and_neutral_prior():
    target = dict(col={"a": {"x": 0.5, "y": 0.5}}, pair={}, edges={"a": None})
    batch = pd.DataFrame({"a": ["x"] * 8 + ["y"] * 2})
    tracker = rtf_rl.EMAFrequencyTracker(decay=0.9)

    r0 = rtf_rl.smoothed_marginal_joint_reward(batch, target, tracker, columns=["a"], pairs=[])
    assert np.allclose(r0, 0.0)

    tracker.col_freq["a"] = {"x": 0.8, "y": 0.2}
    r1 = rtf_rl.smoothed_marginal_joint_reward(batch, target, tracker, columns=["a"], pairs=[])
    assert np.allclose(r1[:8], -0.3, atol=1e-9) and np.allclose(r1[8:], 0.3, atol=1e-9)


def test_tail_aware_bins_give_the_reward_resolution_at_the_99th_percentile():
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"welfare": rng.lognormal(mean=10, sigma=1.0, size=5000)})
    true_q99 = real["welfare"].quantile(0.99)

    target_plain = rtf_rl.compute_target_stats(real, columns=["welfare"], pairs=[])
    target_tail = rtf_rl.compute_target_stats(real, columns=["welfare"], pairs=[], tail_aware_columns=["welfare"])
    assert not np.isclose(target_plain["edges"]["welfare"], true_q99).any()
    assert np.isclose(target_tail["edges"]["welfare"], true_q99).any()

    p91, p999 = real["welfare"].quantile(0.91), real["welfare"].quantile(0.999)
    probe = pd.DataFrame({"welfare": [p91, p999]})
    r_plain = rtf_rl.marginal_joint_reward(probe, target_plain, columns=["welfare"], pairs=[])
    assert np.isclose(r_plain[0], r_plain[1]), "plain deciles should NOT distinguish a p91 from a p99.9 sample"
    r_tail = rtf_rl.marginal_joint_reward(probe, target_tail, columns=["welfare"], pairs=[])
    assert not np.isclose(r_tail[0], r_tail[1]), "tail-aware bins should distinguish them"


def test_tail_boost_reweights_only_rare_bins_and_is_a_no_op_at_1x():
    target = dict(col={"a": {"bulk": 0.9, "mid": 0.08, "rare": 0.02}}, pair={}, edges={"a": None})
    batch = pd.DataFrame({"a": ["bulk", "bulk", "mid", "rare"]})
    base = rtf_rl.marginal_joint_reward(batch, target, columns=["a"], pairs=[])
    boosted = rtf_rl.marginal_joint_reward(
        batch, target, columns=["a"], pairs=[], tail_boost_columns=["a"], tail_boost=5.0, tail_freq_cutoff=0.02,
    )
    assert np.allclose(boosted[:3], base[:3], atol=1e-9)
    assert np.isclose(boosted[3], 5.0 * base[3], atol=1e-9)
    assert base[3] != 0.0

    noop = rtf_rl.marginal_joint_reward(batch, target, columns=["a"], pairs=[], tail_boost_columns=["a"], tail_boost=1.0)
    assert np.allclose(noop, base, atol=1e-9)


def test_regression_influence_reward_hand_computed_direction():
    df = pd.DataFrame({"x1": [1.0, 1.0, 1.0, 1.0], "x2": [0.0, 0.0, 0.0, 0.0], "y": [5.0, 5.0, -5.0, -5.0]})
    target_beta = np.array([0.0, 10.0, 0.0])
    X, y = df[["x1", "x2"]].astype(float).to_numpy(), df["y"].to_numpy()
    reward = rtf_rl.regression_influence_reward(X, y, target_beta=target_beta)
    assert reward[0] > reward[2] and reward[1] > reward[3]
    assert np.isclose(reward[0], reward[1]) and np.isclose(reward[2], reward[3])


def _toy_regression_batch(n, seed, beta_x1):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 2.0 + beta_x1 * x1 - 1.0 * x2 + rng.normal(scale=0.3, size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def test_regression_influence_reward_zero_when_intercept_only_gap_and_weight_intercept_false():
    df = _toy_regression_batch(500, seed=0, beta_x1=3.0)
    X, y = df[["x1", "x2"]].astype(float).to_numpy(), df["y"].to_numpy()
    beta_batch, _ = _ols_fit(X, y)
    target_beta = beta_batch.copy()
    target_beta[0] += 5.0
    reward = rtf_rl.regression_influence_reward(X, y, target_beta=target_beta)
    assert np.allclose(reward, 0.0, atol=1e-8)


def test_regression_influence_reward_causally_shifts_coefficient_toward_target():
    batch = _toy_regression_batch(2000, seed=1, beta_x1=3.0)
    X, y = batch[["x1", "x2"]].astype(float).to_numpy(), batch["y"].to_numpy()
    beta_batch, _ = _ols_fit(X, y)
    target_beta = beta_batch.copy()
    target_beta[1] += 3.0

    reward = rtf_rl.regression_influence_reward(X, y, target_beta=target_beta)
    order = np.argsort(reward)
    bottom_idx, top_idx = order[:200], order[-200:]

    def refit_with_upweighted(idx):
        boosted = pd.concat([batch, batch.iloc[idx]] * 5, ignore_index=True)
        b, _ = _ols_fit(boosted[["x1", "x2"]].astype(float).to_numpy(), boosted["y"].to_numpy())
        return b

    beta_top = refit_with_upweighted(top_idx)
    beta_bottom = refit_with_upweighted(bottom_idx)
    gap_before = abs(beta_batch[1] - target_beta[1])
    gap_top = abs(beta_top[1] - target_beta[1])
    gap_bottom = abs(beta_bottom[1] - target_beta[1])
    assert gap_top < gap_before
    assert gap_top < gap_bottom


def _sample_sequences(sampler, model, device, n, seed):
    t2i = model.vocab["token2id"]
    torch.manual_seed(seed)
    raw = sampler._generate(
        device=torch.device(device), as_numpy=False, constrain_tokens_gen=True,
        inputs=torch.tensor([[t2i[SpecialTokens.BOS]]], device=device), do_sample=True,
        max_length=model.tabular_max_length, num_return_sequences=n,
        bos_token_id=t2i[SpecialTokens.BOS], pad_token_id=t2i[SpecialTokens.PAD],
        eos_token_id=t2i[SpecialTokens.EOS], suppress_tokens=None, forced_decoder_ids=None,
    )
    return raw[:, : model.tabular_max_length]


def test_leave_one_out_reward_grpo_end_to_end_corrects_a_known_marginal_miscalibration():
    """Same falsifiable end-to-end shape `test_marginal_joint_reward`'s own end-to-end test uses
    (see the ported gmd_rl_selftest.py history this module was promoted from): fit a tiny,
    deliberately undertrained model on data where column "a" is EXACTLY 50/50, run real GRPO rounds
    using `leave_one_out_marginal_reward` (not `marginal_joint_reward`) as the reward, and confirm
    the full mechanism -- sample, reward, `group_relative_advantage`, KL-regularised `grpo_loss`,
    optimizer step -- moves the generated marginal toward the true 50/50 split. Proves the new
    reward works correctly plugged into the SAME verified GRPO loop, not just correct in isolation.
    """
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(5)
        df = pd.DataFrame({"a": rng.choice(list("xy"), 40), "b": rng.choice(list("pqr"), 40)})
        target = rtf_rl.compute_target_stats(df, columns=["a", "b"], pairs=[("a", "b")])

        model = REaLTabFormer(
            model_type="tabular", epochs=1, batch_size=8, checkpoints_dir=str(Path(tmp) / "ckpt"),
            tabular_config=GPT2Config(n_layer=1, n_embd=16, n_head=2), random_state=5,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.fit(df, device=device, n_critic=0)
        sampler = TabularSampler.sampler_from_model(model, device=device)
        t2i = model.vocab["token2id"]
        eos = t2i[SpecialTokens.EOS]
        vocab_size = model.model.config.vocab_size
        max_steps = max(model.tabular_max_length, len(model.col_idx_ids))
        col0_ids = model.col_idx_ids[0]

        def decode_col_a(seqs_np):
            return pd.Series(["x" if v == col0_ids[0] else "y" for v in seqs_np[:, 1]])

        def p_a_x(n=4000, seed=999):
            seqs = _sample_sequences(sampler, model, device, n, seed).cpu().numpy()
            return (decode_col_a(seqs) == "x").mean()

        pre_p = p_a_x()

        import copy

        ref_net = copy.deepcopy(model.model).to(device)
        ref_net.eval()
        for p in ref_net.parameters():
            p.requires_grad_(False)

        optimizer = torch.optim.Adam(model.model.parameters(), lr=1e-3)
        torch.manual_seed(44)
        for round_ in range(30):
            seqs_t = torch.as_tensor(_sample_sequences(sampler, model, device, 64, seed=3000 + round_), device=device)
            seqs_np = seqs_t.cpu().numpy()
            batch_df = pd.DataFrame({
                "a": decode_col_a(seqs_np),
                "b": ["p" if v == model.col_idx_ids[1][0] else ("q" if v == model.col_idx_ids[1][1] else "r")
                      for v in seqs_np[:, 2]],
            })
            reward = rtf_rl.leave_one_out_marginal_reward(batch_df, target, columns=["a", "b"], pairs=[("a", "b")])
            reward_t = torch.as_tensor(reward, device=device, dtype=torch.float32)

            new_logp = rtf_rl.log_prob_of_sequences(
                model.model, seqs_t, model.col_idx_ids, eos, vocab_size, max_steps, requires_grad=True
            )
            with torch.no_grad():
                ref_logp = rtf_rl.log_prob_of_sequences(ref_net, seqs_t, model.col_idx_ids, eos, vocab_size, max_steps)
            advantage = rtf_rl.group_relative_advantage(reward_t)
            out = rtf_rl.grpo_loss(new_logp, ref_logp, advantage, kl_coef=0.02)
            optimizer.zero_grad()
            out["loss"].backward()
            optimizer.step()

        post_p = p_a_x()
        true_p = target["col"]["a"].get("x", 0)
        assert abs(post_p - true_p) < abs(pre_p - true_p), (
            f"leave_one_out_marginal_reward did not move the marginal closer to the true training "
            f"distribution (pre={pre_p:.4f}, post={post_p:.4f}, true={true_p:.4f})"
        )


def _brute_force_loo_reward(labels: pd.Series, tgt: dict) -> np.ndarray:
    """Recomputes the leave-one-out reward the SLOW, obviously-correct way: for every row, drop it,
    rebuild the frequency table from scratch, recompute the squared-error loss, and diff against
    the full-batch loss -- used only to cross-check `rtf_rl._loo_categorical_reward`'s closed form,
    never as the real implementation."""
    n = len(labels)

    def sq_loss(sub: pd.Series) -> float:
        m = len(sub)
        counts = sub.value_counts().to_dict()
        values = set(counts) | set(tgt)
        return sum((tgt.get(v, 0.0) - counts.get(v, 0) / m) ** 2 for v in values)

    full = sq_loss(labels)
    out = np.zeros(n)
    for i in range(n):
        without_i = labels.drop(labels.index[i])
        out[i] = sq_loss(without_i) - full
    return out


def test_leave_one_out_marginal_reward_matches_brute_force_recomputation():
    rng = np.random.default_rng(3)
    cats = ["a", "b", "c", "d"]
    labels = pd.Series(rng.choice(cats, size=60, p=[0.5, 0.3, 0.15, 0.05]))
    tgt = {"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1}
    fast = rtf_rl._loo_categorical_reward(labels, tgt)
    slow = _brute_force_loo_reward(labels, tgt)
    assert np.allclose(fast, slow, atol=1e-9), f"max diff {np.max(np.abs(fast - slow))}"


def test_leave_one_out_marginal_reward_hand_computed():
    # target 50/50 x/y; batch is 3 x's and 1 y -- see test file history for the by-hand derivation.
    labels = pd.Series(["x", "x", "x", "y"])
    tgt = {"x": 0.5, "y": 0.5}
    reward = rtf_rl._loo_categorical_reward(labels, tgt)
    assert np.allclose(reward[:3], -5.0 / 72.0, atol=1e-12)
    assert np.isclose(reward[3], 3.0 / 8.0, atol=1e-12)


def test_leave_one_out_marginal_reward_pairs_matches_brute_force():
    rng = np.random.default_rng(4)
    a = rng.choice(["p", "q"], size=50)
    b = rng.choice(["1", "2", "3"], size=50)
    batch = pd.DataFrame({"a": a, "b": b})
    target = dict(
        col={}, edges={"a": None, "b": None},
        pair={("a", "b"): {("p", "1"): 0.3, ("p", "2"): 0.1, ("p", "3"): 0.1,
                           ("q", "1"): 0.1, ("q", "2"): 0.2, ("q", "3"): 0.2}},
    )
    fast = rtf_rl.leave_one_out_marginal_reward(batch, target, columns=[], pairs=[("a", "b")])
    labels = pd.Series(list(zip(batch["a"], batch["b"])))
    slow = _brute_force_loo_reward(labels, target["pair"][("a", "b")])
    assert np.allclose(fast, slow, atol=1e-9)


def test_leave_one_out_marginal_reward_causally_favors_the_underrepresented_value():
    """Mirrors regression_influence_reward's own causal (not just algebraic) verification: build a
    batch skewed away from the target (80/20 x/y vs a 50/50 target), then check that duplicating
    the rows this reward ranks HIGHEST (the minority, under-represented value) moves the resulting
    batch's own frequency closer to the target than duplicating the rows it ranks LOWEST."""
    labels = pd.Series(["x"] * 80 + ["y"] * 20)
    tgt = {"x": 0.5, "y": 0.5}
    reward = rtf_rl._loo_categorical_reward(labels, tgt)
    assert reward[labels == "y"][0] > reward[labels == "x"][0], (
        "the under-represented value ('y') must score higher than the over-represented one ('x')"
    )

    def freq_gap_after_duplicating(value: str, k: int) -> float:
        boosted = pd.concat([labels, pd.Series([value] * k)], ignore_index=True)
        freq = boosted.value_counts(normalize=True)
        return abs(freq.get("x", 0.0) - tgt["x"]) + abs(freq.get("y", 0.0) - tgt["y"])

    gap_before = freq_gap_after_duplicating("x", 0)
    gap_top = freq_gap_after_duplicating("y", 30)  # duplicating the highest-reward (minority) value
    gap_bottom = freq_gap_after_duplicating("x", 30)  # duplicating the lowest-reward (majority) value
    assert gap_top < gap_before
    assert gap_top < gap_bottom


def test_privacy_penalty_zero_at_or_above_threshold():
    d = np.array([0.5, 1.0, 2.0])
    assert np.allclose(rtf_rl.privacy_penalty(d, threshold=0.5), 0.0)


def test_privacy_penalty_hand_computed_below_threshold():
    d = np.array([0.0, 0.2, 0.5, 1.0])
    penalty = rtf_rl.privacy_penalty(d, threshold=0.5, scale=2.0)
    # shortfall = threshold - d, clipped at 0, times -scale
    assert np.allclose(penalty, [-1.0, -0.6, 0.0, 0.0])


def test_privacy_penalty_scale_is_linear():
    d = np.array([0.0, 0.1, 0.3])
    base = rtf_rl.privacy_penalty(d, threshold=0.4, scale=1.0)
    scaled = rtf_rl.privacy_penalty(d, threshold=0.4, scale=5.0)
    assert np.allclose(scaled, base * 5.0)


def test_detect_heavy_tailed_columns():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "heavy": rng.lognormal(mean=10, sigma=1.2, size=5000),
        "light": rng.uniform(0, 100, size=5000),
        "normalish": rng.normal(100, 10, size=5000),
        "bounded_count": rng.integers(1, 8, size=5000),
    })
    detected = rtf_rl.detect_heavy_tailed_columns(df, list(df.columns))
    assert "heavy" in detected
    assert "light" not in detected
    assert "normalish" not in detected
    assert "bounded_count" not in detected
