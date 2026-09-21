from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import stats
from transformers import EncoderDecoderConfig
from transformers.models.gpt2 import GPT2Config

import realtabformer
from realtabformer.realtabformer import ModelType, REaLTabFormer

RANDOM_SEED = 1029


def test_ModelType():
    assert ModelType.types() == ["tabular", "relational"]


def test_default_init():
    model_types = [ModelType.tabular, ModelType.relational]

    for model_type in model_types:
        rtf_model = REaLTabFormer(model_type)

        # Track the variables that we have tested to
        # make sure that all variables that will be
        # added or removed in the future will be caught
        # by the test.
        model_vars_tested = set()

        assert rtf_model.model_type == model_type
        model_vars_tested.add("model_type")

        # Check default dir arguments
        assert isinstance(rtf_model.checkpoints_dir, Path)
        assert rtf_model.checkpoints_dir.name == "rtf_checkpoints"
        model_vars_tested.add("checkpoints_dir")

        assert isinstance(rtf_model.samples_save_dir, Path)
        assert rtf_model.samples_save_dir.name == "rtf_samples"
        model_vars_tested.add("samples_save_dir")

        assert rtf_model.epochs == 100
        model_vars_tested.add("epochs")

        assert rtf_model.batch_size == (32 if model_type == ModelType.tabular else 8)
        model_vars_tested.add("batch_size")

        assert rtf_model.random_state == 1029
        model_vars_tested.add("random_state")

        assert rtf_model.train_size == 1
        model_vars_tested.add("train_size")

        assert rtf_model.early_stopping_patience == 5
        model_vars_tested.add("early_stopping_patience")

        assert rtf_model.early_stopping_threshold == 0
        model_vars_tested.add("early_stopping_threshold")

        assert rtf_model.mask_rate == 0
        model_vars_tested.add("mask_rate")

        assert rtf_model.numeric_nparts == 1
        model_vars_tested.add("numeric_nparts")

        assert rtf_model.numeric_precision == 4
        model_vars_tested.add("numeric_precision")

        assert rtf_model.numeric_max_len == 10
        model_vars_tested.add("numeric_max_len")

        if model_type == ModelType.tabular:
            with pytest.raises(AttributeError):
                # The argument `output_max_length` is not set
                # for the tabular model.
                assert rtf_model.output_max_length

            # Implicitly tests `_init_tabular`
            assert isinstance(rtf_model.tabular_config, GPT2Config)
            assert rtf_model.tabular_config.n_layer == 6
            model_vars_tested.add("tabular_config")
        else:
            assert rtf_model.output_max_length == 512
            model_vars_tested.add("output_max_length")

            assert rtf_model.freeze_parent_model
            model_vars_tested.add("freeze_parent_model")

            # Relational model
            assert rtf_model.parent_vocab is None
            model_vars_tested.add("parent_vocab")

            assert rtf_model.parent_gpt2_config is None
            model_vars_tested.add("parent_gpt2_config")

            assert rtf_model.parent_gpt2_state_dict is None
            model_vars_tested.add("parent_gpt2_state_dict")

            assert rtf_model.parent_col_transform_data is None
            model_vars_tested.add("parent_col_transform_data")

            # Implicitly tests `_init_relational`
            assert isinstance(rtf_model.relational_config, EncoderDecoderConfig)
            assert isinstance(rtf_model.relational_config.encoder, GPT2Config)
            assert isinstance(rtf_model.relational_config.decoder, GPT2Config)
            assert rtf_model.relational_config.encoder.n_layer == 6
            assert rtf_model.relational_config.decoder.n_layer == 6

            model_vars_tested.add("relational_config")

        # Validate the implicit default values in `training_args_kwargs`
        assert rtf_model.training_args_kwargs["evaluation_strategy"] == "steps"
        assert (
            rtf_model.training_args_kwargs["output_dir"]
            == rtf_model.checkpoints_dir.as_posix()
        )
        assert rtf_model.training_args_kwargs["evaluation_strategy"] == "steps"

        assert rtf_model.training_args_kwargs["metric_for_best_model"] == "loss"
        assert rtf_model.training_args_kwargs["overwrite_output_dir"] is True
        assert rtf_model.training_args_kwargs["num_train_epochs"] == rtf_model.epochs
        assert (
            rtf_model.training_args_kwargs["per_device_train_batch_size"]
            == rtf_model.batch_size
        )
        assert (
            rtf_model.training_args_kwargs["per_device_eval_batch_size"]
            == rtf_model.batch_size
        )

        assert rtf_model.training_args_kwargs["gradient_accumulation_steps"] == (
            1 if model_type == ModelType.tabular else 4
        )
        assert rtf_model.training_args_kwargs["remove_unused_columns"] is True
        assert rtf_model.training_args_kwargs["logging_steps"] == 100
        assert rtf_model.training_args_kwargs["save_steps"] == 100
        assert rtf_model.training_args_kwargs["eval_steps"] == 100
        assert rtf_model.training_args_kwargs["load_best_model_at_end"] is True
        assert (
            rtf_model.training_args_kwargs["save_total_limit"]
            == rtf_model.early_stopping_patience + 1
        )
        model_vars_tested.add("training_args_kwargs")

        # Validate empty-initialized attributes
        list_defaults = [
            "columns",
            "drop_na_cols",
            "processed_columns",
            "numeric_columns",
            "datetime_columns",
        ]
        for ld in list_defaults:
            assert (
                isinstance(getattr(rtf_model, ld), list)
                and len(getattr(rtf_model, ld)) == 0
            )

        dict_defaults = ["column_dtypes", "column_has_missing", "vocab", "col_idx_ids"]
        for dd in dict_defaults:
            assert (
                isinstance(getattr(rtf_model, dd), dict)
                and len(getattr(rtf_model, dd)) == 0
            )

        none_defaults = [
            "model",
            "tabular_max_length",
            "relational_max_length",
            "tabular_col_size",
            "relational_col_size",
            "experiment_id",
            "col_transform_data",
            "in_col_transform_data",
            "target_col",
            "trainer_state",
        ]
        for nd in none_defaults:
            assert getattr(rtf_model, nd) is None

        assert rtf_model.realtabformer_version == realtabformer.__version__
        model_vars_tested.add("realtabformer_version")

        model_vars_tested.update(list_defaults)
        model_vars_tested.update(dict_defaults)
        model_vars_tested.update(none_defaults)

        model_vars = set(vars(rtf_model))

        print(model_vars.difference(model_vars_tested))

        assert len(model_vars.difference(model_vars_tested)) == 0
        assert len(model_vars) == len(model_vars_tested), f"{model_vars}...{model_vars_tested}"


# def test_tabular_init():
#     training_args_kwargs = dict(
#         logging_steps=100,
#         save_steps=100,
#         eval_steps=100,
#         save_total_limit=10,
#         gradient_accumulation_steps=4,
#     )

#     samples_save_dir = "samples_save_dir"
#     batch_size = 8
#     epochs = 10
#     mask_rate = 0.2
#     train_size = 1

#     tabular_rtf = REaLTabFormer(
#         model_type="tabular",
#         samples_save_dir=samples_save_dir,
#         epochs=epochs, batch_size=batch_size,
#         mask_rate=mask_rate,
#         train_size=train_size,
#         random_state=RANDOM_SEED,
#         **training_args_kwargs)


# --- digit_entropy_weighting (beta): entropy-based digit-chunk loss weighting ---


def test_digit_entropy_weighting_default_off_leaves_no_chunk_significance_weights():
    rng = np.random.default_rng(RANDOM_SEED)
    df = pd.DataFrame({
        "price": rng.normal(100, 20, size=40).round(2),
        "gender": rng.choice(["m", "f"], size=40),
    })
    model = REaLTabFormer(
        model_type="tabular", epochs=1, batch_size=8, random_state=RANDOM_SEED,
    )
    model.fit(df, device="cpu", n_critic=0)
    assert "chunk_significance_weights" not in model.vocab


def test_digit_entropy_weighting_end_to_end_favors_high_entropy_chunk():
    # Same claim validated for realtabformer2.py: fit a tiny v1 model on a
    # deliberately heavy-tailed numeric column and confirm the computed
    # chunk_significance_weights give the near-constant leading digit
    # chunk a *lower* weight than a higher-variance later chunk.
    rng = np.random.default_rng(11)
    n = 300
    price = np.round(rng.exponential(scale=50, size=n))
    price = np.clip(price, 1, 999)
    df = pd.DataFrame({
        "price": price, "gender": rng.choice(["m", "f"], size=n),
    })

    model = REaLTabFormer(
        model_type="tabular",
        epochs=1,
        batch_size=8,
        random_state=RANDOM_SEED,
        numeric_max_len=6,
        numeric_precision=0,
        numeric_nparts=1,
    )
    model.fit(df, device="cpu", n_critic=0, digit_entropy_weighting=True)

    weights = model.vocab["chunk_significance_weights"]
    price_chunks = sorted(c for c in model.processed_columns if "price" in c)
    assert len(price_chunks) > 1

    leading_w = weights[price_chunks[0]]
    trailing_w = weights[price_chunks[-1]]
    assert leading_w < trailing_w, (
        f"leading chunk weight ({leading_w}) should be lower than the "
        f"trailing chunk's ({trailing_w}) for a heavy-tailed column"
    )

    samples = model.sample(n_samples=5, device="cpu")
    assert len(samples) == 5
    assert list(samples.columns) == list(df.columns)


# --- numeric_categorical_threshold: cardinality-aware numeric dispatch -----


def test_numeric_categorical_threshold_end_to_end_dtype_round_trip():
    rng = np.random.default_rng(9)
    n = 200
    bedrooms = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.05, 0.30, 0.35, 0.20, 0.10])
    rating = rng.choice([1.0, 2.0, 3.0, 4.0, 5.0], size=n).astype(float)
    missing_idx = rng.choice(n, size=int(n * 0.1), replace=False)
    rating[missing_idx] = np.nan
    df = pd.DataFrame({
        "bedrooms": bedrooms.astype("int64"),
        "rating": rating,
        "price": rng.integers(100000, 999999, size=n).astype(float),
        "gender": rng.choice(["m", "f"], size=n),
    })

    model = REaLTabFormer(
        model_type="tabular",
        epochs=3,
        batch_size=16,
        random_state=RANDOM_SEED,
        numeric_categorical_threshold=10,
    )
    model.fit(df, device="cpu", n_critic=0)

    bedroom_cols = [c for c in model.processed_columns if "bedrooms" in c]
    price_cols = [c for c in model.processed_columns if "price" in c]
    assert len(bedroom_cols) == 1 and "CATEGORICAL" in bedroom_cols[0]
    assert len(price_cols) > 1 and all("NUMERIC" in c for c in price_cols)

    samples = model.sample(n_samples=20, device="cpu")

    # Explicit requirement: the recovered dtype must match the original
    # input column's dtype exactly, regardless of which internal pipeline
    # produced it.
    assert samples["bedrooms"].dtype == df["bedrooms"].dtype
    assert samples["rating"].dtype == df["rating"].dtype
    assert samples["price"].dtype == df["price"].dtype
    assert samples["bedrooms"].dropna().isin([1, 2, 3, 4, 5]).all()


# --- numeric_quantile_encoding: CDF-based numeric representation -----------


def test_numeric_quantile_encoding_end_to_end_dtype_and_distributional_fidelity():
    rng = np.random.default_rng(11)
    n = 500
    price = np.round(rng.lognormal(mean=6.0, sigma=2.0, size=n), 2)
    price = np.clip(price, 0.01, 500000)
    df = pd.DataFrame({
        "price": price, "gender": rng.choice(["m", "f"], size=n),
    })

    model = REaLTabFormer(
        model_type="tabular",
        epochs=25,
        batch_size=16,
        random_state=RANDOM_SEED,
        numeric_max_len=8,
        numeric_precision=4,
        numeric_nparts=1,
        numeric_quantile_encoding=True,
    )
    model.fit(df, device="cpu", n_critic=0)

    samples = model.sample(n_samples=200, device="cpu")

    # Explicit dtype-preservation bar, matching numeric_categorical_threshold's.
    assert samples["price"].dtype == df["price"].dtype

    # Bounded by the training range (np.interp's clamp-to-boundary
    # extrapolation rule, the accepted tradeoff of choosing quantile
    # encoding over magnitude+mantissa).
    assert samples["price"].min() >= df["price"].min()
    assert samples["price"].max() <= df["price"].max()

    # Distributional fidelity, measured directly rather than argued from
    # the inverse-transform-sampling theory alone: a two-sample KS test
    # between the generated sample and the training data should not
    # strongly reject the null of equal distributions.
    ks_stat, _ = stats.ks_2samp(samples["price"].dropna(), df["price"])
    assert ks_stat < 0.2, f"KS statistic too high: {ks_stat}"


# --- sensitivity-based training (_train_with_sensitivity): gen_kwargs=None crash ---


def test_sensitivity_training_does_not_crash_with_default_gen_kwargs():
    # Regression test: fit()'s own default is n_critic=5 (> 0), which
    # routes to _train_with_sensitivity -- REaLTabFormer's DCR-bootstrap
    # overfitting-protection mechanism, one of the two headline
    # contributions of the original paper. That method's gen_kwargs
    # parameter defaults to None, but was unpacked directly as
    # `**gen_kwargs` when generating samples for each critic round --
    # crashing with "argument after ** must be a mapping, not NoneType"
    # for any caller who didn't happen to pass gen_kwargs explicitly,
    # which is the ordinary way to call .fit(). Pre-existing since
    # commit 73f23964, unrelated to any change this session made --
    # only found because this session's experiments finally exercised
    # n_critic>0 (every earlier test/experiment used n_critic=0, which
    # routes around this code path entirely via the plain-fit branch).
    rng = np.random.default_rng(RANDOM_SEED)
    n = 40
    df = pd.DataFrame({
        "price": rng.normal(100, 20, size=n).round(2),
        "gender": rng.choice(["m", "f"], size=n),
    })

    model = REaLTabFormer(
        model_type="tabular",
        epochs=2,
        batch_size=8,
        random_state=RANDOM_SEED,
    )
    # n_critic=1 (not the 0 every other test in this file uses) forces at
    # least one critic round -- the code path where gen_kwargs=None was
    # unpacked and crashed -- within a fast, minimal test. num_bootstrap
    # is dropped to keep the sensitivity-threshold bootstrap itself fast;
    # correctness of that mechanism's own statistics isn't what's under
    # test here, just that gen_kwargs=None doesn't crash it.
    model.fit(df, device="cpu", n_critic=1, n_critic_stop=1, num_bootstrap=20)
    assert model.model is not None


def test_sensitivity_training_saves_reloadable_checkpoint(tmp_path):
    # Regression test for the same bug class fixed for the cusum path:
    # _train_with_sensitivity picked the best-discriminator model via
    # trainer.save_model()/from_pretrained() (raw HF weights only), but
    # never wrote a REaLTabFormer-loadable checkpoint (rtf_config.json +
    # rtf_model.pt) anywhere -- load_from_dir on that directory failed.
    rng = np.random.default_rng(RANDOM_SEED)
    n = 40
    df = pd.DataFrame(
        {
            "price": rng.normal(100, 20, size=n).round(2),
            "gender": rng.choice(["m", "f"], size=n),
        }
    )

    model = REaLTabFormer(
        model_type="tabular",
        epochs=2,
        batch_size=8,
        random_state=RANDOM_SEED,
        checkpoints_dir=str(tmp_path / "checkpoints"),
    )
    model.fit(df, device="cpu", n_critic=1, n_critic_stop=1, num_bootstrap=20)

    ckpt_path = Path(model.checkpoints_dir) / "sensitivity_best"
    assert (ckpt_path / "rtf_config.json").exists()
    assert (ckpt_path / "rtf_model.pt").exists()

    reloaded = REaLTabFormer.load_from_dir(ckpt_path)
    assert reloaded.vocab.keys() == model.vocab.keys()
    assert reloaded.processed_columns == model.processed_columns

    current_state = model.model.state_dict()
    reloaded_state = reloaded.model.state_dict()
    assert reloaded_state.keys() == current_state.keys()
    for key in current_state:
        assert torch.equal(reloaded_state[key].cpu(), current_state[key].cpu())

    samples = reloaded.sample(10, device="cpu")
    assert len(samples) == 10


# --- constrained_loss and the sensitivity path's forwarding of loss arguments --


def _stub_rtf(vocab_size, seq_len, allowed_per_step):
    import types

    return types.SimpleNamespace(
        vocab={"token2id": {"[EOS]": 1}},
        col_idx_ids={i: allowed_per_step(i) for i in range(seq_len - 1)},
        tabular_max_length=seq_len,
    )


def test_constrained_loss_equals_standard_cross_entropy_when_nothing_is_masked():
    import types

    import torch
    import torch.nn.functional as F

    from realtabformer.rtf_loss import build_constrained_loss

    torch.manual_seed(0)
    B, L, V = 5, 7, 11
    rtf = _stub_rtf(V, L, lambda i: list(range(V)))
    logits = torch.randn(B, L, V)
    labels = torch.randint(0, V, (B, L))
    loss = build_constrained_loss(rtf)(types.SimpleNamespace(logits=logits), labels)
    ref = F.cross_entropy(logits[:, :-1].reshape(-1, V), labels[:, 1:].reshape(-1))
    assert torch.allclose(loss, ref, atol=1e-6)


def test_constrained_loss_is_lower_with_a_mask_and_skips_ignored_labels():
    import types

    import torch
    import torch.nn.functional as F

    from realtabformer.rtf_loss import build_constrained_loss

    torch.manual_seed(1)
    B, L, V = 6, 6, 12
    allowed = lambda i: [(3 * i + k) % V for k in range(3)]  # 3 valid tokens per step
    rtf = _stub_rtf(V, L, allowed)
    logits = torch.randn(B, L, V)
    labels = torch.zeros(B, L, dtype=torch.long)
    for i in range(L - 1):
        labels[:, i + 1] = torch.tensor(allowed(i))[torch.randint(0, 3, (B,))]
    out = types.SimpleNamespace(logits=logits)

    loss = build_constrained_loss(rtf)(out, labels)
    standard = F.cross_entropy(logits[:, :-1].reshape(-1, V), labels[:, 1:].reshape(-1))
    assert loss < standard  # mass on other columns' tokens is removed

    ignored = labels.clone()
    ignored[:, 3:] = -100  # positions the model is not asked to predict
    loss_ignored = build_constrained_loss(rtf)(out, ignored)
    keep = build_constrained_loss(rtf)(
        types.SimpleNamespace(logits=logits[:, :3]), labels[:, :3]
    )
    assert torch.isfinite(loss_ignored)
    assert torch.allclose(loss_ignored, keep, atol=1e-5)


def test_constrained_loss_option_is_wired_into_the_trainer(tmp_path):
    import numpy as np
    import pandas as pd
    from transformers import GPT2Config

    from realtabformer import REaLTabFormer

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.choice(list("xyz"), 80), "b": rng.integers(0, 40, 80)})
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def trainer_for(**kw):
        m = REaLTabFormer(
            model_type="tabular", epochs=1, batch_size=8,
            checkpoints_dir=str(tmp_path / "c"),
            tabular_config=GPT2Config(n_layer=1, n_embd=32, n_head=2), **kw,
        )
        m.target_col, m.trainer_kwargs = None, {}
        return m._fit_tabular(df, device=device)

    assert trainer_for().compute_loss_func is None  # off by default
    assert trainer_for(constrained_loss=True).compute_loss_func is not None

    # End to end: it actually trains, on the no-critic path.
    m = REaLTabFormer(
        model_type="tabular", epochs=1, batch_size=8, checkpoints_dir=str(tmp_path / "e"),
        tabular_config=GPT2Config(n_layer=1, n_embd=32, n_head=2), constrained_loss=True,
    )
    m.fit(df, device=device, n_critic=0)
    assert len(m.sample(n_samples=5, gen_batch=5, device=device)) == 5


def test_sensitivity_path_forwards_loss_related_fit_arguments(tmp_path, monkeypatch):
    # The default (sensitivity) training path used to call `_fit_tabular` with only
    # device/epochs, silently dropping every one of these.
    import numpy as np
    import pandas as pd

    from realtabformer import REaLTabFormer
    from realtabformer.rtf_analyze import SyntheticDataBench

    class Stop(Exception):
        pass

    seen = {}

    def fake_fit_tabular(self, df, **kwargs):
        seen.update(kwargs)
        raise Stop

    monkeypatch.setattr(REaLTabFormer, "_fit_tabular", fake_fit_tabular)
    monkeypatch.setattr(
        SyntheticDataBench, "compute_sensitivity_threshold",
        staticmethod(lambda *a, **k: np.array([0.1, 0.2, 0.3])),
    )
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.choice(list("xyz"), 60), "b": rng.integers(0, 40, 60)})
    loss = lambda outputs, labels, num_items_in_batch=None: None
    m = REaLTabFormer(model_type="tabular", epochs=2, checkpoints_dir=str(tmp_path / "c"))

    with pytest.raises(Stop):
        m.fit(
            df, device="cpu", n_critic=5, num_bootstrap=3,
            field_weights={"a": 2.0}, compute_loss_func=loss, predict_fields=["a"],
            digit_entropy_weighting=True, digit_entropy_weight_floor=0.2,
        )

    assert seen["compute_loss_func"] is loss
    assert seen["field_weights"] == {"a": 2.0}
    assert seen["predict_fields"] == ["a"]
    assert seen["digit_entropy_weighting"] is True
    assert seen["digit_entropy_weight_floor"] == 0.2


# --- ema_horizon: weight averaging ----------------------------------------


def test_WeightEMA_math_swap_and_restore():
    from realtabformer.rtf_ema import WeightEMA

    torch.manual_seed(0)
    model = torch.nn.Linear(3, 2)
    ema = WeightEMA(model, decay=0.9)
    raw0 = [p.detach().clone() for p in model.parameters()]
    assert all(torch.equal(a, b) for a, b in zip(ema.shadow, raw0))

    # One update against a hand-computed value.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)
    for s, r0 in zip(ema.shadow, raw0):
        assert torch.allclose(s, 0.9 * r0 + 0.1 * (r0 + 1.0), atol=1e-6)

    raw1 = [p.detach().clone() for p in model.parameters()]
    with ema.averaged(model):
        assert all(torch.equal(p.data, s) for p, s in zip(model.parameters(), ema.shadow))
        ema.swap_in(model)  # re-entering must not overwrite the saved raw weights
    assert all(torch.equal(p.data, r) for p, r in zip(model.parameters(), raw1))
    ema.swap_out(model)  # exiting when not entered is a no-op
    assert all(torch.equal(p.data, r) for p, r in zip(model.parameters(), raw1))

    ema.copy_to(model)
    assert all(torch.equal(p.data, s) for p, s in zip(model.parameters(), ema.shadow))


def test_WeightEMA_decay_for_horizon():
    import math

    from realtabformer.rtf_ema import WeightEMA

    assert WeightEMA.decay_for_horizon(1.0, 10) == pytest.approx(math.exp(-0.1))
    assert WeightEMA.decay_for_horizon(4.0, 25) == pytest.approx(math.exp(-0.01))
    assert 0.0 < WeightEMA.decay_for_horizon(0.001, 1) < 1.0
    with pytest.raises(ValueError):
        WeightEMA(torch.nn.Linear(1, 1), decay=1.0)


def _ema_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({"a": rng.choice(list("xyz"), 64), "b": rng.integers(0, 40, 64)})


def test_ema_horizon_final_model_is_the_average_on_the_plain_path(tmp_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(name, **kw):
        torch.manual_seed(3)
        m = REaLTabFormer(
            model_type="tabular", epochs=3, batch_size=8, random_state=3,
            checkpoints_dir=str(tmp_path / name),
            tabular_config=GPT2Config(n_layer=1, n_embd=32, n_head=2), **kw,
        )
        m.fit(_ema_df(), device=device, n_critic=0)
        return m

    plain, avg = fit("plain", ema_horizon=0.0), fit("avg", ema_horizon=1.0)
    shadow = avg._ema_state["ema"].shadow
    assert all(torch.allclose(p.detach().cpu(), s.cpu()) for p, s in zip(avg.model.parameters(), shadow))
    assert not all(
        torch.allclose(a.detach().cpu(), b.detach().cpu(), atol=1e-6)
        for a, b in zip(avg.model.parameters(), plain.model.parameters())
    )  # and it is not just the raw weights
    assert plain._ema_state == {}  # off by default: no state at all


def test_ema_horizon_rejected_where_unsupported(tmp_path):
    m = REaLTabFormer(model_type="tabular", epochs=1, ema_horizon=1.0, checkpoints_dir=str(tmp_path / "c"))
    with pytest.raises(NotImplementedError):
        m.fit(_ema_df(), device="cpu", overfitting_detection_method="cusum")
    with pytest.raises(NotImplementedError):
        m.fit(_ema_df(), device="cpu", objective_callback=lambda *a, **k: (0.0, True))


def test_ema_weights_are_used_by_the_critic_and_restored_for_training(tmp_path, monkeypatch):
    from realtabformer.rtf_analyze import SyntheticDataBench

    device = "cuda" if torch.cuda.is_available() else "cpu"
    monkeypatch.setattr(
        SyntheticDataBench, "compute_sensitivity_threshold",
        staticmethod(lambda *a, **k: np.array([0.1, 0.2, 0.3])),
    )
    monkeypatch.setattr(
        SyntheticDataBench, "compute_sensitivity_metric", staticmethod(lambda *a, **k: 0.05)
    )
    df = _ema_df()
    m = REaLTabFormer(
        model_type="tabular", epochs=3, batch_size=8, random_state=1,
        checkpoints_dir=str(tmp_path / "c"),
        tabular_config=GPT2Config(n_layer=1, n_embd=32, n_head=2), ema_horizon=1.0,
    )
    seen = {"sample_is_avg": [], "build_is_raw": [], "avg_snapshots": []}

    def fake_sample(self, n_samples=None, **kw):
        ema = self._ema_state["ema"]
        p0 = next(self.model.parameters())
        seen["sample_is_avg"].append(bool(torch.equal(p0.data, ema.shadow[0])))
        seen["avg_snapshots"].append(p0.detach().cpu().clone())
        return df.sample(n=n_samples, replace=True, random_state=0)

    real_build = REaLTabFormer._build_tabular_trainer

    def spy_build(self, *a, **k):
        t = real_build(self, *a, **k)
        ema = self._ema_state.get("ema")
        if ema is not None:  # from the second round on
            p0 = next(self.model.parameters())
            seen["build_is_raw"].append(not torch.equal(p0.data, ema.shadow[0]))
        return t

    monkeypatch.setattr(REaLTabFormer, "sample", fake_sample)
    monkeypatch.setattr(REaLTabFormer, "_build_tabular_trainer", spy_build)
    m.fit(df, device=device, n_critic=1, n_critic_stop=5, num_bootstrap=3)

    assert seen["sample_is_avg"] and all(seen["sample_is_avg"])  # critic saw the average
    assert seen["build_is_raw"] and all(seen["build_is_raw"])  # raw weights restored to train
    # The model that fit() loaded back is one of the saved averaged snapshots.
    final = next(m.model.parameters()).detach().cpu()
    assert any(torch.equal(final, s) for s in seen["avg_snapshots"])


# --- defaults: batch 32 x accumulation 1 (tabular) and EMA on -------------------


def test_default_batch_size_and_accumulation_keep_the_effective_batch():
    tab = REaLTabFormer(model_type="tabular")
    assert (tab.batch_size, tab.training_args_kwargs["gradient_accumulation_steps"]) == (32, 1)
    assert tab.training_args_kwargs["per_device_train_batch_size"] == 32

    # The relational model was not measured: unchanged.
    rel = REaLTabFormer(model_type="relational")
    assert (rel.batch_size, rel.training_args_kwargs["gradient_accumulation_steps"]) == (8, 4)

    # Explicit batch sizes keep the ~32 effective batch they always had.
    for bs, ga in [(8, 4), (16, 2), (32, 1), (64, 1)]:
        m = REaLTabFormer(model_type="tabular", batch_size=bs)
        assert m.training_args_kwargs["gradient_accumulation_steps"] == ga, (bs, ga)

    # An explicit accumulation always wins.
    m = REaLTabFormer(model_type="tabular", gradient_accumulation_steps=2)
    assert m.training_args_kwargs["gradient_accumulation_steps"] == 2


def test_ema_is_on_by_default_but_switched_off_where_unsupported(tmp_path, monkeypatch):
    class Stop(Exception):
        pass

    m = REaLTabFormer(model_type="tabular", checkpoints_dir=str(tmp_path / "c"))
    assert m.ema_horizon == 1.0 and m._ema_explicit is False
    assert REaLTabFormer(model_type="tabular", ema_horizon=0.0).ema_horizon == 0.0

    def stop(self, *a, **k):
        raise Stop

    # The default must not break a path that cannot use it: it is just disabled.
    monkeypatch.setattr(REaLTabFormer, "_train_with_cusum", stop)
    with pytest.raises(Stop):
        m.fit(_ema_df(), device="cpu", overfitting_detection_method="cusum")
    assert m._ema_on is False

    # ...whereas an explicit request for it there is an error, not a silent no-op.
    explicit = REaLTabFormer(model_type="tabular", ema_horizon=1.0, checkpoints_dir=str(tmp_path / "e"))
    with pytest.raises(NotImplementedError):
        explicit.fit(_ema_df(), device="cpu", overfitting_detection_method="cusum")
