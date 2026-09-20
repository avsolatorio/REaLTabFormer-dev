"""Logic shared verbatim between `REaLTabFormer` (realtabformer.py) and
`REaLTabFormer2` (realtabformer2.py).

Extracted after a full method-by-method diff of the two classes (see
notes/lab_notebook.md, 2026-09-19 entry) confirmed these specific
pieces have no legitimate reason to differ between the two models --
unlike e.g. `_init_tabular`/`_init_relational`/`_fit_tabular`, which
genuinely diverge for v2's backbone-generality and `any_order`/
`shared_numeric_vocab` features and must NOT be merged in here.

`get_experiment_id` in particular was extracted specifically *because*
independently reimplementing it in realtabformer2.py introduced a real
bug (v2 raised on a resumed `fit()`'s periodic checkpoint save, where
v1 does not) -- the version kept here is v1's original, correct logic.
Sharing it removes the class of bug, not just this one instance of it.
"""

import inspect
import time
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import pandas as pd
import torch
from datasets import Dataset

from .data_utils import ModelType, SpecialTokens, build_vocab
from .rtf_sampler import RelationalSampler, TabularSampler
from .rtf_validators import ObservationValidator


def _normalize_gpt2_state_dict(state_dict):
    state = []
    for key, value in state_dict.items():
        if key.startswith("transformer."):
            # The saved state prefixes the weight names
            # with `transformer.` whereas the
            # encoder expects the weight names to not
            # have the prefix.
            key = key.replace("transformer.", "")

        state.append((key, value))

    return OrderedDict(state)


def _validate_get_device(device: str) -> str:
    if (device == "cuda") and (torch.cuda.device_count() == 0):
        if torch.backends.mps.is_available():
            _device = "mps"
        else:
            _device = "cpu"

        warnings.warn(
            f"The device={device} is not available, using device={_device} instead."
        )
        device = _device

    return device


def _build_training_args(cls: Type, kwargs: Dict[str, Any]):
    """Constructs ``cls(**kwargs)`` (``TrainingArguments`` or
    ``Seq2SeqTrainingArguments``), dropping any key ``cls.__init__``
    doesn't actually accept instead of raising ``TypeError``.

    Found necessary in practice, not preemptively: `training_args_kwargs`
    is a fixed dict of parameter names built once, but which parameters
    a given `transformers` release's `TrainingArguments` accepts can
    (and does) drift between versions -- this repo's own dependency
    constraint on `transformers` has no upper bound, so a fresh install
    can land on a release where a previously-standard argument no
    longer exists, breaking construction outright. Filtering against
    the actually-installed class's real signature adapts to whatever
    version is present, rather than requiring this file to be kept in
    lockstep with every `transformers` release. Warns (once per
    dropped key, per call) so a silently-ignored argument doesn't go
    completely unnoticed.
    """
    accepted = set(inspect.signature(cls.__init__).parameters)
    dropped = [k for k in kwargs if k not in accepted]
    if dropped:
        warnings.warn(
            f"{cls.__name__} in the installed transformers version does not "
            f"accept the following argument(s), which were dropped: "
            f"{sorted(dropped)}. This can happen when the installed "
            f"transformers release differs from what this argument set was "
            f"written against."
        )
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    return cls(**filtered)


class SharedModelMixin:
    """Methods with no legitimate reason to differ between
    `REaLTabFormer` and `REaLTabFormer2`. Both classes inherit this
    directly; nothing here should reference a `self` attribute that
    isn't set up identically (in meaning, not necessarily in value) by
    both classes' own `__init__`.
    """

    def _invalid_model_type(self, model_type):
        raise ValueError(
            f"Model type: {model_type} is not valid. REaLTabFormer only supports \
                `tabular` and `relational` values."
        )

    def _extract_column_info(self, df: pd.DataFrame) -> None:
        # Track the column order of the original data
        self.columns = df.columns.to_list()

        # Store the dtypes of the columns
        self.column_dtypes = df.dtypes.astype(str).to_dict()

        # Track which variables have missing values
        self.column_has_missing = (df.isnull().sum() > 0).to_dict()

        # Get the columns where there should be no missing values
        self.drop_na_cols = [
            col for col, has_na in self.column_has_missing.items() if not has_na
        ]

        # Identify the numeric columns. These will undergo
        # special preprocessing.
        self.numeric_columns = df.select_dtypes(include=np.number).columns.to_list()

        # Identify the datetime columns. These will undergo
        # special preprocessing.
        self.datetime_columns = df.select_dtypes(include="datetime").columns.to_list()

    def _generate_vocab(
        self,
        df: pd.DataFrame,
        compute_chunk_significance: bool = False,
        chunk_significance_floor: float = 0.1,
    ) -> dict:
        return build_vocab(
            df,
            special_tokens=SpecialTokens.tokens(),
            add_columns=False,
            compute_chunk_significance=compute_chunk_significance,
            chunk_significance_floor=chunk_significance_floor,
        )

    def _check_model(self):
        assert self.model is not None, "Model is None. Train the model first!"

    def _split_train_eval_dataset(self, dataset: Dataset):
        test_size = 1 - self.train_size
        if test_size > 0:
            dataset = dataset.train_test_split(
                test_size=test_size, seed=self.random_state
            )
            dataset["train_dataset"] = dataset.pop("train")
            dataset["eval_dataset"] = dataset.pop("test")

            # Override `metric_for_best_model` from "loss" to "eval_loss"
            self.training_args_kwargs["metric_for_best_model"] = "eval_loss"
            # Make this explicit so that no assumption is made on the
            # direction of the metric improvement.
            self.training_args_kwargs["greater_is_better"] = False
        else:
            dataset = dict(train_dataset=dataset)
            self.training_args_kwargs["eval_strategy"] = "no"
            self.training_args_kwargs["load_best_model_at_end"] = False

        return dataset

    def get_full_save_dir(self, epoch: int) -> Path:
        return self.full_save_dir / f"epoch_{epoch:03d}"

    def get_experiment_id(self, epoch: int = None) -> str:
        # A per-epoch full-model-save name is independent of `self.experiment_id`
        # (the model's overall save id): it must be generated fresh every time,
        # even on a resumed `fit()` call where `self.experiment_id` is already set
        # from a prior run.
        if epoch is not None:
            return f"full_model_epoch_{epoch:03d}"
        elif self.experiment_id is not None:
            return self.experiment_id
        else:
            return f"id{int((time.time() * 10**10)):024}"

    def sample(
        self,
        n_samples: int = None,
        input_unique_ids: Optional[Union[pd.Series, List]] = None,
        input_df: Optional[pd.DataFrame] = None,
        input_ids: Optional[torch.tensor] = None,
        gen_batch: Optional[int] = 128,
        device: str = "cuda",
        seed_input: Optional[Union[pd.DataFrame, Dict[str, Any]]] = None,
        save_samples: Optional[bool] = False,
        constrain_tokens_gen: Optional[bool] = True,
        validator: Optional[ObservationValidator] = None,
        continuous_empty_limit: int = 10,
        suppress_tokens: Optional[List[int]] = None,
        forced_decoder_ids: Optional[List[List[int]]] = None,
        related_num: Optional[Union[int, List[int]]] = None,
        **generate_kwargs,
    ) -> pd.DataFrame:
        """Generate synthetic tabular data samples

        Args:
            n_samples: Number of synthetic samples to generate for the tabular data.
            input_unique_ids: The unique identifier that will be used to link the input
              data to the generated values when sampling for relational data.
            input_df: Pandas DataFrame containing the tabular input data.
            input_ids: (NOTE: the `input_df` argument is the preferred input)
              The input_ids that conditions the generation of the relational data.
            gen_batch: Controls the batch size of the data generation process. This parameter
              should be adjusted based on the compute resources.
            device: The device used by the generator.
              Use torch devices, e.g., `cpu`, `cuda`, `mps` (experimental)
            seed_input: A dictionary of `col_name:values` for the seed data. Only `col_names`
              that are actually in the first sequence of the training input will be used.
            constrain_tokens_gen: Set whether we impose a constraint at each step of the generation
              limited only to valid tokens for the column.
            validator: An instance of `ObservationValidator` for validating the generated samples.
              The validators are applied to observations only, and don't support inter-observation
              validation. See `ObservationValidator` docs on how to set up a validator.
            continuous_invalid_limit: The sampling will raise an exception if
              `continuous_empty_limit` empty sample batches have been produced continuously. This
              will prevent an infinite loop if the quality of the data generated is not good and
              always produces invalid observations.
            suppress_tokens: (from docs) A list of tokens that will be supressed at generation.
              The SupressTokens logit processor will set their log probs to -inf so that they are
              not sampled. This is a useful feature for imputing missing values.
            forced_decoder_ids: (from docs) A list of pairs of integers which indicates a mapping
              from generation indices to token indices that will be forced before sampling. For
              example, [[1, 123]] means the second generated token will always be a token of
              index 123. This is a useful feature for constraining the model to generate only
              specific stratification variables in surveys, e.g., GEO1, URBAN/RURAL variables.
            related_num: A column name in the input_df containing the number of observations that the child
             table is expected to have for the parent observation. It can also be an integer if the input_df
             corresponds to a set of observations having the same number of expected observations.
             This parameter is only valid for the relational model.
            generate_kwargs: Additional keywords arguments that will be supplied to `.generate`
              method. For a comprehensive list of arguments, see:
              https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate

        Returns:
            DataFrame with n_samples rows of generated data
        """
        self._check_model()
        device = _validate_get_device(device)

        # Clear the cache
        torch.cuda.empty_cache()

        if self.model_type == ModelType.tabular:
            assert n_samples
            assert self.tabular_max_length is not None
            assert self.tabular_col_size is not None
            assert self.col_transform_data is not None
            assert self.orig_to_processed_col_map is not None

            tabular_sampler = TabularSampler.sampler_from_model(
                rtf_model=self, device=device
            )
            synth_df = tabular_sampler.sample_tabular(
                n_samples=n_samples,
                gen_batch=gen_batch,
                device=device,
                seed_input=seed_input,
                constrain_tokens_gen=constrain_tokens_gen,
                validator=validator,
                continuous_empty_limit=continuous_empty_limit,
                suppress_tokens=suppress_tokens,
                forced_decoder_ids=forced_decoder_ids,
                **generate_kwargs,
            )

        elif self.model_type == ModelType.relational:
            assert (input_ids is not None) or (input_df is not None)
            assert self.relational_max_length is not None
            assert self.relational_col_size is not None
            assert self.col_transform_data is not None
            assert self.in_col_transform_data is not None
            assert self.orig_to_processed_col_map is not None

            relational_sampler = RelationalSampler.sampler_from_model(
                rtf_model=self, device=device
            )
            synth_df = relational_sampler.sample_relational(
                input_unique_ids=input_unique_ids,
                input_df=input_df,
                input_ids=input_ids,
                device=device,
                gen_batch=gen_batch,
                constrain_tokens_gen=constrain_tokens_gen,
                validator=validator,
                continuous_empty_limit=continuous_empty_limit,
                suppress_tokens=suppress_tokens,
                forced_decoder_ids=forced_decoder_ids,
                related_num=related_num,
                **generate_kwargs,
            )

        if save_samples:
            samples_fname = (
                self.samples_save_dir
                / f"rtf_{self.model_type}-exp_{self.experiment_id}-{int(time.time())}-samples_{synth_df.shape[0]}.pkl"
            )
            samples_fname.parent.mkdir(parents=True, exist_ok=True)
            synth_df.to_pickle(samples_fname)

        return synth_df

    def predict(
        self,
        data: pd.DataFrame,
        target_col: str,
        target_pos_val: Any = None,
        batch: int = 32,
        obs_sample: int = 30,
        fillunk: bool = True,
        device: str = "cuda",
        disable_progress_bar: bool = True,
        **generate_kwargs,
    ) -> pd.Series:
        """
        Use the trained model to make predictions on a given dataframe.

        Args:
            data: The data to make predictions on, in the form of a Pandas dataframe.
            target_col: The name of the target column in the data to predict.
            target_pos_val: The positive value in the target column to use for binary
              classification. This is produces a one-to-many prediction relative to
              `target_pos_val` for targets that are multi-categorical.
            batch: The batch size to use when making predictions.
            obs_sample: The number of observations to sample from the data when making predictions.
            fillunk: If True, the function will fill any missing values in the data before making
              predictions. Fill unknown tokens with the mode of the batch in the given step.
            device: The device to use for prediction. Can be either "cpu" or "cuda".
            **generate_kwargs: Additional keyword arguments to pass to the model's `generate`
              method.

        Returns:
            A Pandas series containing the predicted values for the target column.
        """

        assert self.model_type == ModelType.tabular, (
            "The predict method is only implemented for tabular data..."
        )
        self._check_model()
        device = _validate_get_device(device)
        batch = min(batch, data.shape[0])

        # Clear the cache
        torch.cuda.empty_cache()

        tabular_sampler = TabularSampler.sampler_from_model(self, device=device)

        return tabular_sampler.predict(
            data=data,
            target_col=target_col,
            target_pos_val=target_pos_val,
            batch=batch,
            obs_sample=obs_sample,
            fillunk=fillunk,
            device=device,
            disable_progress_bar=disable_progress_bar,
            **generate_kwargs,
        )
