import math
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import torch
from datasets import Dataset
from torch import nn
from transformers import (
    DataCollator,
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    logging,
)
from transformers.optimization import get_scheduler
import torch.nn.functional as F

from transformers.trainer import _is_peft_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from accelerate import is_accelerate_available
from accelerate import is_sagemaker_mp_enabled
from accelerate import smp_forward_backward
from accelerate import OptimizerNames

from .grokfast import gradfilter_ma, gradfilter_ema

if is_accelerate_available():
    from accelerate.utils import (
        DistributedType,
    )
    from accelerate.utils.memory import clear_device_cache

logger = logging.get_logger(__name__)


def weighted_causal_lm_loss(
    logits: torch.Tensor,  # (B, T, V)
    labels: torch.Tensor,  # (B, T)
    token_weights: torch.Tensor,  # (B, T) weights aligned to labels positions
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Weighted causal LM loss. Uses the standard GPT-style shift:
    logits[:, :-1] predicts labels[:, 1:].
    token_weights should align with labels positions (B, T),
    and we apply weights to the *predicted* label positions (i.e., labels[:, 1:]).
    """
    # Shift for causal LM
    shift_logits = logits[:, :-1, :].contiguous()  # (B, T-1, V)
    shift_labels = labels[:, 1:].contiguous()  # (B, T-1)
    shift_weights = token_weights[:, 1:].contiguous()  # (B, T-1)

    # Flatten
    B, Tp1, V = shift_logits.shape
    loss_per_tok = F.cross_entropy(
        shift_logits.view(-1, V),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=ignore_index,
    ).view(B, Tp1)

    # Zero-out weights where labels are ignored
    active = (shift_labels != ignore_index).float()
    w = shift_weights * active

    denom = w.sum().clamp_min(1e-12)
    return (loss_per_tok * w).sum() / denom


@dataclass
class WeightedLabelSmoother:
    """
    Label smoothing with optional per-token weights.

    token_weights: (B, T) aligned with labels positions.
    If shift_labels=True (causal LM), weights are shifted too.
    """

    epsilon: float = 0.1
    ignore_index: int = -100

    def __call__(
        self, model_output, labels, shift_labels: bool = False, token_weights=None
    ):
        logits = (
            model_output["logits"]
            if isinstance(model_output, dict)
            else model_output[0]
        )
        # logits: (B, T, V)

        if shift_labels:
            logits = logits[..., :-1, :].contiguous()  # (B, T-1, V)
            labels = labels[..., 1:].contiguous()  # (B, T-1)
            if token_weights is not None:
                token_weights = token_weights[
                    ..., 1:
                ].contiguous()  # align with shifted labels

        log_probs = -nn.functional.log_softmax(logits, dim=-1)  # (B, T, V)

        # Make labels shape (B, T, 1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(self.ignore_index)  # (B, T, 1)

        # gather requires non-negative indices
        labels_clamped = torch.clamp(labels, min=0)

        # Per-token NLL (B, T, 1)
        nll_loss = log_probs.gather(dim=-1, index=labels_clamped)

        # Per-token "smoothed" loss (B, T, 1): average negative log-prob over vocab
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

        # Zero out ignored positions
        nll_loss = nll_loss.masked_fill(padding_mask, 0.0)
        smoothed_loss = smoothed_loss.masked_fill(padding_mask, 0.0)

        # Build per-token weights (B, T, 1)
        if token_weights is None:
            w = (~padding_mask).float()  # 1 for active tokens, 0 for ignored
        else:
            w = token_weights.to(device=logits.device, dtype=torch.float).unsqueeze(-1)
            w = w.masked_fill(padding_mask, 0.0)

        denom = w.sum().clamp_min(1e-12)

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded).
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()

        nll = (nll_loss * w).sum() / denom / num_active_elements
        smooth = (
            (smoothed_loss * w).sum()
            / denom
            / (num_active_elements * log_probs.shape[-1])
        )

        return (1 - self.epsilon) * nll + self.epsilon * smooth


class SaveEpochEndCallback(TrainerCallback):
    """This callback forces a checkpoint save at each epoch end."""

    def __init__(self, save_epochs: int = None) -> None:
        super().__init__()

        self.save_epochs = save_epochs

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self.save_epochs is not None:
            control.should_save = math.ceil(state.epoch) % self.save_epochs == 0
        else:
            control.should_save = True

        return control


class ResumableTrainer(Trainer):
    """This trainer makes the scheduler consistent over pauses
    in the training. The scheduler should return values similar
    to when a training is done either intermittently or continuously
    over the `target_epochs`.
    """

    def __init__(
        self,
        target_epochs: int = None,
        save_epochs: int = None,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = None,
        compute_loss_func: Callable | None = None,
        grokfast_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        # Declare here for typing
        self.lr_scheduler: torch.optim.lr_scheduler.LambdaLR = None

        if callbacks is None:
            callbacks = []

        callbacks.append(SaveEpochEndCallback(save_epochs=save_epochs))
        self._vocab_size: int = model.config.vocab_size

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            compute_loss_func=compute_loss_func,
            **kwargs,
        )
        self.target_epochs = target_epochs
        self._grads = None  # Grokfast
        self._grokfast_args = grokfast_args
        self._grokfast_trigger = False  # Grokfast

    def create_scheduler(
        self, num_training_steps: int, optimizer: torch.optim.Optimizer = None
    ) -> torch.optim.lr_scheduler.LambdaLR:
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.
        Args:
            num_training_steps (int): The number of training steps to do.
        """

        if self.lr_scheduler is None:
            if self.target_epochs is not None:
                # Compute the max_steps based from the
                # `target_epochs`.
                train_dataloader = self.get_train_dataloader()
                len_dataloader = len(train_dataloader)
                num_update_steps_per_epoch = (
                    len_dataloader // self.args.gradient_accumulation_steps
                )
                num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)

                max_steps = math.ceil(self.target_epochs * num_update_steps_per_epoch)
                num_training_steps = max_steps

            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )

        return self.lr_scheduler

    # def set_weighted_compute_loss_func(self):
    #     def _func(outputs, labels, num_items_in_batch):

    #     self._compute_loss_func =
    #     , model, inputs, return_outputs=False, **kwargs):

    # def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
    #     """
    #     If `token_weights` is provided in the batch, use weighted token-level loss.
    #     Otherwise fall back to the model's default loss (Trainer default behavior).
    #     """
    #     token_weights = inputs.pop("token_weights", None)

    #     outputs = model(**inputs)
    #     logits = outputs.logits

    #     labels = inputs.get("labels", None)
    #     if labels is None:
    #         # No labels => can't compute LM loss
    #         loss = (
    #             outputs.loss
    #             if hasattr(outputs, "loss") and outputs.loss is not None
    #             else None
    #         )
    #         if loss is None:
    #             raise ValueError(
    #                 "No `labels` in inputs and model didn't return `loss`."
    #             )
    #         return (loss, outputs) if return_outputs else loss

    #     if token_weights is None:
    #         # Default Hugging Face causal LM loss from the model
    #         loss = outputs.loss
    #         # Some models only compute loss when labels are passed; GPT2LMHeadModel does.
    #         if loss is None:
    #             # As a fallback, compute unweighted loss ourselves
    #             token_weights = torch.ones_like(
    #                 labels, dtype=torch.float, device=labels.device
    #             )
    #             loss = weighted_causal_lm_loss(logits, labels, token_weights)
    #     else:
    #         # Ensure float + correct device
    #         token_weights = token_weights.to(device=labels.device, dtype=torch.float)
    #         loss = weighted_causal_lm_loss(logits, labels, token_weights)

    #     return (loss, outputs) if return_outputs else loss

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ):
        """
        https://github.com/huggingface/transformers/blob/37974267efefe020168ff27081fbab8bbce04720/src/transformers/trainer.py#L3755
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Args:
            model (`nn.Module`):
                The model to compute the loss for.
            inputs (`dict[str, Union[torch.Tensor, Any]]`):
                The input data for the model.
            return_outputs (`bool`, *optional*, defaults to `False`):
                Whether to return the model outputs along with the loss.
            num_items_in_batch (Optional[torch.Tensor], *optional*):
                The number of items in the batch. If num_items_in_batch is not passed,

        Returns:
            The loss of the model along with its output if return_outputs was set to True

        Subclass and override for custom behavior. If you are not using `num_items_in_batch` when computing your loss,
        make sure to overwrite `self.model_accepts_loss_kwargs` to `False`. Otherwise, the loss calculating might be slightly inaccurate when performing gradient accumulation.
        """
        pc = getattr(self.accelerator, "parallelism_config", None)
        if pc is not None and pc.sp_backend == "deepspeed" and pc.sp_enabled:
            return self._deepspeed_sp_compute_loss(model, inputs, return_outputs, pc)

        if (
            self.label_smoother is not None or self.compute_loss_func is not None
        ) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            kwargs = {}
            if num_items_in_batch is not None:
                kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **kwargs}

        # print(f"inputs: {inputs}")
        outputs = model(**inputs)

        # print(f"outputs: {outputs}")

        # User-defined compute_loss function
        if self.compute_loss_func is not None:
            if labels is None:
                logger.warning(
                    "Trainer: `compute_loss_func` is defined but `labels=None`. "
                    "Your custom loss function will still be called with labels=None. "
                )
            loss = self.compute_loss_func(
                outputs,
                labels,
                num_items_in_batch=num_items_in_batch,
                inputs=inputs,
                vocab_size=self._vocab_size,
            )
        # Default HF loss handling (label smoothing) if no custom loss function
        elif labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            model_name = (
                unwrapped_model.base_model.model._get_name()
                if _is_peft_model(unwrapped_model)
                else unwrapped_model._get_name()
            )
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            # _loss = loss.detach().cpu().tolist()

            loss *= (
                self.accelerator.num_processes
                if self.args.n_gpu <= 1
                else self.args.n_gpu
            )

            # print(f"loss: {_loss} -> {loss}")

        return (loss, outputs) if return_outputs else loss

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        # Prepare buffers for context parallelism

        cp_context, inputs = self._prepare_context_parallel_inputs(model, inputs)

        # Context manager is no-op if CP isn't enabled
        with cp_context():
            model.train()
            if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()

            inputs = self._prepare_inputs(inputs)
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(
                    model, inputs, self.args.gradient_accumulation_steps
                )
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(
                    model, inputs, num_items_in_batch=num_items_in_batch
                )

            del inputs
            if (
                self.args.torch_empty_cache_steps is not None
                and self.state.global_step % self.args.torch_empty_cache_steps == 0
            ):
                clear_device_cache()

            kwargs = {}

            # For LOMO optimizers you need to explicitly use the learning rate
            if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
                kwargs["learning_rate"] = self._get_learning_rate()

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            # Finally we need to normalize the loss for reporting if GA loss bug is not fixed during compute loss
            if (
                not self.model_accepts_loss_kwargs or num_items_in_batch is None
            ) and self.compute_loss_func is None:
                # If the model does not accept loss kwargs, we need to normalize the loss by the number of gradient accumulation steps
                loss = loss / self.current_gradient_accumulation_steps

            # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
            # https://github.com/huggingface/transformers/pull/35808
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False

            self.accelerator.backward(loss, **kwargs)

            if self._grokfast_args is not None:
                grokfast_filter = self._grokfast_args.get("filter", "none")
                grokfast_window_size = self._grokfast_args.get("window_size", 10)
                grokfast_lamb = self._grokfast_args.get("lamb", 0.1)
                grokfast_alpha = self._grokfast_args.get("alpha", 0.1)
                if grokfast_filter == "none":
                    pass
                elif grokfast_filter == "ma":
                    self._grads = gradfilter_ma(
                        model,
                        grads=self._grads,
                        window_size=grokfast_window_size,
                        lamb=grokfast_lamb,
                        trigger=self._grokfast_trigger,
                    )
                elif grokfast_filter == "ema":
                    self._grads = gradfilter_ema(
                        model,
                        grads=self._grads,
                        alpha=grokfast_alpha,
                        lamb=grokfast_lamb,
                    )
                else:
                    raise ValueError(
                        f"Invalid gradient filter type `{grokfast_filter}`"
                    )

            return loss.detach()
