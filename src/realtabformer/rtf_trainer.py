import math
from typing import Callable, Dict, List, Optional, Tuple, Union, Any

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
    ):
        # Declare here for typing
        self.lr_scheduler: torch.optim.lr_scheduler.LambdaLR = None

        if callbacks is None:
            callbacks = []

        callbacks.append(SaveEpochEndCallback(save_epochs=save_epochs))

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
        )
        self.target_epochs = target_epochs

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
        outputs = model(**inputs)

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
            loss *= (
                self.accelerator.num_processes
                if self.args.n_gpu <= 1
                else self.args.n_gpu
            )

        return (loss, outputs) if return_outputs else loss
