import math
from typing import Callable, Dict, List, Optional, Tuple, Union

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

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        If `token_weights` is provided in the batch, use weighted token-level loss.
        Otherwise fall back to the model's default loss (Trainer default behavior).
        """
        token_weights = inputs.pop("token_weights", None)

        outputs = model(**inputs)
        logits = outputs.logits

        labels = inputs.get("label_ids", None)
        if labels is None:
            # No label_ids => can't compute LM loss
            loss = (
                outputs.loss
                if hasattr(outputs, "loss") and outputs.loss is not None
                else None
            )
            if loss is None:
                raise ValueError(
                    "No `label_ids` in inputs and model didn't return `loss`."
                )
            return (loss, outputs) if return_outputs else loss

        if token_weights is None:
            # Default Hugging Face causal LM loss from the model
            loss = outputs.loss
            # Some models only compute loss when label_ids are passed; GPT2LMHeadModel does.
            if loss is None:
                # As a fallback, compute unweighted loss ourselves
                token_weights = torch.ones_like(
                    labels, dtype=torch.float, device=labels.device
                )
                loss = weighted_causal_lm_loss(logits, labels, token_weights)
        else:
            # Ensure float + correct device
            token_weights = token_weights.to(device=labels.device, dtype=torch.float)
            loss = weighted_causal_lm_loss(logits, labels, token_weights)

        return (loss, outputs) if return_outputs else loss
