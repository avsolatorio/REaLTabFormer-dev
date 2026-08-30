from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch


@dataclass
class RelationalDataCollator:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Adopted from the DataCollatorForSeq2Seq:
     https://github.com/huggingface/transformers/blob/v4.24.0/src/transformers/data/data_collator.py#L510

    Args:
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        labels = (
            [feature["labels"] for feature in features]
            if "labels" in features[0].keys()
            else None
        )

        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(label) for label in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            for feature in features:
                remainder = [self.label_pad_token_id] * (
                    max_label_length - len(feature["labels"])
                )
                if isinstance(feature["labels"], list):
                    feature["labels"] = feature["labels"] + remainder
                else:
                    # Pad always at the right.
                    feature["labels"] = np.concatenate(
                        [feature["labels"], remainder]
                    ).astype(np.int64)

        labels = [feature["labels"] for feature in features]
        input_ids = [feature["input_ids"] for feature in features]

        if return_tensors == "np":
            labels = np.vstack(labels)
            input_ids = np.vstack(input_ids)
        elif return_tensors == "pt":
            labels = torch.vstack([torch.tensor(label) for label in labels])
            input_ids = torch.vstack([torch.tensor(ii) for ii in input_ids])
        elif return_tensors == "tf":
            raise ValueError("Tensorflow tensor is not supported yet.")

        return dict(
            labels=labels,
            input_ids=input_ids,
        )


@dataclass
class AnyOrderColumnCollator:
    """REaLTabFormerV2-only (`any_order=True`, requires `shared_numeric_vocab=True`):
    permutes each row's *original*-column order on the fly, once per batch,
    so training sees a different random column order essentially every time
    a row is drawn -- the "any-order" training signal that lets
    `_process_seed_input` (rtf_sampler.py) condition on an arbitrary subset
    of columns at inference time, not just a prefix of the fixed order.

    Unlike `RelationalDataCollator`, no padding logic is needed here: v2
    tabular rows from `make_dataset_with_column_types` are already
    fixed-length (`tabular_max_length`), via `.set_format(type="torch",
    ...)` -- `features` arrives as a list of dicts of already-equal-length
    1D tensors, one dict per row.

    `column_blocks` (`data_utils.compute_column_blocks(processed_columns)`)
    groups every processed column into its *original*-column block --
    numeric/datetime partition sub-columns (`price_00`, `price_01`, ...)
    always move together, in their existing internal order; only the order
    of blocks is permuted. Position 0 (BOS) and the last position (EOS)
    are never touched.

    `seed`: seeds a dedicated `numpy.random.Generator`, independent of the
    legacy global `random`/`np.random` state (same rationale as
    `data_utils.dataset.make_dataset`'s RNG), advancing across calls so
    consecutive batches draw different permutations rather than repeating
    one. Under multi-worker `DataLoader`s (`dataloader_num_workers > 0`),
    each worker process gets its own pickled copy of this collator and
    therefore its own independent RNG stream -- every batch still gets a
    valid random permutation, it just isn't one single globally-advancing
    sequence across workers, which doesn't matter for what this is used
    for (order robustness in expectation, not a reproducible draw
    sequence).
    """

    column_blocks: List[List]
    seed: Optional[int] = None
    return_tensors: str = "pt"
    _rng: np.random.Generator = field(init=False, repr=False)
    _block_lengths: np.ndarray = field(init=False, repr=False)
    _block_id_per_pos: np.ndarray = field(init=False, repr=False)
    _local_offset_per_pos: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

        block_lengths = []
        block_id_per_pos = []
        local_offset_per_pos = []
        for block_id, (_name, indices) in enumerate(self.column_blocks):
            block_lengths.append(len(indices))
            for local_offset in range(len(indices)):
                block_id_per_pos.append(block_id)
                local_offset_per_pos.append(local_offset)

        self._block_lengths = np.array(block_lengths, dtype=np.int64)
        self._block_id_per_pos = np.array(block_id_per_pos, dtype=np.int64)
        self._local_offset_per_pos = np.array(local_offset_per_pos, dtype=np.int64)

    def _sample_gather_index(self, batch_size: int) -> np.ndarray:
        """Returns `(batch_size, middle_len)`, where `index[b, new_pos]` is
        the *canonical* middle-slice position whose value should land at
        `new_pos` for row `b` -- i.e. exactly what `torch.gather(dim=1,
        index=...)` needs.

        Fully vectorized over the batch (no Python loop over rows): derives
        the forward mapping "each canonical position's content moves to
        new_pos_of_canonical[b, p]" from a random per-row block
        permutation, then inverts it per row via `argsort` to get the
        gather index `torch.gather` actually needs.
        """
        n_blocks = len(self._block_lengths)

        # A random permutation of block order per row.
        perm = np.argsort(self._rng.random((batch_size, n_blocks)), axis=1)

        permuted_lengths = self._block_lengths[perm]
        slot_start = np.cumsum(permuted_lengths, axis=1) - permuted_lengths

        # inv_perm[b, block_id] = the slot index block_id lands in for row b.
        inv_perm = np.argsort(perm, axis=1)
        new_start_of_block = np.take_along_axis(slot_start, inv_perm, axis=1)

        # For every canonical position p, where does its content move to?
        new_pos_of_canonical = (
            new_start_of_block[:, self._block_id_per_pos]
            + self._local_offset_per_pos[None, :]
        )

        # Invert per row: gather_index[b, new_pos] = canonical_pos.
        return np.argsort(new_pos_of_canonical, axis=1)

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        if return_tensors != "pt":
            raise ValueError(
                f"AnyOrderColumnCollator only supports return_tensors='pt', got {return_tensors!r}."
            )

        keys = [
            k
            for k in ("input_ids", "labels", "token_type_ids", "token_weights")
            if k in features[0]
        ]
        batch = {k: torch.stack([f[k] for f in features]) for k in keys}

        batch_size, seq_len = batch["input_ids"].shape
        middle_len = seq_len - 2
        assert middle_len == len(self._block_id_per_pos), (
            f"Row length {seq_len} (middle_len={middle_len}) doesn't match "
            f"column_blocks' {len(self._block_id_per_pos)} positions -- "
            "column_blocks must come from the same processed_columns this "
            "dataset was built from."
        )

        gather_index = torch.from_numpy(self._sample_gather_index(batch_size)).long()

        for k in keys:
            batch[k][:, 1:-1] = torch.gather(batch[k][:, 1:-1], 1, gather_index)

        return batch
