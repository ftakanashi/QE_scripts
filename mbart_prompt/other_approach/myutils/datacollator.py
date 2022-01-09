# coding: utf-8

import torch
from typing import List, Union, Dict
from dataclasses import dataclass

from torch.nn.utils.rnn import pad_sequence
from transformers.tokenization_utils import PreTrainedTokenizer, BatchEncoding

@dataclass
class DataCollatorForMaskedLanguageModeling:

    tokenizer: PreTrainedTokenizer
    with_src: bool = False

    def __call__(
            self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        batch_input_ids = [e["input_ids"] for e in examples]
        batch = self._tensorize_batch(batch_input_ids)
        batch_data = {"input_ids": batch}
        batch_data.update(self._generate_extra(batch))
        if "labels" in examples[0]:
            labels = self._tensorize_batch([e["labels"] for e in examples])
            mask_indices = batch == self.tokenizer.mask_token_id
            labels.masked_fill_(~mask_indices, -100)
            batch_data["labels"] = labels

        return batch_data

    def _tensorize_batch(
            self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> torch.Tensor:
        # In order to accept both lists of lists and lists of Tensors
        if isinstance(examples[0], (list, tuple)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

    def _generate_extra(self, batch):
        batch_size, seq_len = batch.shape
        pad = self.tokenizer.pad_token_id
        sep = self.tokenizer.sep_token_id
        extra = {}
        if self.with_src:
            language_ids = torch.zeros_like(batch)
            for batch_i in range(batch_size):
                flag = 0
                for seq_j in range(seq_len):
                    if batch[batch_i, seq_j] == pad: break
                    language_ids[batch_i, seq_j] = flag
                    if seq_j < seq_len-1 and batch[batch_i, seq_j] == sep and batch[batch_i, seq_j + 1] == sep:
                        flag = 1

            position_ids = torch.zeros_like(batch)
            for batch_i in range(batch_size):
                flag = 0
                for seq_j in range(seq_len):
                    if batch[batch_i, seq_j] == pad: break
                    position_ids[batch_i, seq_j] = flag
                    flag += 1
                    if seq_j < seq_len-1 and batch[batch_i, seq_j] == sep and batch[batch_i, seq_j + 1] == sep:
                        flag = 0

            extra["language_ids"] = language_ids
            extra["position_ids"] = position_ids

        attention_mask = torch.ones_like(batch)
        attention_mask[batch == self.tokenizer.pad_token_id] = 0

        extra["attention_mask"] = attention_mask
        return extra