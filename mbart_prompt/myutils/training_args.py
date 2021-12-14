#!/usr/bin/env python

import contextlib
import torch

from dataclasses import dataclass, field
from transformers import logging
from transformers.training_args import TrainingArguments
from transformers.file_utils import is_torch_available
from typing import Optional

logger = logging.get_logger(__name__)

@dataclass
class MyTrainingArguments(TrainingArguments):
    '''
    Some adaption applied for new transformer version
    '''

    label_smoothing_factor: float = field(
        default=0.0, metadata={"help": "The label smoothing epsilon to apply (zero means no label smoothing)."}
    )

    @contextlib.contextmanager
    def main_process_first(self, local=True, desc="work"):
        if is_torch_available() and self.world_size > 1:
            if local:
                is_main_process = self.local_process_index == 0
                main_process_desc = "main local process"
            else:
                is_main_process = self.process_index == 0
                main_process_desc = "main process"

            try:
                if not is_main_process:
                    # tell all replicas to wait
                    logger.debug(f"{self.process_index}: waiting for the {main_process_desc} to perform {desc}")
                    torch.distributed.barrier()
                yield
            finally:
                if is_main_process:
                    # the wait is over
                    logger.debug(f"{self.process_index}: {main_process_desc} completed {desc}, releasing all replicas")
                    torch.distributed.barrier()
        else:
            yield

    @property
    def world_size(self):
        """
        The number of processes used in parallel.
        """
        if self.local_rank != -1:
            return torch.distributed.get_world_size()
        return 1

@dataclass
class Seq2SeqTrainingArguments(MyTrainingArguments):

    sortish_sampler: bool = field(default=False, metadata={"help": "Whether to use SortishSampler or not."})
    predict_with_generate: bool = field(
        default=False, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
    generation_max_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The `max_length` to use on each evaluation loop when `predict_with_generate=True`. Will default "
            "to the `max_length` value of the model configuration."
        },
    )
    generation_num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "The `num_beams` to use on each evaluation loop when `predict_with_generate=True`. Will default "
            "to the `num_beams` value of the model configuration."
        },
    )