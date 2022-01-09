#!/usr/bin/env python

import numpy as np
import torch
import warnings

from tqdm import tqdm
from typing import Optional, List

from torch.utils.data import DataLoader
from transformers.trainer import Trainer
from transformers.trainer_utils import PredictionOutput, nested_concat, nested_numpify, distributed_concat, \
    EvalPrediction, distributed_broadcast_scalars
from transformers.utils import logging

logger = logging.get_logger(__name__)

class MyTrainer(Trainer):

    def prediction_loop(
            self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None,
            nbest: Optional[int] = 1
    ) -> PredictionOutput:
        if hasattr(self, "_prediction_loop"):
            warnings.warn(
                "The `_prediction_loop` method is deprecated and won't be called in a future version, define `prediction_loop` in your subclass.",
                FutureWarning,
            )
            return self._prediction_loop(dataloader, description, prediction_loss_only=prediction_loss_only)

        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        assert not getattr(
            self.model.config, "output_attentions", False
        ), "The prediction loop does not work with `output_attentions=True`."
        assert not getattr(
            self.model.config, "output_hidden_states", False
        ), "The prediction loop does not work with `output_hidden_states=True`."

        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        else:
            model = self.model
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        batch_size = dataloader.batch_size
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)
        eval_losses: List[float] = []
        label_ids: torch.Tensor = None
        model.eval()

        preds = []
        disable_tqdm = not self.is_local_process_zero() or self.args.disable_tqdm
        for inputs in tqdm(dataloader, desc=description, disable=disable_tqdm):
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only)
            # batch_size = inputs[list(inputs.keys())[0]].shape[0]
            # if loss is not None:
            #     eval_losses.extend([loss] * batch_size)
            # if logits is not None:
            #     preds = logits if preds is None else nested_concat(preds, logits, dim=0)
            # if labels is not None:
            #     label_ids = labels if label_ids is None else nested_concat(label_ids, labels, dim=0)
            input_ids = inputs["input_ids"]
            batch_size, seq_len = input_ids.shape
            batch_preds = []
            for batch_i in range(batch_size):
                row_preds = []
                for token_j in range(seq_len):
                    if input_ids[batch_i, token_j] != self.tokenizer.mask_token_id: continue
                    scores = logits[batch_i, token_j]
                    row_preds.append(scores.topk(nbest)[1])
                row_preds = torch.stack(row_preds, dim=0)
                batch_preds.append(row_preds)

            preds.extend(batch_preds)

        if self.args.local_rank != -1:
            # In distributed mode, concatenate all results from all nodes:
            if preds is not None:
                preds = distributed_concat(preds, num_total_examples=self.num_examples(dataloader))
            if label_ids is not None:
                label_ids = distributed_concat(label_ids, num_total_examples=self.num_examples(dataloader))

        # # Finally, turn the aggregated tensors into numpy arrays.
        # if preds is not None:
        #     preds = nested_numpify(preds)
        # if label_ids is not None:
        #     label_ids = nested_numpify(label_ids)
        #
        # if self.compute_metrics is not None and preds is not None and label_ids is not None:
        #     metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        # else:
        #     metrics = {}
        # if len(eval_losses) > 0:
        #     if self.args.local_rank != -1:
        #         metrics["eval_loss"] = (
        #             distributed_broadcast_scalars(eval_losses, num_total_examples=self.num_examples(dataloader))
        #                 .mean()
        #                 .item()
        #         )
        #     else:
        #         metrics["eval_loss"] = np.mean(eval_losses)
        #
        # # Prefix all keys with eval_
        # for key in list(metrics.keys()):
        #     if not key.startswith("eval_"):
        #         metrics[f"eval_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=None)