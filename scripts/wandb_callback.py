from utils import precision_at_k, recall_at_k, mean_average_precision
from transformers.integrations import WandbCallback
import pandas as pd

import pandas as pd
import numpy as np
import wandb
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
from transformers.integrations import WandbCallback
from datasets import Dataset


class WandbPredictionProgressCallback(TrainerCallback):

    def __init__(
        self,
        trainer,
        sample_dataset: Dataset,
        num_samples: int = 100,
        freq: int = 1,
        ks: list[int] = [1, 5, 10],
        use_cosine: bool = True,
    ):
        super().__init__()
        self.trainer = trainer
        self.sample_ds = sample_dataset.select(range(num_samples))
        self.freq = freq
        self.ks = ks
        self.use_cosine = use_cosine

        self.all_cats = pd.read_csv("categories.csv", header=None)[0].tolist()
        cat_emb = self.trainer.model.encode(
            self.all_cats, convert_to_numpy=True
        )
        if use_cosine:
            cat_emb = cat_emb / np.linalg.norm(cat_emb, axis=1, keepdims=True)
        self.cat_emb = cat_emb

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict = None,
        **kwargs,
    ):
        WandbCallback().on_evaluate(args, state, control, metrics=metrics)

        if args.eval_steps is None or state.global_step % (args.eval_steps * self.freq) != 0:
            return

        df_flat = pd.DataFrame({
            "anchor":   self.sample_ds["anchor"],
            "positive": self.sample_ds["positive"],
        })
        grouped = (
            df_flat
            .groupby("anchor")["positive"]
            .apply(list)
            .reset_index()
            .rename(columns={"positive": "positives"})
        )

        anchors = grouped["anchor"].tolist()
        anc_emb = self.trainer.model.encode(
            anchors, convert_to_numpy=True
        )
        if self.use_cosine:
            anc_emb = anc_emb / np.linalg.norm(anc_emb, axis=1, keepdims=True)

        logits = anc_emb @ self.cat_emb.T 

        sorted_inds = np.argsort(-logits, axis=1)
        preds_str = [
            [ self.all_cats[idx] for idx in row_inds ]
            for row_inds in sorted_inds
        ]

        gt_lists = grouped["positives"].tolist()

        metric_dict = {}
        for k in self.ks:
            metric_dict[f"precision@{k}"] = precision_at_k(preds_str, gt_lists, k)
            metric_dict[f"recall@{k}"]    = recall_at_k(preds_str, gt_lists, k)
            metric_dict[f"map@{k}"]       = mean_average_precision(preds_str, gt_lists, k)

        records = []
        for i, anc in enumerate(anchors):
            rec = {
                "anchor":    anc,
                "positives": ", ".join(gt_lists[i]),
            }
            for k in self.ks:
                rec[f"preds@{k}"] = ", ".join(preds_str[i][:k])
            records.append(rec)

        table = wandb.Table(dataframe=pd.DataFrame(records))

        wandb.log(
            {
                "sample_predictions": table,
                **metric_dict,
            },
            step=state.global_step
        )

            