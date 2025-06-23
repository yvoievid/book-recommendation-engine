from utils import precision_at_k, recall_at_k, mean_average_precision
from transformers.integrations import WandbCallback
import pandas as pd

class WandbPredictionProgressCallback(WandbCallback):
    def __init__(self, trainer, tokenizer, val_dataset, num_samples=100, freq=2):
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.sample_dataset = val_dataset.select(range(num_samples))
        self.freq = freq

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control)

        if state.epoch is not None and state.epoch % self.freq == 0:
            pred_output = self.trainer.predict(self.sample_dataset)
            
            print(pred_output)
            df = pd.DataFrame(pred_output)
            df["epoch"] = state.epoch
            table = self._wandb.Table(dataframe=df)
            

            logits = (
                pred_output.predictions[0]
                if isinstance(pred_output.predictions, tuple)
                else pred_output.predictions
            )
            labels = pred_output.label_ids.tolist() if hasattr(pred_output.label_ids, "tolist") else pred_output.label_ids
            
            ks = [1, 5, 10]
            metric_dict = {}
            for k in ks:
                metric_dict[f"precision@{k}"] = precision_at_k(logits, labels, k)
                metric_dict[f"recall@{k}"]    = recall_at_k(logits, labels, k)
                metric_dict[f"map@{k}"]       = mean_average_precision(logits, labels, k)
            
            self._wandb.log(
                {
                    "sample_predictions": table,
                    **metric_dict
                },
                step=state.global_step
            )
            