from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses
)

import wandb
from wandb_callback import WandbPredictionProgressCallback
from pathlib import Path
from dotenv import load_dotenv


def load_env():
    # look for a .env file in your project root
    env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=env_path)


def not_null(ex):
    return ex["anchor"] is not None and ex["positive"] is not None

def train():
    load_env()
    wandb.login()
    
    train_ds = load_dataset("csv", data_files="../data/train_val_split/train.csv")["train"]
    val_ds   = load_dataset("csv", data_files="../data/train_val_split/val.csv")["val"]

    train_ds = train_ds.filter(not_null)
    val_ds   = val_ds.filter(not_null)
    
    model = SentenceTransformer("microsoft/mpnet-base")
    loss  = losses.MultipleNegativesSymmetricRankingLoss(model)

    args = SentenceTransformerTrainingArguments(
        output_dir="./models/microsoft_mpnet-base",
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        logging_steps=200,
        save_steps=3000,
        eval_steps=200,
        eval_strategy="steps",
        report_to="wandb",
    )

    trainer = SentenceTransformerTrainer(
        args=args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        loss=loss,
        
    )

    evals_cb = WandbPredictionProgressCallback(
        trainer=trainer,
        sample_dataset=val_ds,
        num_samples=500,
        freq=1,
    )
    trainer.add_callback(evals_cb)
    trainer.train()

if __name__ == "__main__":
    load_env()
    train()