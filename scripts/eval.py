import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from utils import precision_at_k, recall_at_k, mean_average_precision

def evaluate(
    test_csv_path: str = "../data/train_val_split/test.csv",
    model_path: str    = "yuriivoievidka/microsoft_mpnet-base-librarian"
):
    df_pairs = pd.read_csv(test_csv_path)
    assert {'anchor','positive'}.issubset(df_pairs.columns), \
        "CSV must contain 'anchor' and 'positive' columns"
    
    grouped = (
        df_pairs
        .groupby('anchor')['positive']
        .apply(list)
        .reset_index(name='CategoryList')
    )
    
    titles       = grouped['anchor'].tolist()
    ground_truth = grouped['CategoryList'].tolist()
    
    all_cats = sorted(df_pairs['positive'].unique())
    
    model     = SentenceTransformer(model_path)
    title_emb = model.encode(titles,    convert_to_numpy=True)
    cat_emb   = model.encode(all_cats,  convert_to_numpy=True)
    
    title_emb_norm = title_emb / np.linalg.norm(title_emb, axis=1, keepdims=True)
    cat_emb_norm   = cat_emb   / np.linalg.norm(cat_emb,   axis=1, keepdims=True)
    logits = np.dot(title_emb_norm, cat_emb_norm.T)  # [n_titles, n_cats]
    
    cat2idx = {cat: idx for idx, cat in enumerate(all_cats)}
    ground_truth_idx = [
        [cat2idx[cat] for cat in cats]
        for cats in ground_truth
    ]
    
    metrics = {
        'P@1':   precision_at_k(logits, ground_truth_idx, k=1),
        'P@5':   precision_at_k(logits, ground_truth_idx, k=5),
        'P@10':   precision_at_k(logits, ground_truth_idx, k=10),
        'R@1':   recall_at_k   (logits, ground_truth_idx, k=1), 
        'R@5':   recall_at_k   (logits, ground_truth_idx, k=5),
        'R@10':   recall_at_k   (logits, ground_truth_idx, k=10),
        'mAP':    mean_average_precision(logits, ground_truth_idx),
        'mAP@15': mean_average_precision(logits, ground_truth_idx, k=15),
    }
    
    for name, val in metrics.items():
        print(f"{name:6s}: {val:.4f}")


if __name__ == "__main__":
    evaluate()
