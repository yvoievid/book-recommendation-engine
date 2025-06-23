import numpy as np

def precision_at_k(logits: np.ndarray, gt: list[list[int]], k: int) -> float:
    topk = np.argsort(-logits, axis=1)[:, :k]
    p = [ len(set(pred)&set(true))/k for pred,true in zip(topk, gt) ]
    return float(np.mean(p))

def recall_at_k(logits: np.ndarray, gt: list[list[int]], k: int) -> float:
    topk = np.argsort(-logits, axis=1)[:, :k]
    r = [ len(set(pred)&set(true))/(len(true) or 1) for pred,true in zip(topk, gt) ]
    return float(np.mean(r))

def mean_average_precision(logits: np.ndarray, gt: list[list[int]], k: int = None) -> float:
    sorted_inds = np.argsort(-logits, axis=1)
    APs = []
    for preds, true in zip(sorted_inds, gt):
        true_set = set(true)
        if not true_set:
            APs.append(1.0)
            continue
        if k is not None:
            preds = preds[:k]
        hits = 0; score = 0.0
        for rank,p in enumerate(preds, start=1):
            if p in true_set:
                hits += 1
                score += hits/rank
        APs.append(score/len(true_set))
    return float(np.mean(APs))
