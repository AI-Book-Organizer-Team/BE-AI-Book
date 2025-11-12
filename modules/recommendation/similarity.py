import numpy as np

def l2norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-10
    return v / n

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(l2norm(a), l2norm(b)))

def mean_vector(vecs: list[np.ndarray]) -> np.ndarray:
    return l2norm(np.mean(np.vstack(vecs), axis=0))
