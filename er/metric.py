"""Official metric: macro-averaged F0.5 over Source-1 entities (singletons included)."""
import numpy as np


def f_beta_entity(pred: set, true: set, beta=0.5):
    if not true and not pred:
        return 1.0
    if not true or not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def macro_f05(pred: dict, truth: dict):
    """pred/truth: {s1_id: set(ids)}; averaged over the keys of `truth`."""
    return float(np.mean([f_beta_entity(pred.get(k, set()), v) for k, v in truth.items()]))


if __name__ == '__main__':
    # example from the problem statement -> 0.714
    print(round(f_beta_entity({'a', 'b', 'c'}, {'a', 'c'}), 3))
