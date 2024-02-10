import numpy as np
from offlinerllib.utils.functional import discounted_cum_sum
from operator import itemgetter


def compute_gae(rewards, values, last_v, gamma=0.99, lam=0.97, dim=0):
    values = np.concatenate([values, last_v], axis=dim)
    seq_len = values.shape[dim]
    deltas = rewards + gamma * np.take(values, np.arange(1, seq_len), dim) - np.take(values, np.arange(0, seq_len-1), dim)
    # 检查计算delta的过程是否会导致reward之类的发生变化
    gae = discounted_cum_sum(deltas, gamma * lam)
    ret = gae + np.take(values, np.arange(0, seq_len-1), dim)
    # gae = (gae - gae.mean()) / (gae.std() + 1e-8)
    return gae, ret