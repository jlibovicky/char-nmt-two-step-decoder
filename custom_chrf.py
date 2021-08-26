from typing import List

from collections import Counter
import string

import numpy as np


def pairwise_chrf(sentences: List[str], order: int = 6, beta: float = 2.0):
    # 1. represent each sentece as n-grams
    sentences = [
        s.translate(str.maketrans("", "", string.whitespace))
        for s in sentences]
    sent_n_grams = [[
        Counter([sent[i:i + o]
            for i in range(len(sent) - o + 1)])
        for o in range(1, order + 1)]
        for sent in sentences]

    # 2. prepare precision table
    precisions = np.ones((len(sentences), len(sentences)))

    # 3. compute the precisions
    for i, sent_a in enumerate(sent_n_grams):
        for j, sent_b in enumerate(sent_n_grams):
            if i >= j:
                continue

            avg_precision = 0.0
            avg_recall = 0.0
            effective_order = 0

            for ngrams_a, ngrams_b in zip(sent_a, sent_b):
                a_count = sum(ngrams_a.values())
                b_count = sum(ngrams_b.values())
                common_count = sum((ngrams_a & ngrams_b).values())

                if a_count > 0 and b_count > 0:
                    avg_precision += common_count / a_count
                    avg_recall += common_count / b_count
                    effective_order += 1

            if effective_order == 0:
                avg_precision, avg_recall = 0.0, 0.0
            else:
                avg_precision /= effective_order
                avg_recall /= effective_order
            precisions[i, j] = avg_precision
            precisions[j, i] = avg_recall


    # 4. recall is transposed precision
    recalls = precisions.T

    # 5. compute score
    beta_sq = beta ** 2
    scores = (
        (1 + beta_sq) * precisions * recalls /
        ((beta_sq * precisions) + recalls))

    # 6. masked outliers
    scores = np.where(
        (precisions == 0) + (recalls == 0),
        np.zeros((len(sentences), len(sentences))),
        scores)

    return scores

