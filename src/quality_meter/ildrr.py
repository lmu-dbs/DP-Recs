import math

import pandas as pd

from src.quality_meter.quality_measure import QualityMeasure


class IldRR(QualityMeasure):
    def __init__(self, y):
        super().__init__(y)
        self.mean_ild_rr = None
        # self.recommendations = []

    def add_sequence(self, sequence=None, recommendations: pd.Series = None):
        pass

    # Adapted from https://github.com/alirezagharahi/d_SBRS/blob/main/models/sknn.py
    def add_sequence_succ(self, sequence, gt_sequence, recommendations: pd.Series = None):
        # print("Adding sequence")
        negative_sample = 0.01
        avg_dists = []
        disc_weights = []
        # print(f"Recommendations {recommendations.tolist()}")
        recs = recommendations.tolist()
        if len(recs) <= 1:  # there is nothing to compute if there is one recommendation at the maximum
            return
        for i in range(len(recs) - 1):
            dists = []
            weights = []
            itemA = recs[i]
            for j in range(i + 1, len(recs)):
                itemB = recs[j]
                dist = 0.0 if itemA == itemB else 1
                relevance_j = 1 if recs[j] == gt_sequence[len(sequence)-1] else negative_sample
                rel_discount = self.log_rank_discount(max(0, j - i - 1))
                dists.append(dist * rel_discount * relevance_j)
                weights.append(rel_discount * relevance_j)

            avg_dists_i = sum(dists) / float(sum(weights))

            # Weights item by relevance
            relevance_i = 1 if recs[i] == gt_sequence[len(sequence)-1] else negative_sample

            # Logarithmic rank discount, to prioritize more diverse items in the top of the list
            rank_discount_i = self.log_rank_discount(i)
            avg_dists.append(avg_dists_i * rank_discount_i * relevance_i)
            disc_weights.append(rank_discount_i)

        # Expected Intra-List Diversity (EILD) with logarithmic rank discount
        # From "Incorporating Diversity in a Learning to Rank Recommender System" (2016)
        # print(avg_dists)
        # print(disc_weights)
        avg_cos_dist = sum(avg_dists) / float(sum(disc_weights))

        self.qualities.append(avg_cos_dist)
        # print(self.qualities)



    def compute_quality(self):
        self.mean_quality = sum(self.qualities) / len(self.qualities)

    def log_rank_discount(self, k):
        return 1. / math.log2(k + 2)
