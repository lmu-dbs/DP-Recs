import collections
import itertools
import logging
import math

import pandas as pd

from src.quality_meter.quality_measure import QualityMeasure

logger = logging.getLogger(__name__)


class Div(QualityMeasure):
    def __init__(self, y):
        super().__init__(y)
        self.mean_idiv_change = None
        self.mean_inter_div_pairwise = None
        self.mean_inter_div_set_theoretic = None
        self.recommendations = []

    def add_sequence(self, sequence=None, recommendations: pd.Series = None):
        # print(f"Number of combinations: {len(list(itertools.combinations(recommendations, r=2)))}")
        divs = []
        # print(recommendations)
        self.recommendations.append(recommendations)
        if len(recommendations) > 1:
            for comb in itertools.combinations(recommendations, r=2):
                divs.append(1 - (len(self.intersection(comb[0], comb[1])) / len(self.union(comb[0], comb[1]))))
            self.qualities.append(sum(divs) / len(divs))
        # print(self.mean_divs)

    def compute_quality(self):
        self.mean_quality = sum(self.qualities) / len(self.qualities)

    @staticmethod
    def union(lst1, lst2):
        final_list = list(set(lst1) | set(lst2))
        return final_list

    @staticmethod
    def intersection(lst1, lst2):
        return list(set(lst1) & set(lst2))

    def reset_values(self):
        self.qualities = []
        self.mean_quality = None
        self.mean_inter_div_pairwise = None
        self.mean_inter_div_set_theoretic = None

    @staticmethod
    def is_equal(l1, l2):
        if len(l1) != len(l2):
            return False
        return collections.Counter(l1) == collections.Counter(l2)

    def compute_inter_div_set_theoretic(self, topk):
        inter_divs = []
        # logger.info(self.recommendations)
        for succ_a, succ_b in zip(self.recommendations, self.recommendations[1:]):
            # print(f"First reclist\n{succ_a}")
            # print(f"Second reclist\n{succ_b}")
            merge_df = succ_b.to_frame().merge(succ_a.to_frame(), how='left', indicator=True)
            merge_df = merge_df[merge_df['_merge'] == 'left_only']
            merge_df = merge_df[['consequent']]
            inter_divs.append(len(merge_df.index)/topk)

        if len(inter_divs) > 0:
            self.mean_inter_div_set_theoretic = sum(inter_divs) / (len(inter_divs))


    def compute_inter_div_pairwise(self):
        inter_divs = []
        # logger.info(self.recommendations)
        for succ_a, succ_b in zip(self.recommendations, self.recommendations[1:]):
            # print(f"First reclist\n{succ_a}")
            # print(f"Second reclist\n{succ_b}")
            rec_comb_equality = []
            for x, y in zip(succ_a, succ_b):
                rec_comb_equality.append(self.is_equal(x, y))
            inter_divs.append(rec_comb_equality.count(True) / len(rec_comb_equality))


        # changes = []
        if len(inter_divs) > 0:
            self.mean_inter_div_pairwise = sum(inter_divs) / (len(inter_divs))

            # ypoints = np.array(inter_divs)
            # plt.plot(ypoints, linestyle='dotted')
            # plt.show()

            # Change rate
            # for idiv1, idiv2 in zip(inter_divs, inter_divs[1:]):
            #     if idiv1 == 0:
            #         changes.append(1)
            #     else:
            #         changes.append((idiv2 - idiv1) / idiv1)
            # if len(changes) > 1:
            #     self.mean_idiv_change = sum(changes) / (len(changes))

        else:
            self.mean_inter_div_pairwise = 0
