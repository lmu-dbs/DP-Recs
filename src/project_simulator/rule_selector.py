import logging
import random

import numpy as np
import pandas as pd
from scipy import integrate
from sklearn.neighbors import KernelDensity

import utils
from constants import DELTA
from src.enums.selection_methods import SelectionMethods

logger = logging.getLogger(__name__)


class RuleSelector:
    """
    1. Filters rules which antecedent matches to the given sequence that already exists
    2. Selects rule which consequent will be used to expand the given sequence by
        a) Sampling one rule randomly
        b) Taking the first rule by using the sorted list of rules (sort by support and confidence)
    * Implement dynamic ranking with discrete gaps
        - Number of gaps is required
    * Implement dynamic ranking with continuous gaps
        - Timestamps at which the events are activated are required
    Todo
    * Weight the rules to be applied by frequency with which the consequent occurs in the training set
    """

    def __init__(self, topk, bandwidth, rules, sequence=None, sequence_ts=None, method=None):
        self.method = method
        self.sequence = sequence
        self.sequence_ts = sequence_ts
        self.bandwidth = bandwidth
        self.rules = rules
        self.rules.reset_index(drop=True, inplace=True)
        self.stepwise_rules = None
        self.selection_size = []
        self.topk = topk
        if method == SelectionMethods.FIRST:
            self.rules = self.rules.nlargest(1, "support")
        # elif method == SelectionMethods.NAIVE:
        #     self.rules = self.rules.nlargest(topk, "support")

    def __iter__(self):
        return self

    def __next__(self):
        # logger.info("Copying rules")
        # if self.method == SelectionMethods.NAIVE or self.method == SelectionMethods.FIRST:
        if self.method == SelectionMethods.FIRST:
            self.stepwise_rules = self.rules.copy(deep=True)
        else:
            # self.stepwise_rules = self.rules.copy(deep=True).sample(n=self.topk)
            self.stepwise_rules = self.filter_matching_antecedents().copy(deep=True)

        self.selection_size.append(len(self.stepwise_rules.index))

        if self.method == SelectionMethods.DGAP:
            # logger.info("Ranking rules")
            # row = self.dynamic_ranking_discrete()
            row = self.dynamic_ranking_discrete_new()
            if row is None:
                return None
            return row["consequent"].values[0]
        elif self.method == SelectionMethods.DGAP_ACC:
            row = self.dynamic_ranking_discrete_acc()
            if row is None:
                return None
            return row["consequent"].values[0]
        elif self.method == SelectionMethods.CGAP:
            row = self.dynamic_ranking_continuous()
            if row is None:
                return None, None
            ts_list = row["cgaps"].values[0]
            return row["consequent"].values[0], (sum(ts_list) / len(ts_list))
        elif self.method == SelectionMethods.CGAP_ACC:
            row = self.dynamic_ranking_continuous_acc()
            if row is None:
                return None, None
            ts_list = row["cgaps"].values[0]
            return row["consequent"].values[0], (sum(ts_list) / len(ts_list))
        elif self.method == SelectionMethods.RANDOM or self.method == SelectionMethods.NAIVE or self.method == SelectionMethods.FIRST:
            if self.stepwise_rules.empty:
                return None
            return self.select_consequent(self.stepwise_rules)
        elif self.method == SelectionMethods.UNIQUE_CONSEQUENT:
            return self.select_unique_consequent(self.stepwise_rules)
        elif self.method == SelectionMethods.UNIQUE_CONSEQUENT_WITHIN_TIME_WINDOW:
            return self.select_unique_consequent_within_time_window(self.stepwise_rules, delta=DELTA)

    __call__ = __next__

    def filter_matching_antecedents(self):
        ind = []
        for index, row in self.rules.iterrows():
            a = row["antecedent"]
            if utils.is_multi_subset(self.sequence, a):
                ind.append(index)
        return self.rules.iloc[ind]

    def select_consequent(self, rules):
        if self.method == SelectionMethods.RANDOM or self.method == SelectionMethods.NAIVE:
            # logger.info(f"Sampling on dataframe: {topk}")
            sample = rules.sample()
            # logger.info(f"Sampling following rule: {sample['rule']}")
            return sample["consequent"].values[0]
        elif self.method == SelectionMethods.FIRST:
            return rules.head(1)["consequent"].values[0]

    def select_unique_consequent(self, rules):
        seq_of_tuples = [(x,) for x in self.sequence]
        # print("Selecting unique consequent")
        # print(f"Sequence: {self.sequence}")
        # print(f"Rules: {rules}")
        options = rules[~rules["consequent"].isin(seq_of_tuples)]
        # print(f"Options: {options}")
        if len(options.index) == 0:
            return None
        else:
            sample = options.sample()
            return sample["consequent"].values[0]

    def select_unique_consequent_within_time_window(self, rules, delta: int):
        if len(self.sequence) < delta:
            seq_of_tuples = [(x,) for x in self.sequence]
        else:
            seq_of_tuples = [(x,) for x in self.sequence[-delta:]]

        # print("Selecting unique consequent")
        # print(f"Sequence: {self.sequence}")
        # print(f"Rules: {rules}")
        options = rules[~rules["consequent"].isin(seq_of_tuples)]
        # print(f"Options: {options}")
        if len(options.index) == 0:
            return None
        else:
            sample = options.sample()
            return sample["consequent"].values[0]

    def dynamic_ranking_discrete_new(self):
        seq_len = len(self.sequence)
        ranking = dict()
        for index, row in self.stepwise_rules.iterrows():
            # get each position in the sequence which matches with the antecedent of the current rule
            latest_index = max([i for i, x in enumerate(self.sequence) if x in row["antecedent"]])
            gap = seq_len - (latest_index + 1)
            # if this position exists in the histogram of the current rule save it as a possible value
            ranking_value = row["ms_dgaps"][gap] if gap in row["ms_dgaps"] else None
            # ranking_value = sum(row["ms_dgaps"][:gap])
            if ranking_value is not None:
                # save it for the current rule
                ranking[index] = ranking_value
        sorted_ranking = sorted(ranking, key=ranking.get, reverse=True)
        recommendation_list = self.stepwise_rules.loc[sorted_ranking].head(self.topk)
        self.stepwise_rules = recommendation_list  # mandatory in this variable, because this is evaluated for interlist-diversity

        if len(ranking) == 0:
            logger.info(f"Could not find matching rule")
            return None
        max_val = max(ranking.values())
        max_indices = [key for key, value in ranking.items() if value == max_val][:self.topk]
        # logger.info(f"Indices of rules with highest gap values: {max_indices}")
        # logger.info(f"Gap fitting rules: {ranking.values()}")
        if len(max_indices) == 0:
            # no rule could be found -- abort simulation of current sequence
            return None
        # Take the rules with the smallest index / first in dataframe
        min_index = min(max_indices)
        return recommendation_list.loc[[min_index]]

    def dynamic_ranking_discrete_acc(self):
        seq_len = len(self.sequence)
        ranking = dict()
        for index, row in self.stepwise_rules.iterrows():
            # get each position in the sequence which matches with the antecedent of the current rule
            latest_index = max([i for i, x in enumerate(self.sequence) if x in row["antecedent"]])
            gap = seq_len - (latest_index + 1)
            # if this position exists in the histogram of the current rule save it as a possible value
            # ranking_value = row["ms_dgaps"][gap] if gap in row["ms_dgaps"] else None
            ranking_value = sum([v for k, v in row["ms_dgaps"].items() if k <= gap])
            if ranking_value is not None:
                # save it for the current rule
                ranking[index] = ranking_value
        sorted_ranking = sorted(ranking, key=ranking.get, reverse=True)
        recommendation_list = self.stepwise_rules.loc[sorted_ranking].head(self.topk)
        self.stepwise_rules = recommendation_list  # mandatory in this variable, because this is evaluated for interlist-diversity

        if len(ranking) == 0:
            logger.info(f"Could not find matching rule")
            return None
        max_val = max(ranking.values())
        max_indices = [key for key, value in ranking.items() if value == max_val][:self.topk]
        # logger.info(f"Indices of rules with highest gap values: {max_indices}")
        # logger.info(f"Gap fitting rules: {ranking.values()}")
        if len(max_indices) == 0:
            # no rule could be found -- abort simulation of current sequence
            return None
        # Take the rules with the smallest index / first in dataframe
        min_index = min(max_indices)
        return recommendation_list.loc[[min_index]]

    def dynamic_ranking_discrete(self):
        seq_len = len(self.sequence)
        ranking = dict()
        for index, row in self.stepwise_rules.iterrows():
            # logger.info(row)
            if seq_len > 0:
                # get each position in the sequence which matches with the antecedent of the current rule
                indices = [i for i, x in enumerate(self.sequence) if x in row["antecedent"]]
            else:
                indices = [seq_len]
            # if this position exists in the histogram of the current rule save it as a possible value
            ranking_values = []
            for i in indices:
                gap = seq_len - (i + 1)
                if gap in row["ms_dgaps"]:
                    ranking_values.append(row["ms_dgaps"][gap])
                elif gap - 1 in row["ms_dgaps"]:
                    ranking_values.append(row["ms_dgaps"][gap - 1])
                elif gap + 1 in row["ms_dgaps"]:
                    ranking_values.append(row["ms_dgaps"][gap + 1])
                # else:
                # print("Could not find gap match")
            # ranking_values = [row["ms_dgaps"][seq_len - (i+1)] for i in indices if seq_len - (i+1) in row["ms_dgaps"]]
            if len(ranking_values) > 0:
                # save it for the current rule
                ranking[index] = max(ranking_values)

            # for index, row in self.stepwise_rules.iterrows():
            #    ms = self.create_multiset(row["dgaps"])
            # logging.info(ms)
            # source = pd.DataFrame({
            #    '|G|': ms.keys(),
            #    'Number of occurences': ms.values()
            # })
            # c = alt.Chart(source).mark_bar().encode(
            #    x='|G|',
            #    y='Number of occurences'
            # )
            # st.write(c)

        # logger.info("Sort ranking")
        sorted_ranking = sorted(ranking, key=ranking.get, reverse=True)
        # logger.info(f"Ranking: {sorted_ranking}")
        # df1 = self.stepwise_rules.set_index('Tm')
        # st.write("Ranking rules")
        pd.set_option('display.max_columns', None)
        # print(self.stepwise_rules)
        # logger.info(self.stepwise_rules.loc[sorted_ranking])
        # logger.info(self.topk)
        recommendation_list = self.stepwise_rules.loc[sorted_ranking].head(self.topk)
        # Fill up recommendation list
        if len(self.stepwise_rules.index) < self.topk:
            indices_of_recommended_rules = set(recommendation_list.index.values)
            indices_of_all_rules = set(self.stepwise_rules.index.values)
            # print(indices_of_recommended_rules)
            # print(indices_of_all_rules)
            possible_filler = list(indices_of_all_rules.difference(indices_of_recommended_rules))
            if len(possible_filler) > 0:
                nr_filler = (self.topk - len(self.stepwise_rules.index)) if (self.topk - len(
                    self.stepwise_rules.index)) < len(possible_filler) else len(possible_filler)
                filler_indices = random.sample(possible_filler, nr_filler)
                filler = self.stepwise_rules.loc[filler_indices]
                # filler = self.stepwise_rules.nlargest((), ['support'])
                # print("Filler:")
                # print(filler)
                recommendation_list = pd.concat([recommendation_list.loc[:], filler])
        self.stepwise_rules = recommendation_list  # mandatory in this variable, because this is evaluated for interlist-diversity
        # print(self.stepwise_rules)
        # if there is no matching rule -> do not recommend anything
        # if there is at least one matching rule fill recommendation list -> it does not fake the next step recommndation
        # NDCG suffers extremely by this
        # if 0 < len(self.stepwise_rules.index) < self.topk:
        #     indices_of_recommended_rules = set(self.stepwise_rules.index.values)
        #     indices_of_all_rules = set(self.rules.index.values)
        #     possible_filler = list(indices_of_all_rules.difference(indices_of_recommended_rules))
        #     if len(possible_filler) > 0:
        #         nr_filler = self.topk - len(self.stepwise_rules.index)
        #         filler_indices = random.sample(possible_filler, nr_filler)
        #         filler = self.rules.loc[filler_indices]
        #         # filler = self.stepwise_rules.nlargest((), ['support'])
        #         # print("Filler:")
        #         # print(filler)
        #         self.stepwise_rules = pd.concat([recommendation_list.loc[:], filler])
        # st.write("Ranking rules")
        # self.stepwise_rules = recommendation_list
        # print(len(self.stepwise_rules.index))
        # logger.info(recommendation_list)
        # logger.info(type(ranking))
        # logger.info(ranking)
        # Get all rules with the highest value of their respective gap
        # print(ranking)
        if len(ranking) == 0:
            logger.info(f"Could not find matching rule")
            return None
        max_val = max(ranking.values())
        max_indices = [key for key, value in ranking.items() if value == max_val][:self.topk]
        logger.info(f"Indices of rules with highest gap values: {max_indices}")
        logger.info(f"Gap fitting rules: {ranking.values()}")
        if len(max_indices) == 0:
            # no rule could be found -- abort simulation of current sequence
            return None
        # Take the rules with the smallest index / first in dataframe
        min_index = min(max_indices)
        # st.write(f"Min index: {min_index}")
        # print(rules)
        return recommendation_list.loc[[min_index]]

    def dynamic_ranking_continuous(self):
        ranking = dict()
        # print("Rules")
        # logger.debug(self.stepwise_rules)
        # print(f"Analysing sequence: {self.sequence} with ts: {self.sequence_ts}")
        # for i, el in enumerate(self.sequence):
        for index, row in self.stepwise_rules.iterrows():
            cgap = utils.retrieve_cgap(self.sequence, self.sequence_ts, row['antecedent'])
            logger.info(f"########################### CGAP: {cgap.gap}")
            ts_gaps = row["cgaps"]
            if ts_gaps is None:
                continue
            # logger.info(f"TSGaps: {ts_gaps}")
            ts_arr = np.array(ts_gaps)
            if self.bandwidth:
                h = self.bandwidth
            else:
                # if bandwidth is not passed as a parameter we use Silverman's rule of thumb
                h = np.std(ts_arr) * (4 / 3 / len(ts_arr)) ** (1 / 5)
                if h == 0.0:
                    h = 1.0
            # st.write(f"Bandwidth={h}")
            # sns.kdeplot(ts_arr, bw=h)
            # plt.show()
            kde = KernelDensity(kernel="gaussian", bandwidth=h).fit(ts_arr.reshape(-1, 1))
            prob = np.exp(kde.score_samples([[cgap.get_gap()]]))
            # st.write(cgap.get_gap())
            # X_plot = np.linspace(0, 2000000, 1000)[:, np.newaxis]
            # log_dens = kde.score_samples(X_plot)
            # fig, ax = plt.subplots()
            # ax.plot(
            #    X_plot[:, 0],
            #    np.exp(log_dens),
            #    color="blue",
            #    lw=2,
            #    linestyle="-",
            #    label="kernel = '{0}'".format("gaussian"),
            # )
            # # st.pyplot(fig)
            # plt.show()
            # #logging.info(f"Got probability: {prob[0]}")
            ranking[index] = prob[0]

        # logger.info(f"Ranking: {ranking}")
        sorted_ranking = sorted(ranking, key=ranking.get, reverse=True)
        # st.write("Ranking rules")
        # st.write(self.stepwise_rules)
        recommendation_list = self.stepwise_rules.loc[sorted_ranking].head(self.topk)
        self.stepwise_rules = recommendation_list  # mandatory in this variable, because this is evaluated for interlist-diversity
        if len(ranking) == 0:
            logger.info(f"Could not find matching rule")
            return None
        # Get all rules with the highest value of their respective gap
        max_val = max(ranking.values())
        max_indices = [key for key, value in ranking.items() if value == max_val][:self.topk]
        # logger.info(f"Max indices: {max_indices}")
        if len(max_indices) == 0:
            # no rule could be found -- abort simulation
            return None
        # Take the rules with the smallest index / first in dataframe
        min_index = min(max_indices)
        return recommendation_list.loc[[min_index]]

    def dynamic_ranking_continuous_acc(self):
        logger.info("Using CGAP ACC")
        ranking = dict()
        # print("Rules")
        # logger.debug(self.stepwise_rules)
        # print(f"Analysing sequence: {self.sequence} with ts: {self.sequence_ts}")
        # for i, el in enumerate(self.sequence):
        for index, row in self.stepwise_rules.iterrows():
            cgap = utils.retrieve_cgap(self.sequence, self.sequence_ts, row['antecedent'])
            # logger.info(f"########################### CGAP: {cgap.gap}")
            ts_gaps = row["cgaps"]
            if ts_gaps is None:
                continue
            ts_arr = np.array(ts_gaps)
            if self.bandwidth:
                h = self.bandwidth
            else:
                # if bandwidth is not passed as a parameter we use Silverman's rule of thumb
                h = np.std(ts_arr) * (4 / 3 / len(ts_arr)) ** (1 / 5)
                if h == 0.0:
                    h = 1.0
            # st.write(f"Bandwidth={h}")
            # sns.kdeplot(ts_arr, bw=h)
            # plt.show()
            kde = KernelDensity(kernel="gaussian", bandwidth=h).fit(ts_arr.reshape(-1, 1))
            # logger.info(cgap.get_gap())
            if cgap.get_gap() > 0:
                gaps = [[x] for x in np.arange(start=0, stop=cgap.get_gap(), step=cgap.get_gap() / 100)]
                # logger.info(gaps)
                # logger.info(np.exp(kde.score_samples(gaps)))
                prob = [integrate.simpson(np.exp(kde.score_samples(gaps)))]
                # prob = [sum(np.exp(kde.score_samples(gaps)))]
                # logger.info(prob)
            else:
                prob = np.exp(kde.score_samples([[cgap.get_gap()]]))
                # logger.info(prob)

            # prob = np.exp(kde.score_samples([[cgap.get_gap()]]))
            # st.write(cgap.get_gap())
            # X_plot = np.linspace(0, 2000000, 1000)[:, np.newaxis]
            # log_dens = kde.score_samples(X_plot)
            # fig, ax = plt.subplots()
            # ax.plot(
            #    X_plot[:, 0],
            #    np.exp(log_dens),
            #    color="blue",
            #    lw=2,
            #    linestyle="-",
            #    label="kernel = '{0}'".format("gaussian"),
            # )
            # # st.pyplot(fig)
            # plt.show()
            # #logging.info(f"Got probability: {prob[0]}")
            ranking[index] = prob[0]

        # logger.info(f"Ranking: {ranking}")
        sorted_ranking = sorted(ranking, key=ranking.get, reverse=True)
        # st.write("Ranking rules")
        # st.write(self.stepwise_rules)
        recommendation_list = self.stepwise_rules.loc[sorted_ranking].head(self.topk)
        self.stepwise_rules = recommendation_list  # mandatory in this variable, because this is evaluated for interlist-diversity
        if len(ranking) == 0:
            logger.info(f"Could not find matching rule")
            return None
        # Get all rules with the highest value of their respective gap
        max_val = max(ranking.values())
        max_indices = [key for key, value in ranking.items() if value == max_val][:self.topk]
        # logger.info(f"Max indices: {max_indices}")
        if len(max_indices) == 0:
            # no rule could be found -- abort simulation
            return None
        # Take the rules with the smallest index / first in dataframe
        min_index = min(max_indices)
        return recommendation_list.loc[[min_index]]

    def get_first_item(self):
        return self.rules.head(1)["consequent"].values[0]

    def get_first_timestamp(self):
        return [0]

