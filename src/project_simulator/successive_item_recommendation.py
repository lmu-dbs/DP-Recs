import copy
import logging

import constants
import utils
from src.enums.selection_methods import SelectionMethods
from src.project_simulator.qualities import Qualities
from src.project_simulator.rule_selector import RuleSelector

logger = logging.getLogger(__name__)


class FullSequenceSimulator:
    """
    * Creates n sequences by calling rule selector
    * Recommendation base: Uses consequent of first rule if there is no recommendation base
    * Stopping criterion: Mean length of sequence (from train set) +/- one standard deviation
    * Length of recommendation base is variable
    """

    def __init__(self, test_set, ts_test_set, rules, filt, selection_method, bandwidth, db_size, rel_div_aft, topk):
        self.kernel = 'gaussian'
        self.rules = rules
        self.sequences = []
        self.bandwidth = bandwidth
        self.test_set = test_set
        self.ts_test_set = ts_test_set
        self.db_size = db_size
        self.selection_method = selection_method
        self.selection_sizes = []
        self.filter = filt
        self.rel_div_aft = rel_div_aft
        self.rel_recommendation_base = None
        self.topk = topk
        self.qualities = Qualities(self.test_set, self.topk)
        # if len(test_sequence) > CUTOFFSIZE then it is testable
        self.testable_sequences = 0

    def run(self):
        if self.selection_method == SelectionMethods.DGAP.value or self.selection_method == SelectionMethods.DGAP_ACC.value:
            self.generate_sequences_with_dgap()
        elif self.selection_method == SelectionMethods.CGAP.value or self.selection_method == SelectionMethods.CGAP_ACC.value:
            self.generate_sequences_with_cgap()
        elif self.selection_method == SelectionMethods.UNIQUE_CONSEQUENT.value or self.selection_method == SelectionMethods.UNIQUE_CONSEQUENT_WITHIN_TIME_WINDOW.value:
            self.generate_sequences_unique_random()
        elif self.selection_method == SelectionMethods.NAIVE:
            self.generate_sequences_naive()

    def prepare_sequence_simulation(self, index):
        rule_selector = RuleSelector(self.topk, self.bandwidth, rules=self.rules, method=self.selection_method)
        gt_sequence = self.test_set[index]
        new_sequence = copy.deepcopy(gt_sequence)
        new_sequence = self.cut_at_n(new_sequence)
        return rule_selector, gt_sequence, new_sequence

    def save_for_single_item_qualities(self, new_sequence, gt_sequence, rule_selector):
        self.qualities.successful_recommendations += 1
        self.qualities.hr.add_sequence_succ(new_sequence, gt_sequence,
                                            rule_selector.stepwise_rules["consequent"])
        self.qualities.ndcg.add_sequence(new_sequence, rule_selector.stepwise_rules["consequent"])
        self.qualities.div.add_sequence(new_sequence, rule_selector.stepwise_rules["consequent"])
        self.qualities.ildrr.add_sequence_succ(new_sequence, gt_sequence, rule_selector.stepwise_rules["consequent"])
        self.qualities.accuracy.add_sequence_succ(new_sequence, gt_sequence)
        self.selection_sizes.extend(rule_selector.selection_size)

    def complete_sequence_simulation(self, new_sequence, gt_sequence):
        logger.info(f"Simulated sequence: {new_sequence}")
        self.qualities.damerau_levenshtein.add_sequence_succ(new_sequence[constants.CUTOFFSIZE:],
                                                             gt_sequence[constants.CUTOFFSIZE:])
        self.qualities.rel_div.add_sequence_succ(new_sequence[constants.CUTOFFSIZE:],
                                                 gt_sequence[constants.CUTOFFSIZE:])
        self.sequences.append(new_sequence)  # has to be pruned because last recommendation can be a batch of items

    def generate_sequences_unique_random(self):
        logger.info("Generating sequences with UNIQUE RANDOM")
        for index in range(len(self.test_set)):
            if index > self.db_size:
                break
            rule_selector, gt_sequence, new_sequence = self.prepare_sequence_simulation(index)
            if new_sequence is None:
                continue
            logger.info(f"Ground truth sequence: {gt_sequence}")
            gt_sequence_length = len(gt_sequence)
            rule_selector.sequence = new_sequence
            rule_iterator = iter(rule_selector)
            while len(new_sequence) < gt_sequence_length:
                next_item = next(rule_iterator)
                if next_item is None:
                    self.qualities.cancelled_recommendations += 1
                    break
                new_sequence.extend(next_item)
                self.save_for_single_item_qualities(new_sequence, gt_sequence, rule_selector)
            self.complete_sequence_simulation(new_sequence, gt_sequence)
        self.qualities.compute_qualities()

    def generate_sequences_naive(self):
        logger.info("Generating sequences with Naive")
        for index in range(len(self.test_set)):
            if index > self.db_size:
                break
            # if index % 1000 == 0:
            #     print(f"{index}/{len(self.test_set)}")
            rule_selector, gt_sequence, new_sequence = self.prepare_sequence_simulation(index)
            gt_sequence_length = len(gt_sequence)
            if new_sequence is None:  # go on to next test sequence if it is not longer than n
                continue
            logger.info(f"Ground truth sequence: {gt_sequence}")
            rule_selector.sequence = new_sequence
            rule_iterator = iter(rule_selector)
            while len(new_sequence) < gt_sequence_length:
                next_item = next(rule_iterator)
                if next_item is None:
                    self.qualities.cancelled_recommendations += 1
                    break
                new_sequence.extend(next_item)
                self.save_for_single_item_qualities(new_sequence, gt_sequence, rule_selector)
            self.complete_sequence_simulation(new_sequence, gt_sequence)
        self.qualities.compute_qualities()


    def generate_sequences_with_dgap(self):
        logger.info("Generating sequences with DGap")
        # ToDo: defer to pre-processing
        ms = len(self.rules.index) * [None]
        for index, row in self.rules.iterrows():
            if row["dgaps"]:
                ms[index - 1] = utils.create_multiset(row["dgaps"])
            else:
                ms[index - 1] = dict()
        self.rules["ms_dgaps"] = ms

        for index in range(len(self.test_set)):
            if index > self.db_size:
                break
            # if index % 1000 == 0:
            #     print(f"{index}/{len(self.test_set)}")
            rule_selector, gt_sequence, new_sequence = self.prepare_sequence_simulation(index)
            gt_sequence_length = len(gt_sequence)
            if new_sequence is None:  # go on to next test sequence if it is not longer than n
                continue
            logger.info(f"Ground truth sequence: {gt_sequence}")
            rule_selector.sequence = new_sequence
            rule_iterator = iter(rule_selector)
            while len(new_sequence) < gt_sequence_length:
                next_item = next(rule_iterator)
                if next_item is None:
                    self.qualities.cancelled_recommendations += 1
                    break
                new_sequence.extend(next_item)
                self.save_for_single_item_qualities(new_sequence, gt_sequence, rule_selector)
            self.complete_sequence_simulation(new_sequence, gt_sequence)
        self.qualities.compute_qualities()

    def generate_sequences_with_cgap(self):
        for index in range(len(self.test_set)):
            if index > self.db_size:
                break
            if index % 1000 == 0:
                print(f"{index}/{len(self.test_set)}")
            rule_selector = RuleSelector(self.topk, self.bandwidth, rules=self.rules, method=self.selection_method)
            gt_sequence = self.test_set[index]
            logger.info(f"Ground truth sequence: {gt_sequence}")
            gt_sequence_length = len(gt_sequence)
            new_sequence = copy.deepcopy(gt_sequence)
            new_sequence = self.cut_at_n(new_sequence)
            if new_sequence is None:
                continue

            # logger.info(f"Using the following recommendation base: {new_sequence}")
            rule_selector.sequence = new_sequence
            if self.selection_method == SelectionMethods.CGAP or self.selection_method == SelectionMethods.CGAP_ACC:
                if gt_sequence_length > 1:
                    new_sequence_ts = self.cut_at_n(self.ts_test_set[index], constants.CUTOFFSIZE)  # recommendation base timestamp
                    # logger.debug(f"Corresponding recommendation base ts: {new_sequence_ts}")
                else:
                    new_sequence_ts = []
                rule_selector.sequence_ts = new_sequence_ts
                if self.bandwidth:
                    rule_selector.bandwidth = self.bandwidth
            # st.write("Creating rule iterator")
            rule_iterator = iter(rule_selector)

            while len(new_sequence) < gt_sequence_length:
                # print("Getting next rule")
                if self.selection_method == SelectionMethods.CGAP or self.selection_method == SelectionMethods.CGAP_ACC:
                    next_item, next_ts = next(rule_iterator)
                    if next_item is None:
                        self.qualities.cancelled_recommendations += 1
                        break
                    # logger.info(f"Next item: {next_item}, next ts: {next_ts}")
                    new_sequence.extend(next_item)
                    # Add mean timestamp of predicted element to last timestamp
                    if len(new_sequence) == 1:
                        latest = 0.0
                    else:
                        latest = new_sequence_ts[-1]
                    for _ in range(len(next_item)):  # next recommendation can also be a batch of items
                        new_sequence_ts.append(latest + next_ts)

                self.save_for_single_item_qualities(new_sequence, gt_sequence, rule_selector)
            self.complete_sequence_simulation(new_sequence, gt_sequence)
        self.qualities.compute_qualities()


    def replace_event(self, sequence, position):
        recommendation_base = sequence[:position]
        remainder = sequence[position:]
        rule_selector = iter(RuleSelector(self.topk, self.rules, recommendation_base), self.l)
        recommendation_base.extend(next(rule_selector))
        new_sequence = recommendation_base + remainder[1:]
        return new_sequence

    def cut_at_recommendation_basis(self, index):
        # logger.info(f"Index: {index}, {self.rel_recommendation_base}")
        if self.rel_recommendation_base > 0.0 and index is not None:
            s = self.test_set[index]
            l = int(len(s) * self.rel_recommendation_base)
            if l == 0:  # Use at least one item as recommendation basis
                return s[:1]
            return s[:l]
        else:
            s = self.test_set[index]
            return s[:1]

    def cut_at_n(self, sequence, n=constants.CUTOFFSIZE):
        seq_length = len(sequence)
        if seq_length <= n:
            # raise ValueError("Cut would result in less than one item in test sequence.")
            return None
        else:
            self.testable_sequences += 1
            return sequence[:n]

    # def recommendation_basis_ts(self, index):
    #     if self.rel_recommendation_base > 0.0 and index is not None:
    #         s = self.prelim["test_ts"][index]
    #         l = int(len(s) * self.rel_recommendation_base)
    #         if l == 0:
    #             return s[:1]
    #         return s[:l]
    #     else:
    #         return [0]


