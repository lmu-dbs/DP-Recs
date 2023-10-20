import logging
import os
import random
from abc import abstractmethod

import numpy as np

from src.data_analyzer import DataAnalyzer
from src.quality_meter.sequence_match import SequenceMatch

logger = logging.getLogger(__name__)


class Gap:

    def __init__(self, ant_indices, cons_indices):
        self.ant_indices = ant_indices
        self.cons_indices = cons_indices
        self.gap = None

    def get_gap(self):
        return self.gap

    @abstractmethod
    def compute_gap(self):
        pass


class DGap(Gap):
    def __init__(self, ant_indices, cons_indices):
        super().__init__(ant_indices, cons_indices)
        self.compute_gap()

    def compute_gap(self):
        end_index = min(self.cons_indices)
        start_index = max(self.ant_indices)
        gap = end_index - start_index - 1
        # print(f"Computed the following gap: {gap}")
        if gap >= 0:
            self.gap = gap


class CGap(Gap):
    def __init__(self, ant_indices, cons_indices, ts_sequence):
        super().__init__(ant_indices, cons_indices)
        logger.info(ant_indices)
        logger.info(cons_indices)
        self.ts_sequence = ts_sequence

    def compute_gap(self):
        end_index = min(self.cons_indices)
        start_index = max(self.ant_indices)
        # logger.info(f"timestamp sequence: {self.ts_sequence} at start {start_index} and end {end_index}")
        self.gap = self.ts_sequence[end_index] - self.ts_sequence[start_index]


def retrieve_cgap(sequence, ts_sequence, antecedent):
    ant_indices = []
    contains_all_items = True
    if len(sequence) > 0:
        for item in antecedent:
            try:
                ant_indices.append(sequence.index(item))
            except ValueError as ve:
                contains_all_items = False
        if contains_all_items:
            cgap = CGap(ant_indices, [(len(sequence) - 1)], ts_sequence)
            cgap.compute_gap()
            return cgap
        else:
            return None
    else:
        cgap = CGap(None, None, None)
        cgap.gap = 0
        return cgap


def retrieve_dgap(sequence, antecedent, consequent):
    # print(f"Checking sequence : {sequence}")
    # ToDo: Special case where the same item appears more than once in the sequence is not covered
    ant_indices = []
    cons_indices = []
    contains_all_items = True
    for item in antecedent:
        try:
            ant_indices.append(sequence.index(item))
        except ValueError as ve:
            contains_all_items = False
    for item in consequent:
        try:
            cons_indices.append(sequence.index(item))
        except ValueError as ve:
            contains_all_items = False
    # print(f"Antecedent: {ant_indices}")
    # print(f"Consequent: {cons_indices}")
    if contains_all_items:
        # Subtract 1 to get the actual gap (number of elements in between) and not the distance
        # print(f"Found matching rule for sequence {sequence}")
        return DGap(ant_indices, cons_indices)
    else:
        # print("Did not match")
        return None


# Compute the relative diversity of a list
# Returns ratio of unique elements to number of all elements in a list
def get_relative_diversity(l):
    return (len(set(l))) / (len(l))


def compute_length(l, std):
    length = int(random.uniform(l - std, l + std))
    if length == 0:
        return 1
    else:
        return length


# from https://www.geeksforgeeks.org/damerau-levenshtein-distance/
def optimal_string_alignment_distance(s1, s2):
    # Create a table to store the results of subproblems
    dp = [[0 for j in range(len(s2) + 1)] for i in range(len(s1) + 1)]

    # Initialize the table
    for i in range(len(s1) + 1):
        dp[i][0] = i
    for j in range(len(s2) + 1):
        dp[0][j] = j

    # Populate the table using dynamic programming
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    # Return the edit distance
    return dp[len(s1)][len(s2)]


def write_output(title, sequences):
    # write simulated projects to file
    with open(title, 'w', encoding='UTF8') as f:
        for proj in sequences:
            temp = ""
            for el in proj:
                temp += str(el) + " -1 "
            temp += "-2"
            if not temp == "-2":
                f.write(temp)
                f.write("\n")


def read_input_data_from_path(path):
    data = []
    with open(path, "r") as f:
        lines = f.readlines()
        for sequence in lines:
            temp = []
            # sequence = sequence.decode("utf-8")
            for x in sequence.strip().split(" "):
                if x != "-1" and x != "-2":
                    temp.append(int(x))
            # logger.info(f"Reading sequence: {temp}")
            data.append(temp)
    return data


def read_input_data(raw_data):
    # st.write(raw_data)
    if raw_data is not None:
        lines = raw_data.readlines()
        data = []
        for sequence in lines:
            temp = []
            sequence = sequence.decode("utf-8")
            for x in sequence.strip().split(" "):
                if x != "-1" and x != "-2":
                    temp.append(int(x))
            # logger.info(f"Reading sequence: {temp}")
            data.append(temp)
        return data
    else:
        return None


def filter_test_sequences(test_data, sequence):
    matches = []
    for s in test_data:
        contains_all = True
        indices = []
        for item in sequence:
            try:
                found_index = s.index(item)
                if found_index not in indices:
                    indices.append(found_index)
                else:
                    raise ValueError
            except ValueError as ve:
                contains_all = False
        if contains_all:
            matches.append(SequenceMatch(s, indices))
    return matches


def split_training_test(data: list, split: int, seed: int):
    random.seed(seed)
    random.shuffle(data)
    train_data = data[:split]
    test_data = data[split:]
    return train_data, test_data


def split_by_fifths(data: list):
    return [x.tolist() for x in np.array_split(data, 5)]


def create_multiset(l):
    d = dict()
    if len(l) > 0:
        for el in set(l):
            d[el] = l.count(el)
    return d


# sample_size represents the amount of sequences in the original database (in percent)
def get_sample(data: list, sample_size: float):
    sample = []
    for sequence in data:
        if random.random() < sample_size:
            sample.append(sequence)
    return sample


def write_metdata(path, title, analyzer: DataAnalyzer):
    # write simulated projects to file
    with open(os.path.join(path, title), 'w', encoding='UTF8') as f:
        f.write(f"Number of sequences: {analyzer.nr_of_sequences}\n")
        f.write(f"Number of distinct items: {analyzer.nr_dist_items}\n")
        f.write(f"Mean sequence length: {analyzer.mean_seq_len}\n")
        f.write(f"Number of distinct items per sequence: {analyzer.nr_dist_items_per_seq}\n")
        f.close()


def is_sorted(l):
    if l == sorted(l):
        return True
    else:
        return False


def is_multi_subset(l, to_check):
    d = create_multiset(l)
    c = create_multiset(to_check)
    matching = True
    for k, v in c.items():
        if k not in d:
            matching = False
            break
        else:
            if v > d[k]:
                matching = False
                break
    return matching


def write_dataset(title, seq_train, seq_test, ts_train, ts_test):
    with open("seq_train_" + title, 'w', encoding='UTF8') as f:
        for s in seq_train:
            temp = ""
            for el in s:
                temp += str(el) + " -1 "
            temp += "-2"
            f.write(temp)
            f.write("\n")

    with open("seq_test_" + title, 'w', encoding='UTF8') as f:
        for s in seq_test:
            temp = ""
            for el in s:
                temp += str(el) + " -1 "
            temp += "-2"
            f.write(temp)
            f.write("\n")

    with open("ts_train_" + title, 'w', encoding='UTF8') as f:
        for t in ts_train:
            temp = ""
            for el in t:
                temp += str(el) + " -1 "
            temp += "-2"
            f.write(temp)
            f.write("\n")

    with open("ts_test_" + title, 'w', encoding='UTF8') as f:
        for t in ts_test:
            temp = ""
            for el in t:
                temp += str(el) + " -1 "
            temp += "-2"
            f.write(temp)
            f.write("\n")
