import os
import pickle
import time

import pandas as pd

import utils
from dp_recs import load_data, load_data_seq_only
from src.enums.filter import Filter
from src.project_simulator.extractor import Extractor
from src.project_simulator.rule_filter import RuleFilter


def remove_generator_patterns(s):
    res_antecedent = set()
    res_index = set()
    # print(s)
    # print(sorted(s, key=lambda x: len(x[1])))
    for el in sorted(s, key=lambda x: len(x[1])):
        # print(el)
        # getting smallest set and checking for already
        # present smaller set for subset
        if not any(el[1][idx: idx + y + 1] in res_antecedent
                   for idx in range(len(el[1]))
                   for y in range(len(el[1]) - idx)):
            res_antecedent.add(el[1])
            res_index.add((el[0]))
    # print(res_index)
    # print(res_antecedent)
    return res_index

def filter_antecedent_generators(rules):
    grouped_rules = rules.groupby('consequent')
    # print(f"Number of groups: {len(grouped_rules)}")
    df_list = []
    for name, group in grouped_rules:
        ant_set = list(group.itertuples(index=True, name=None))
        ant_set = set(map(lambda x: (x[0], x[2]), ant_set))
        indices_to_keep = remove_generator_patterns(ant_set)
        group = group.loc[list(indices_to_keep)]
        df_list.append(group)
    # print(pd.concat(df_list, axis=0))
    return pd.concat(df_list, axis=0)


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)

    path = os.path.join("")
    train = os.path.join(path, "seq_train.txt")
    test = os.path.join(path, "seq_test.txt")
    rules = os.path.join(path, "erminer_rules.txt")

    input = load_data_seq_only(train, test, rules)
    rules = input['rules']
    print(len(rules.index))


    # rules = filter_antecedent_generators(rules)
    # rules.reset_index(drop=True, inplace=True)
    # print(rules)
    # print(f"Number of rules {len(rules.index)}")
    # with open('52000_rules_filtered.pkl', 'wb') as handle:
    #     pickle.dump(rules, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open("52000_rules_filtered.pkl", "rb") as f:
        rules = pickle.load(f)

    print(len(rules))
    extractor = Extractor(input, rules.copy(deep=True), "DGap")
    s_time = time.time()
    extractor.run()
    print(f"Took {time.time() - s_time}")

    rules = extractor.rules.copy(deep=True)

    with open('520000_filtered_rules_with_dgaps.pkl', 'wb') as handle:
        pickle.dump(rules, handle, protocol=pickle.HIGHEST_PROTOCOL)
