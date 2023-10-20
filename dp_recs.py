import csv
import logging
import os
import pickle
import time

import pandas as pd
from spmf import Spmf

import utils
from src.enums.algorithm import Algorithm
from src.enums.filter import Filter
from src.enums.selection_methods import SelectionMethods
from src.project_simulator.rule_filter import RuleFilter
from src.project_simulator.rule_reader import RuleReader
from src.project_simulator.successive_item_recommendation import FullSequenceSimulator

logger = logging.getLogger(__name__)

def load_data_seq_only(train_seq, test_seq, rule_path):
    input_data = dict()
    input_data["train"] = utils.read_input_data_from_path(train_seq)
    input_data["test"] = utils.read_input_data_from_path(test_seq)
    rule_reader = RuleReader(rule_path, from_path=True)
    input_data["rules"] = rule_reader.data
    input_data["rel_div_bef"] = utils.get_relative_diversity(input_data["rules"]['consequent'].tolist())
    input_data["filter"] = dict()
    input_data["simulators"] = []
    input_data["std_seq"] = None
    input_data["mean_seq_len"] = None
    return input_data


def load_data(train_seq, test_seq, train_seq_ts, test_seq_ts, rule_path):
    input_data = dict()
    train_set = utils.read_input_data_from_path(train_seq)
    test_set = utils.read_input_data_from_path(test_seq)
    input_data["train"] = train_set
    input_data["test"] = test_set
    input_data["train_ts"] = utils.read_input_data_from_path(train_seq_ts)
    input_data["test_ts"] = utils.read_input_data_from_path(test_seq_ts)
    rule_reader = RuleReader(rule_path, from_path=True)
    input_data["rules"] = rule_reader.data
    input_data["rel_div_bef"] = utils.get_relative_diversity(input_data["rules"]['consequent'].tolist())
    input_data["filter"] = dict()
    input_data["simulators"] = []
    input_data["std_seq"] = None
    input_data["mean_seq_len"] = None
    return input_data


def er_miner():
    input_data = "./datasets/fifa/seq_train.txt"
    out_path = "./datasets/fifa/erminer_rules.txt"

    support = 0.0005
    confidence = 0.5
    max_antecedent = 10
    max_consequent = 1
    spmf = Spmf(Algorithm.ERMINER,
                input_filename=input_data,
                output_filename=out_path,
                spmf_bin_location_dir="C:/path/to/spmf/folder/",
                arguments=[support, confidence, max_antecedent, max_consequent])
    spmf.run()

def run_exp(rule_path, dataset="bmswebview1", sample_size=None, topk=10):
    path = os.path.join("datasets", dataset)
    test = os.path.join(path, "seq_test.txt")
    test_set = utils.read_input_data_from_path(test)
    if sample_size is None:
        sample_size = len(test_set)

    logger.info(f"\n\n\nStarting experiments for {dataset} with size {sample_size}")

    filter = Filter.NO_FILTERING
    with open(rule_path, "rb") as f:
        rules = pickle.load(f)

    selection_methods = [SelectionMethods.NAIVE,
                         SelectionMethods.DGAP,
                         SelectionMethods.DGAP_ACC,
                         SelectionMethods.UNIQUE_CONSEQUENT,
                         SelectionMethods.UNIQUE_CONSEQUENT_WITHIN_TIME_WINDOW]
    ts_test_set = None

    nr_of_generated_seq = sample_size
    rule_filter = RuleFilter(rules, filter)
    rules = rule_filter.run()
    rules.reset_index(drop=True, inplace=True)

    start_time = time.time()
    simulator = None
    simulators = []
    for selection_method in selection_methods:
        logger.info(f"Simulating with {selection_method}")
        simulator = FullSequenceSimulator(test_set, ts_test_set, rules, filter, selection_method,
                                          bandwidth=None,
                                          db_size=nr_of_generated_seq,
                                          rel_div_aft=rule_filter.rel_div_aft,
                                          topk=topk)
        simulator.run()
        simulators.append(simulator)
        # print(simulator.sequences)
        print(f"NDCG: {simulator.qualities.ndcg.mean_quality}")
        print(f"HR/Recall: {simulator.qualities.hr.mean_quality}")
        print(f"Accuracy/Precision: {simulator.qualities.accuracy.mean_quality}")
        print(f"Damerau Levenshtein: {simulator.qualities.damerau_levenshtein.mean_quality}")
        print(f"REL DIV Y: {simulator.qualities.rel_div.mean_quality_y}")
        print(f"REL DIV GT: {simulator.qualities.rel_div.mean_quality_gt}")
        print(f"ILD: {simulator.qualities.div.mean_quality}")
        print(f"DIV F1: {simulator.qualities.div_f1}")
        print(f"ILD-RR: {simulator.qualities.ildrr.mean_quality}")
        print(f"Inter-list Div (pairwise): {simulator.qualities.div.mean_inter_div_pairwise}")
        print(f"Inter-list Div (set theoretic): {simulator.qualities.div.mean_inter_div_set_theoretic}")

        print(f"Testable sequences: {simulator.testable_sequences}")
        print(f"Cancelled recommendations: {simulator.qualities.cancelled_recommendations}")
        print(f"Successful recommendations: {simulator.qualities.successful_recommendations}")
        # print(f"Inter-list Div Change: {simulator.qualities.div.mean_idiv_change}")

    write_to_csv("results_" + dataset + ".csv", selection_methods, simulators=simulators)
    seconds = time.time()-start_time
    minutes = seconds/60
    hours = minutes/60
    logger.info(f"Took {seconds}s = {minutes}min = {hours}h")

    return simulator

def df_to_ibmgenerator_format(path, train=True, session_key='SessionId', item_key='ItemId', time_key='Time'):
    pd.set_option('display.max_columns', None)
    df = pd.read_csv(path, sep="\t")
    df.sort_values([session_key, time_key], inplace=True)
    if not train:
        df = df.groupby([session_key]).filter(lambda x: len(x) >= 3)

    df.sort_values([session_key, time_key], inplace=True)
    grouped = df.groupby([session_key])

    act_sequences = []
    ts_sequences = []
    for name, group in grouped:
        act_seq = group["ItemId"].tolist()
        ts_seq = group["Time"].tolist()
        act_sequences.append(act_seq)
        ts_sequences.append(ts_seq)

    return act_sequences, ts_sequences


def mine():
    # scorer_gap()
    er_miner()

def write_to_csv(title, selection_methods, simulators):
    header = ['selection_method', 'ndcg', 'hr_recall', 'accuracy_precision', 'dam_levenshtein', 'rel_dic_y', 'rel_div_gt', 'ild', 'div_f1', 'ildrr', 'inter-ld_pairwise', 'inter-ld_set_theoretic', 'testable_seq', 'canc_rec', 'success_rec']

    with open(title, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)
        for i, simulator in enumerate(simulators):
            data = [
                selection_methods[i],
                simulator.qualities.ndcg.mean_quality,
                simulator.qualities.hr.mean_quality,
                simulator.qualities.accuracy.mean_quality,
                simulator.qualities.damerau_levenshtein.mean_quality,
                simulator.qualities.rel_div.mean_quality_y,
                simulator.qualities.rel_div.mean_quality_gt,
                simulator.qualities.div.mean_quality,
                simulator.qualities.div_f1,
                simulator.qualities.ildrr.mean_quality,
                simulator.qualities.div.mean_inter_div_pairwise,
                simulator.qualities.div.mean_inter_div_set_theoretic,
                simulator.testable_sequences,
                simulator.qualities.cancelled_recommendations,
                simulator.qualities.successful_recommendations,
            ]
            # write the data
            writer.writerow(data)


if __name__ == "__main__":
    # mine()
    sample_size = None  # all test sequences are used
    bmswebview1_sim = run_exp(rule_path="datasets/bmswebview1/520000_filtered_rules_with_dgaps.pkl", dataset="bmswebview1", sample_size=sample_size)
#     bmswebview2_sim = run_exp(rule_path="datasets/bmswebview2/46428_filtered_rules_with_dgaps.pkl", dataset="bmswebview2", sample_size=sample_size)
    fifa_sim = run_exp(rule_path="datasets/fifa/64057_filtered_rules_with_dgaps.pkl", dataset="fifa", sample_size=sample_size)
#     amzn_arts_crafts_sewing_sim = run_exp(rule_path="datasets/arts_craft_sewing/8452_rules_with_both_gaps.pkl", dataset="arts_craft_sewing", sample_size=sample_size)
