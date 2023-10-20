import logging
import pandas as pd

logger = logging.getLogger(__name__)

class RuleReader:
    """
    Loads a given file of rules and transforms the SPMF format into a list of given events.
    """
    def __init__(self, rules, from_path=False):
        self.rules = rules
        self.from_path = from_path
        self.data = pd.DataFrame(columns=['rule', 'antecedent', 'consequent', 'support', 'confidence'])
        if from_path:
            self.read_input_from_path()
        else:
            self.read_input()

    def read_input(self):
        lines = self.rules.readlines()
        self.transform_input(lines)

    def read_input_from_path(self):
        lines = None
        with open(self.rules, "r") as f:
            lines = f.readlines()
        self.transform_input(lines)

    def transform_input(self, d):
        logger.info(f"Number of rules: {len(d)}")
        i = 0
        for sequence in d:
            temp = []
            i += 1
            if not self.from_path:
                sequence = sequence.decode("utf-8")
            for x in sequence.strip().split(" #"):
                if '==>' in x:
                    temp += [x]
                    rule = x.split(" ==> ")
                    antecedent = rule[0].split(",")
                    consequent = rule[1].split(",")
                    # tuples are immutable as lists but hashable (required for groupby operation)
                    antecedent_int = tuple([int(y) for y in antecedent])
                    consequent_int = tuple([int(y) for y in consequent])
                    temp.append(antecedent_int)
                    temp.append(consequent_int)
                elif 'CONF:' in x:
                    temp.append(float(x.split(" ")[1]))
                elif 'SUP:' in x and not 'DECSUP:' in x:
                    temp.append(float(x.split(" ")[1]))
            self.data.loc[i] = temp
        logger.info("Finished transforming rules")

