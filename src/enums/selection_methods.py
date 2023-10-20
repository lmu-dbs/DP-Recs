from enum import Enum


class SelectionMethods(str, Enum):
    CGAP_ACC = "CGap Accumulated"
    UNIQUE_CONSEQUENT = "Unique Consequent"
    UNIQUE_CONSEQUENT_WITHIN_TIME_WINDOW = "Unique Consequent Within Time Window"
    CGAP = "CGap"
    DGAP = "DGap"
    DGAP_ACC = "DGap Accumulated"
    RANDOM = "Random"
    NAIVE = "Naive"
    FIRST = "First"
