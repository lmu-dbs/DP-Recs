from src.quality_meter.quality_measure import QualityMeasure
from utils import optimal_string_alignment_distance
from fastDamerauLevenshtein import damerauLevenshtein

class DamerauLevenshtein(QualityMeasure):
    def __init__(self, y):
        super().__init__(y)

    def add_sequence_succ(self, pred, gt):
        # self.qualities.append(optimal_string_alignment_distance(new_sequence, gt_sequence))
        self.qualities.append(damerauLevenshtein(pred, gt, True))

    def compute_quality(self):
        self.mean_quality = sum(self.qualities) / len(self.qualities)


