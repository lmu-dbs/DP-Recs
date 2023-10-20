from src.quality_meter.quality_measure import QualityMeasure


class Accuracy(QualityMeasure):
    def __init__(self, y):
        super().__init__(y)

    def add_sequence_succ(self, new_sequence, gt_sequence):
        # print(new_sequence)
        # print(gt_sequence)
        if new_sequence[-1] == gt_sequence[len(new_sequence)-1]:
            self.qualities.append(1)
        else:
            self.qualities.append(0)

    def compute_quality(self):
        # print("Computing accuracy average")
        self.mean_quality = sum(self.qualities) / len(self.qualities)

