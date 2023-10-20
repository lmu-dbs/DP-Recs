import collections
import itertools
import logging
import math

import pandas as pd

import utils
from src.quality_meter.quality_measure import QualityMeasure

logger = logging.getLogger(__name__)


class RelativeDiversity(QualityMeasure):
    def __init__(self, y):
        super().__init__(y)
        self.mean_quality_y = None
        self.mean_quality_gt = None
        self.qualities_y = []
        self.qualities_gt = []

    def add_sequence(self, sequence, recommendations):
        pass

    def add_sequence_succ(self, new_sequence, gt_sequence):
        if len(new_sequence) > 0:
            self.qualities_y.append(utils.get_relative_diversity(new_sequence))
            self.qualities_gt.append(utils.get_relative_diversity(gt_sequence))

    def compute_quality(self):
        self.mean_quality_y = sum(self.qualities_y) / len(self.qualities_y)
        self.mean_quality_gt = sum(self.qualities_gt) / len(self.qualities_gt)

    def reset_values(self):
        self.qualities = []
        self.mean_quality = None
        self.mean_quality_y = None
        self.mean_quality_gt = None
        self.qualities_y = []
        self.qualities_gt = []
