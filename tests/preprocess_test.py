import os
import unittest

import numpy as np

from namaco.utils import load_data_and_labels
from namaco.preprocess import StaticPreprocessor, DynamicPreprocessor, UNK, pad_char


class TestStaticPreprocessor(unittest.TestCase):

    def setUp(self):
        self.p = StaticPreprocessor()

    @classmethod
    def setUpClass(cls):
        filename = os.path.join(os.path.dirname(__file__), './data/datasets.tsv')
        cls.X, cls.y = load_data_and_labels(filename)

    def test_preprocessor(self):
        X, y = self.p.fit_transform(self.X, self.y)
        words, chars = X
        for doc in words:
            for w in doc:
                self.assertIsInstance(w, np.int32)
        for doc in chars:
            for c in doc:
                self.assertIsInstance(c, np.int32)
