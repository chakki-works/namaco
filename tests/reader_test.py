import os
import unittest

from namaco.data.reader import load_data_and_labels, batch_iter
from namaco.data.preprocess import prepare_preprocessor


class ReaderTest(unittest.TestCase):

    def setUp(self):
        self.filename = os.path.join(os.path.dirname(__file__), 'data/conll.txt')

    def test_extract(self):
        sents, labels = load_data_and_labels(self.filename)
        self.assertTrue(len(sents) == len(labels))

    def test_batch_iter(self):
        sents, labels = load_data_and_labels(self.filename)
        batch_size = 32
        p = prepare_preprocessor(sents, labels)
        steps, batches = batch_iter(sents, labels, batch_size, preprocessor=p)
        for _ in range(steps):
            next(batches)
