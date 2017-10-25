# -*- coding: utf-8 -*-
import os
import unittest

import numpy as np

from namaco.data import reader
from namaco.data.preprocess import Preprocessor, UNK, PAD, dense_to_one_hot, label_bies_tags


class PreprocessorTest(unittest.TestCase):

    def setUp(self):
        self.filename = os.path.join(os.path.dirname(__file__), 'data/conll.txt')

    def test_fit(self):
        X, y = reader.load_data_and_labels(self.filename)
        p = Preprocessor(padding=False)
        p = p.fit(X, y)
        self.assertTrue(PAD in p.vocab_char)
        self.assertTrue(UNK in p.vocab_char)
        self.assertTrue(PAD in p.vocab_tag)
        char_set = set(p.vocab_char) - {PAD, UNK}
        for ch in char_set:
            self.assertEqual(len(ch), 1)

    def test_transform(self):
        X, y = reader.load_data_and_labels(self.filename)
        preprocessor = Preprocessor(padding=False, return_lengths=False)
        p = preprocessor.fit(X, y)
        X, y = p.transform(X, y)
        char_id = X[0][0]  # 1th character of 1th sent.
        tag_id = y[0][0]   # 1th tag of 1th sent.
        self.assertIsInstance(char_id, int)
        self.assertIsInstance(tag_id, int)

    def test_transform_with_padding(self):
        X, y = reader.load_data_and_labels(self.filename)
        preprocessor = Preprocessor(padding=True, return_lengths=False)
        p = preprocessor.fit(X, y)
        X, y = p.transform(X, y)
        char_id = X[0][0]
        tag_id = y[0][0]
        self.assertIsInstance(char_id, np.int32)
        self.assertIsInstance(tag_id, np.int32)

        length_set1 = set(map(len, X))
        length_set2 = set(map(len, y))
        self.assertEqual(len(length_set1), 1)  # all sequence has same length.
        self.assertEqual(len(length_set2), 1)

    def test_return_lengths(self):
        X, y = reader.load_data_and_labels(self.filename)
        preprocessor = Preprocessor(padding=False, return_lengths=True)
        p = preprocessor.fit(X, y)
        X, y = p.transform(X, y)
        chars, lengths = X
        char_id = chars[0][0]
        tag_id = y[0][0]
        self.assertIsInstance(char_id, int)
        self.assertIsInstance(tag_id, int)

        for seq, leng in zip(chars, lengths):
            self.assertEqual(len(seq), leng)

    def test_use_word(self):
        X, y = reader.load_data_and_labels(self.filename)
        preprocessor = Preprocessor(padding=True, return_lengths=True, use_word=True)
        p = preprocessor.fit(X, y)
        X, y = p.transform(X, y)
        chars, lengths, bies_tags = X
        print(chars)
        print(lengths)
        print(bies_tags)

    def test_unknown_word(self):
        X, y = reader.load_data_and_labels(self.filename)
        preprocessor = Preprocessor(padding=False, return_lengths=False)
        p = preprocessor.fit(X, y)
        X = [['$unknownword$']]
        X = p.transform(X)
        self.assertEqual(X[0][0], p.vocab_char[UNK])

    def test_save(self):
        preprocessor = Preprocessor()
        filepath = os.path.join(os.path.dirname(__file__), 'data/preprocessor.pkl')
        preprocessor.save(filepath)
        self.assertTrue(os.path.exists(filepath))
        if os.path.exists(filepath):
            os.remove(filepath)

    def test_load(self):
        X, y = reader.load_data_and_labels(self.filename)
        p = Preprocessor()
        p.fit(X, y)
        filepath = os.path.join(os.path.dirname(__file__), 'data/preprocessor.pkl')
        p.save(filepath)
        self.assertTrue(os.path.exists(filepath))

        loaded_p = Preprocessor.load(filepath)
        x_test1, y_test1 = p.transform(X, y)
        x_test2, y_test2 = loaded_p.transform(X, y)
        np.testing.assert_array_equal(x_test1[0], x_test2[0])  # word
        np.testing.assert_array_equal(x_test1[1], x_test2[1])  # char
        np.testing.assert_array_equal(y_test1, y_test2)
        if os.path.exists(filepath):
            os.remove(filepath)


class PreprocessTest(unittest.TestCase):

    def test_dense_to_onehot(self):
        # 1d vector
        labels = np.array([1, 2, 3])
        labels_one_hot = dense_to_one_hot(labels, num_classes=9)
        for labels in labels_one_hot:
            self.assertEqual(sum(labels), 1)

        # 2d matrix
        labels = np.array([[1, 2, 3],
                           [4, 5, 6]])
        labels_one_hot = dense_to_one_hot(labels, num_classes=9, nlevels=2)
        for labels in labels_one_hot:
            for l in labels:
                self.assertEqual(sum(l), 1)

        # nlevels test
        with self.assertRaises(ValueError):
            labels_one_hot == dense_to_one_hot(labels, num_classes=9, nlevels=3)


class MorphTagging(unittest.TestCase):

    def test_tagging(self):
        sent = '安倍首相が訪米した'
        y_pred = label_bies_tags(sent)
        y_true = ['B', 'E', 'B', 'E', 'S', 'B', 'E', 'S', 'S']
        y_true = [2, 4, 2, 4, 1, 2, 4, 1, 1]
        self.assertEqual(y_pred, y_true)

        sent = 'マクマスター国務長官'
        y_pred = label_bies_tags(sent)
        y_true = [2, 3, 3, 3, 3, 4, 2, 4, 2, 4]
        self.assertEqual(y_pred, y_true)