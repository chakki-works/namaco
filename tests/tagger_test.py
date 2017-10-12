# -*- coding: utf-8 -*-
import os
import unittest

import namaco
from namaco.config import ModelConfig
from namaco.data.preprocess import Preprocessor


class TaggerTest(unittest.TestCase):

    def setUp(self):
        SAVE_ROOT = os.path.join(os.path.dirname(__file__), 'models')

        model_config = ModelConfig()

        p = Preprocessor.load(os.path.join(SAVE_ROOT, 'preprocessor.pkl'))
        model_config.vocab_size = len(p.vocab_char)

        weights = 'model.h5'

        self.tagger = namaco.Tagger(model_config, weights, save_path=SAVE_ROOT, preprocessor=p, tokenizer=list)
        self.sent = '安倍首相が訪米した。'

    def test_tagging(self):
        res = self.tagger.tag(self.sent)
        print(res)
        self.assertIsInstance(res, list)
        self.assertIsInstance(res[0], tuple)
        self.assertEqual(len(res[0]), 2)
        self.assertIsInstance(res[0][0], str)
        self.assertIsInstance(res[0][1], str)

    def test_get_entities(self):
        res = self.tagger.get_entities(self.sent)
        print(res)
        self.assertIsInstance(list(res.keys())[0], str)
        self.assertIsInstance(list(res.values())[0], list)
        self.assertIsInstance(list(res.values())[0][0], str)
