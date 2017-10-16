# -*- coding: utf-8 -*-
import os
import unittest

import namaco
from namaco.data.preprocess import Preprocessor


class TaggerTest(unittest.TestCase):

    def setUp(self):
        SAVE_ROOT = os.path.join(os.path.dirname(__file__), 'models')
        p = Preprocessor.load(os.path.join(SAVE_ROOT, 'preprocessor.pkl'))
        model_path = os.path.join(SAVE_ROOT, 'model.h5')

        self.tagger = namaco.Tagger(model_path, preprocessor=p, tokenizer=list)
        self.sent = '安倍首相が訪米した。'

    def test_tagging(self):
        res = self.tagger.tag(self.sent)
        print(res)
        self.assertIsInstance(res, list)
        self.assertIsInstance(res[0], tuple)
        self.assertEqual(len(res[0]), 2)
        self.assertIsInstance(res[0][0], str)
        self.assertIsInstance(res[0][1], str)

    def test_analyze(self):
        res = self.tagger.analyze(self.sent)
        print(res)
        self.assertEqual(res['text'], self.sent)
        self.assertTrue('entities' in res)
        self.assertIsInstance(res['entities'], list)
        if len(res['entities']) > 0:
            for entity in res['entities']:
                self.assertIsInstance(entity, dict)
                self.assertTrue('text' in entity)
                self.assertTrue('type' in entity)
                self.assertTrue('score' in entity)
                self.assertTrue('beginOffset' in entity)
                self.assertTrue('endOffset' in entity)
                self.assertIsInstance(entity['text'], str)
                self.assertIsInstance(entity['type'], str)
                self.assertIsInstance(entity['score'], float)
                self.assertIsInstance(entity['beginOffset'], int)
                self.assertIsInstance(entity['endOffset'], int)

    def test_get_entities(self):
        res = self.tagger.get_entities(self.sent)
        print(res)
        self.assertIsInstance(list(res.keys())[0], str)
        self.assertIsInstance(list(res.values())[0], list)
        self.assertIsInstance(list(res.values())[0][0], str)
