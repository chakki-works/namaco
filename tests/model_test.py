import os
import unittest

from keras.optimizers import Adam

from namaco.config import ModelConfig, TrainingConfig
from namaco.data.preprocess import prepare_preprocessor
from namaco.data.reader import load_data_and_labels
from namaco.models import CharNER


class ModelTest(unittest.TestCase):

    def setUp(self):
        self.model_config = ModelConfig()
        self.training_config = TrainingConfig()
        self.model_config.vocab_size = 100
        self.filename = os.path.join(os.path.dirname(__file__), 'data/conll.txt')
        self.valid_file = os.path.join(os.path.dirname(__file__), 'data/conll.txt')

    def test_build(self):
        model = CharNER(self.model_config, vocab_size=100, ntags=10)

    def test_compile(self):
        model = CharNER(self.model_config, vocab_size=100, ntags=10)
        model.compile(loss=model.loss,
                      optimizer=Adam(lr=self.training_config.learning_rate)
                      )

    def test_predict(self):
        X, y = load_data_and_labels(self.filename)
        p = prepare_preprocessor(X, y)
        self.model_config.vocab_size = len(p.vocab_char)

        model = CharNER(self.model_config, p.vocab_size(), p.tag_size())
        model.predict(p.transform(X))

    def test_save(self):
        model = CharNER(self.model_config, vocab_size=100, ntags=10)
        path = 'data/test.h5'
        model.save(path)
        self.assertTrue(os.path.exists(path))
        if os.path.exists(path):
            os.remove(path)

    def test_load(self):
        model = CharNER(self.model_config, vocab_size=100, ntags=10)
        path = 'data/test.h5'
        model.save(path)
        self.assertTrue(os.path.exists(path))
        if os.path.exists(path):
            os.remove(path)
