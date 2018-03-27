import os
import unittest

from sklearn.model_selection import train_test_split

from namaco.utils import load_data_and_labels
from namaco.preprocess import StaticPreprocessor, DynamicPreprocessor
from namaco.trainer import Trainer
from namaco.models import CharNER


DATA_ROOT = os.path.join(os.path.dirname(__file__), 'data')
SAVE_ROOT = os.path.join(os.path.dirname(__file__), 'models')  # trained model
LOG_ROOT = os.path.join(os.path.dirname(__file__), 'logs')     # tensorboard


class TrainerTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(LOG_ROOT):
            os.mkdir(LOG_ROOT)

        if not os.path.exists(SAVE_ROOT):
            os.mkdir(SAVE_ROOT)

    def setUp(self):
        print('Loading datasets...')
        path = os.path.join(DATA_ROOT, 'datasets.tsv')
        X, y = load_data_and_labels(path)
        x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)

        print('Transforming datasets...')
        self.p = StaticPreprocessor()
        self.p.fit(X, y)
        self.x_train, self.y_train = self.p.transform(x_train, y_train)
        self.x_valid, self.y_valid = self.p.transform(x_valid, y_valid)
        self.dp = DynamicPreprocessor(n_labels=len(self.p.label_dic))

        print('Building a model...')
        self.model = CharNER(char_vocab_size=len(self.p.char_dic),
                             word_vocab_size=len(self.p.word_dic),
                             ntags=len(self.p.label_dic))

    def test_train(self):
        print('Training the model...')
        trainer = Trainer(self.model, preprocessor=self.dp,
                          inverse_transform=self.p.inverse_transform)
        trainer.train(self.x_train, self.y_train, self.x_valid, self.y_valid)
