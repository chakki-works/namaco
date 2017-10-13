import os
import unittest

import namaco
from namaco.data.reader import load_data_and_labels
from namaco.data.preprocess import prepare_preprocessor
from namaco.config import ModelConfig, TrainingConfig
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

    def test_train(self):
        model_config = ModelConfig()
        training_config = TrainingConfig()

        train_path = os.path.join(DATA_ROOT, 'conll.txt')
        valid_path = os.path.join(DATA_ROOT, 'conll.txt')
        x_train, y_train = load_data_and_labels(train_path)
        x_valid, y_valid = load_data_and_labels(valid_path)

        p = prepare_preprocessor(x_train, y_train)
        p.save(os.path.join(SAVE_ROOT, 'preprocessor.pkl'))
        model_file = os.path.join(SAVE_ROOT, 'model.h5')

        model = CharNER(model_config, p.vocab_size(), p.tag_size())

        trainer = namaco.Trainer(model,
                                 model.loss,
                                 training_config,
                                 log_dir=LOG_ROOT,
                                 save_path=model_file,
                                 preprocessor=p)
        trainer.train(x_train, y_train, x_valid, y_valid)
