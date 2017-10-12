import os
import unittest

import numpy as np

import namaco
from namaco.data.reader import load_data_and_labels
from namaco.data.preprocess import prepare_preprocessor
from namaco.config import ModelConfig, TrainingConfig
from namaco.models import CharacterNER


class TrainerTest(unittest.TestCase):

    def test_train(self):
        DATA_ROOT = os.path.join(os.path.dirname(__file__), 'data')
        SAVE_ROOT = os.path.join(os.path.dirname(__file__), 'models')  # trained model
        LOG_ROOT = os.path.join(os.path.dirname(__file__), 'logs')     # checkpoint, tensorboard

        model_config = ModelConfig()
        training_config = TrainingConfig()

        train_path = os.path.join(DATA_ROOT, 'conll.txt')
        valid_path = os.path.join(DATA_ROOT, 'conll.txt')
        test_path = os.path.join(DATA_ROOT, 'conll.txt')
        x_train, y_train = load_data_and_labels(train_path)
        x_valid, y_valid = load_data_and_labels(valid_path)
        x_test, y_test = load_data_and_labels(test_path)

        p = prepare_preprocessor(np.r_[x_train, x_valid, x_test], y_train)  # np.r_ is for vocabulary expansion.
        p.save(os.path.join(SAVE_ROOT, 'preprocessor.pkl'))

        model = CharacterNER(model_config, len(p.vocab_char), len(p.vocab_tag))
        loss = model.loss
        model = model.build()

        trainer = namaco.Trainer(model,
                                 loss,
                                 training_config,
                                 checkpoint_path=LOG_ROOT,
                                 save_path=SAVE_ROOT,
                                 preprocessor=p)
        trainer.train(x_train, y_train, x_test, y_test)
