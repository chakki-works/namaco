import os

import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from sklearn.model_selection import train_test_split

from namaco.utils import load_data_and_labels
from namaco.trainer import Trainer
from namaco.models import create_model
from tests.preprocess import StaticPreprocessor, DynamicPreprocessor


DATA_ROOT = os.path.join(os.path.dirname(__file__), 'data')
SAVE_ROOT = os.path.join(os.path.dirname(__file__), 'models')  # trained model
LOG_ROOT = os.path.join(os.path.dirname(__file__), 'logs')     # tensorboard
# EMBEDDING_PATH = os.path.join(DATA_ROOT, 'jawiki-embeddings/wiki.ja.word2vec.model')
EMBEDDING_PATH = os.path.join(DATA_ROOT, 'jawiki-embeddings-neologd/wiki.ja.neologd.word2vec.model')


def filter_embeddings(embeddings, vocab, dim):
    """Loads word vectors in numpy array.

    Args:
        embeddings (dict): a dictionary of numpy array.
        vocab (dict): word_index lookup table.

    Returns:
        numpy array: an array of word embeddings.
    """
    _embeddings = np.zeros([len(vocab), dim])
    for word in vocab:
        if word in embeddings:
            word_idx = vocab[word]
            _embeddings[word_idx] = embeddings[word]

    return _embeddings


if __name__ == '__main__':
    print('Loading datasets...')
    path = os.path.join(DATA_ROOT, 'datasets.tsv')
    X, y = load_data_and_labels(path)
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)
    embeddings = KeyedVectors.load(EMBEDDING_PATH).wv

    print('Transforming datasets...')
    p = StaticPreprocessor()
    p.fit(X, y)
    x_train, y_train = p.transform(x_train, y_train)
    x_valid, y_valid = p.transform(x_valid, y_valid)
    dp = DynamicPreprocessor(n_labels=len(p.label_dic))
    embeddings = filter_embeddings(embeddings, p.word_dic, embeddings.vector_size)

    print('Building a model...')
    model = create_model(char_vocab_size=len(p.char_dic),
                         word_vocab_size=len(p.word_dic),
                         pos_vocab_size=len(p.pos_dic),
                         ntags=len(p.label_dic),
                         embeddings=embeddings)

    print('Training the model...')
    trainer = Trainer(model, preprocessor=dp,
                      inverse_transform=p.inverse_transform)
    trainer.train(x_train, y_train, x_valid, y_valid)
