# -*- coding: utf-8 -*-
"""
Preprocessors.
"""
import itertools
import re

import MeCab
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

UNK = '<UNK>'
PAD = '<PAD>'
t = MeCab.Tagger('-Owakati')


def tokenize(text):
    words = t.parse(text).rstrip().split()

    return words


def normalize_number(text):
    return re.sub(r'[0-9０１２３４５６７８９]', r'0', text)


class StaticPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, lowercase=True, num_norm=True, vocab_init=None):
        self._lowercase = lowercase
        self._num_norm = num_norm
        self._vocab_init = vocab_init or {}
        self.word_dic = {PAD: 0, UNK: 1}
        self.char_dic = {PAD: 0, UNK: 1}
        self.label_dic = {PAD: 0}

    def fit(self, X, y=None):
        for doc in X:
            text = ''.join(doc)
            if self._lowercase:
                text = text.lower()
            if self._num_norm:
                text = normalize_number(text)
            words = tokenize(text)
            for w in words:
                if w in self.word_dic:
                    continue
                self.word_dic[w] = len(self.word_dic)
                for c in w:
                    if c in self.char_dic:
                        continue
                    self.char_dic[c] = len(self.char_dic)

        # create label dictionary
        for t in set(itertools.chain(*y)):
            self.label_dic[t] = len(self.label_dic)

        return self

    def transform(self, X, y=None):
        x_words = []
        x_chars = []
        for doc in X:
            text = ''.join(doc)
            if self._lowercase:
                text = text.lower()
            if self._num_norm:
                text = normalize_number(text)
            words = tokenize(text)
            word_ids = [[self.word_dic.get(w, self.word_dic[UNK]) for _ in range(len(w))]
                        for w in words]
            char_ids = [self._get_char_ids(w) for w in words]
            word_ids = list(itertools.chain(*word_ids))
            char_ids = list(itertools.chain(*char_ids))
            x_words.append(np.array(word_ids, dtype=np.int32))
            x_chars.append(np.array(char_ids, dtype=np.int32))

            assert len(char_ids) == len(word_ids)

        if y is not None:
            y = np.array([[self.label_dic[t] for t in sent] for sent in y])

        inputs = [np.array(x_words), np.array(x_chars)]

        return (inputs, y) if y is not None else inputs

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X, y)

    def inverse_transform(self, docs):
        id2label = {i: t for t, i in self.label_dic.items()}

        return [[id2label[t] for t in doc] for doc in docs]

    def _get_char_ids(self, word):
        return [self.char_dic.get(c, self.char_dic[UNK]) for c in word]

    def save(self, file_path):
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path):
        p = joblib.load(file_path)

        return p


class DynamicPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, n_labels):
        self.n_labels = n_labels

    def transform(self, X, y=None):
        words, chars = X
        words = pad_sequences(words, padding='post')
        chars = pad_sequences(chars, padding='post')

        if y is not None:
            y = pad_sequences(y, padding='post')
            y = np.array([to_categorical(y_, self.n_labels) for y_ in y])
        sents = [words, chars]

        return (sents, y) if y is not None else sents

    def save(self, file_path):
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path):
        p = joblib.load(file_path)

        return p
