from collections import defaultdict

import numpy as np

from namaco.data.metrics import get_entities
from namaco.models import load


class Tagger(object):

    def __init__(self,
                 model_path,
                 preprocessor,
                 tokenizer=str.split):

        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.model = load(model_path)

    def predict(self, words):
        length = np.array([len(words)])
        X = self.preprocessor.transform([words])
        # print(X)
        # pred = self.model.predict([X[0], length])
        pred = self.model.predict([X[0], length, X[2]])
        # pred = np.argmax(pred, -1)
        # pred = self.preprocessor.inverse_transform(pred[0])

        return pred

    def _get_tags(self, pred):
        pred = np.argmax(pred, -1)
        tags = self.preprocessor.inverse_transform(pred[0])

        return tags

    def _get_prob(self, pred):
        prob = np.max(pred, -1)[0]

        return prob

    def _build_response(self, sent, tags, prob):
        res = {
            'language': 'jp',
            'text': sent,
            'entities': [

            ]
        }
        chunks = get_entities(tags)
        for chunk_type, chunk_start, chunk_end in chunks:
            entity = {
                'text': sent[chunk_start: chunk_end],
                'type': chunk_type,
                'score': float(np.average(prob[chunk_start: chunk_end])),
                'beginOffset': chunk_start,
                'endOffset': chunk_end
            }
            res['entities'].append(entity)

        return res

    def analyze(self, sent):
        assert isinstance(sent, str)

        words = self.tokenizer(sent)
        pred = self.predict(words)
        tags = self._get_tags(pred)
        prob = self._get_prob(pred)
        res = self._build_response(sent, tags, prob)

        return res

    def tag(self, sent):
        """Tags a sentence named entities.

        Args:
            sent: a sentence

        Return:
            labels_pred: list of (word, tag) for a sentence

        Example:
            >>> sent = 'President Obama is speaking at the White House.'
            >>> print(self.tag(sent))
            [('President', 'O'), ('Obama', 'PERSON'), ('is', 'O'),
             ('speaking', 'O'), ('at', 'O'), ('the', 'O'),
             ('White', 'LOCATION'), ('House', 'LOCATION'), ('.', 'O')]
        """
        assert isinstance(sent, str)

        words = self.tokenizer(sent)
        pred = self.predict(words)
        tags = self._get_tags(pred)
        tags = [t.split('-')[-1] for t in tags]  # remove prefix: e.g. B-Person -> Person

        return list(zip(words, tags))

    def get_entities(self, sent):
        """Gets entities from a sentence.

        Args:
            sent: a sentence

        Return:
            labels_pred: dict of entities for a sentence

        Example:
            sent = 'President Obama is speaking at the White House.'
            result = {'Person': ['Obama'], 'LOCATION': ['White House']}
        """
        assert isinstance(sent, str)

        words = self.tokenizer(sent)
        pred = self.predict(words)
        entities = self._get_chunks(words, pred)

        return entities

    def _get_chunks(self, words, tags):
        """
        Args:
            words: sequence of word
            tags: sequence of labels

        Returns:
            dict of entities for a sequence

        Example:
            words = ['President', 'Obama', 'is', 'speaking', 'at', 'the', 'White', 'House', '.']
            tags = ['O', 'B-Person', 'O', 'O', 'O', 'O', 'B-Location', 'I-Location', 'O']
            result = {'Person': ['Obama'], 'LOCATION': ['White House']}
        """
        chunks = get_entities(tags)
        res = defaultdict(list)
        for chunk_type, chunk_start, chunk_end in chunks:
            res[chunk_type].append(' '.join(words[chunk_start: chunk_end]))  # todo delimiter changeable

        return res
