"""
Model definition.
"""
import json

from keras.layers import Dense, LSTM, Bidirectional, Embedding, Input, Dropout, BatchNormalization, SimpleRNN
from keras.models import Model
from keras.layers.merge import Concatenate


class BaseModel(object):

    def __init__(self):
        self.model = None

    def save(self, weights_file, params_file):
        self.save_weights(weights_file)
        self.save_params(params_file)

    def save_weights(self, file_path):
        self.model.save_weights(file_path)

    def save_params(self, file_path):
        with open(file_path, 'w') as f:
            params = {name.lstrip('_'): val for name, val in vars(self).items()
                      if name not in {'_embeddings', '_loss', 'model'}}
            json.dump(params, f, sort_keys=True, indent=4)

    @classmethod
    def load(cls, weights_file, params_file):
        params = cls.load_params(params_file)
        self = cls(**params)
        self.build()
        self.load_weights(weights_file)

        return self

    @classmethod
    def load_params(cls, file_path):
        with open(file_path) as f:
            params = json.load(f)

        return params

    def __getattr__(self, name):
        return getattr(self.model, name)


class CharModel(BaseModel):

    def __init__(self, char_vocab_size, word_vocab_size, pos_vocab_size, ntags,
                 char_emb_size=50, word_emb_size=100, pos_emb_size=25,
                 char_lstm_units=25, word_lstm_units=50, pos_lstm_units=25,
                 dropout=0.5, embeddings=None):
        super(CharModel).__init__()
        self._char_emb_size = char_emb_size
        self._word_emb_size = word_emb_size
        self._pos_emb_size = pos_emb_size
        self._char_lstm_units = char_lstm_units
        self._word_lstm_units = word_lstm_units
        self._pos_lstm_units = pos_lstm_units
        self._char_vocab_size = char_vocab_size
        self._word_vocab_size = word_vocab_size
        self._pos_vocab_size = pos_vocab_size
        self._dropout = dropout
        self._embeddings = embeddings
        self._ntags = ntags
        self._loss = None

    def build(self):
        word_ids = Input(batch_shape=(None, None), dtype='int32')
        char_ids = Input(batch_shape=(None, None), dtype='int32')
        bies_ids = Input(batch_shape=(None, None), dtype='int32')
        pos_ids = Input(batch_shape=(None, None), dtype='int32')

        char_embeddings = Embedding(input_dim=self._char_vocab_size,
                                    output_dim=self._char_emb_size,
                                    mask_zero=True)(char_ids)
        bies_embeddings = Embedding(input_dim=5,
                                    output_dim=5,
                                    mask_zero=True)(bies_ids)
        pos_embeddings = Embedding(input_dim=self._pos_vocab_size,
                                   output_dim=self._pos_emb_size,
                                   mask_zero=True)(pos_ids)

        x1 = Bidirectional(LSTM(units=self._char_lstm_units, return_sequences=True))(char_embeddings)

        if self._embeddings is None:
            word_embeddings = Embedding(input_dim=self._word_vocab_size,
                                        output_dim=self._word_emb_size,
                                        mask_zero=True)(word_ids)
        else:
            word_embeddings = Embedding(input_dim=self._embeddings.shape[0],
                                        output_dim=self._embeddings.shape[1],
                                        mask_zero=True,
                                        weights=[self._embeddings])(word_ids)

        word_embeddings = Concatenate()([word_embeddings, bies_embeddings])
        word_embeddings = BatchNormalization()(word_embeddings)
        x2 = Bidirectional(LSTM(units=self._word_lstm_units, return_sequences=True))(word_embeddings)

        pos_embeddings = Concatenate()([pos_embeddings, bies_embeddings])
        pos_embeddings = BatchNormalization()(pos_embeddings)
        x3 = Bidirectional(LSTM(units=self._pos_lstm_units, return_sequences=True))(pos_embeddings)

        x = Concatenate()([x1, x2, x3])
        x = BatchNormalization()(x)
        x = Dropout(self._dropout)(x)
        x = Dense(self._word_lstm_units, activation='tanh')(x)
        pred = SimpleRNN(units=self._ntags, activation='softmax', return_sequences=True)(x)

        self.model = Model(inputs=[word_ids, char_ids, bies_ids, pos_ids], outputs=[pred])
