from keras.layers import Dense, LSTM, Bidirectional, Embedding, Input, Dropout
from keras.models import Model, load_model

from namaco.layers import ChainCRF, create_custom_objects


class BaseModel(object):

    def __init__(self, config, ntags):
        self.config = config
        self.ntags = ntags
        self.model = None

    def predict(self, X, *args, **kwargs):
        y_pred = self.model.predict(X, batch_size=1)
        return y_pred

    def evaluate(self, X, y):
        score = self.model.evaluate(X, y, batch_size=1)
        return score

    def save(self, filepath):
        self.model.save(filepath)

    def load(self, filepath):
        self.model = load_model(filepath, custom_objects=create_custom_objects())

    def __getattr__(self, name):
        return getattr(self.model, name)


class CharacterNER(BaseModel):

    def __init__(self, config, ntags=None):
        sequence_lengths = Input(batch_shape=(None, 1), dtype='int32')
        char_ids = Input(batch_shape=(None, None), dtype='int32')
        char_embeddings = Embedding(input_dim=config.vocab_size,
                                    output_dim=config.embedding_size,
                                    mask_zero=True)(char_ids)
        x = Dropout(config.dropout)(char_embeddings)

        x = Bidirectional(LSTM(units=config.num_lstm_units, return_sequences=True))(x)
        x = Dropout(config.dropout)(x)
        x = Dense(config.num_lstm_units, activation='tanh')(x)
        x = Dense(ntags)(x)
        self.crf = ChainCRF()
        pred = self.crf(x)

        self.model = Model(inputs=[char_ids, sequence_lengths], outputs=[pred])
        self.config = config
