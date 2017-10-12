from keras.layers import Dense, LSTM, Bidirectional, Embedding, Input, Dropout
from keras.models import Model, load_model

from namaco.layers import ChainCRF, create_custom_objects


def load(filepath):
    return load_model(filepath, custom_objects=create_custom_objects())


class CharacterNER(object):

    def __init__(self, config, vocab_size, ntags):
        sequence_lengths = Input(batch_shape=(None, 1), dtype='int32')
        char_ids = Input(batch_shape=(None, None), dtype='int32')
        char_embeddings = Embedding(input_dim=vocab_size,
                                    output_dim=config.embedding_size,
                                    mask_zero=True)(char_ids)
        x = Dropout(config.dropout)(char_embeddings)

        x = Bidirectional(LSTM(units=config.num_lstm_units, return_sequences=True))(x)
        x = Dropout(config.dropout)(x)
        x = Dense(config.num_lstm_units, activation='tanh')(x)
        x = Dense(ntags)(x)
        crf = ChainCRF()
        pred = crf(x)

        self.loss = crf.loss
        self.model = Model(inputs=[char_ids, sequence_lengths], outputs=[pred])

    def build(self):
        return self.model
