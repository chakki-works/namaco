from keras.layers import Dense, LSTM, Bidirectional, Embedding, Input, Dropout, BatchNormalization
from keras.models import Model
from keras.layers.merge import Concatenate


def CharNER(char_vocab_size, word_vocab_size, ntags,
            char_embedding_size=50, word_embedding_size=100,
            num_lstm_units=50, dropout=0.5, embeddings=None):

    word_ids = Input(batch_shape=(None, None), dtype='int32')
    char_ids = Input(batch_shape=(None, None), dtype='int32')
    bies_ids = Input(batch_shape=(None, None), dtype='int32')
    char_embeddings = Embedding(input_dim=char_vocab_size,
                                output_dim=char_embedding_size,
                                mask_zero=True)(char_ids)
    bies_embeddings = Embedding(input_dim=5,
                                output_dim=5,
                                mask_zero=True)(bies_ids)
    char_embeddings = Concatenate()([char_embeddings, bies_embeddings])
    # x = Dropout(dropout)(char_embeddings)
    x = Bidirectional(LSTM(units=num_lstm_units, return_sequences=True))(char_embeddings)

    if embeddings is None:
        word_embeddings = Embedding(input_dim=word_vocab_size,
                                    output_dim=word_embedding_size,
                                    mask_zero=True)(word_ids)
    else:
        word_embeddings = Embedding(input_dim=embeddings.shape[0],
                                    output_dim=embeddings.shape[1],
                                    mask_zero=True,
                                    weights=[embeddings])(word_ids)

    x = Concatenate(axis=-1)([x, word_embeddings])
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    x = Dense(num_lstm_units, activation='tanh')(x)
    # pred = Dense(ntags, activation='softmax')(x)
    pred = LSTM(units=ntags, activation='softmax', return_sequences=True)(x)

    model = Model(inputs=[word_ids, char_ids, bies_ids], outputs=[pred])

    return model
