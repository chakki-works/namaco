class ModelConfig(object):
    """Wrapper class for model hyperparameters."""

    def __init__(self):
        """Sets the default model hyperparameters."""

        # LSTM input and output dimensionality, respectively.
        self.embedding_size = 100
        self.num_lstm_units = 100

        # If < 1.0, the dropout keep probability applied to LSTM variables.
        self.dropout = 0.5


class TrainingConfig(object):
    """Wrapper class for training hyperparameters."""

    def __init__(self):
        """Sets the default training hyperparameters."""

        # Batch size
        self.batch_size = 20

        # Optimizer for training the model.
        self.optimizer = 'adam'

        # Learning rate for the initial phase of training.
        self.learning_rate = 0.001
        self.lr_decay = 0.9

        # If not None, clip gradients to this value.
        self.clip_gradients = 5.0

        # The number of max epoch size
        self.max_epoch = 35

        # Parameters for early stopping
        self.early_stopping = True
        self.patience = 3

        # Fine-tune word embeddings
        self.train_embeddings = True
