from namaco.data.metrics import F1score
from namaco.data.reader import batch_iter
from namaco.models import load


class Evaluator(object):

    def __init__(self,
                 model_path,
                 preprocessor=None):

        self.model = load(model_path)
        self.preprocessor = preprocessor

    def eval(self, x_test, y_test):

        # Prepare test data(steps, generator)
        train_steps, train_batches = batch_iter(
            x_test, y_test, batch_size=32, preprocessor=self.preprocessor, shuffle=False)

        # Build the evaluator and evaluate the model
        f1score = F1score(train_steps, train_batches, self.preprocessor)
        f1score.model = self.model
        f1score.on_epoch_end(epoch=-1)  # epoch takes any integer.
