# namaco
***namaco*** is a library for character-based Named Entity Recognition.
Especially, namaco will focus on Japanese and Chinese named entity recognition.

The following picture is an expected completion image:
<img src="https://github.com/Hironsan/anago/blob/docs/docs/images/example.ja2.png?raw=true">

## Feature Support
namaco would provide following features:
* learning model by your data.
* tagging sentences by learned model.


## Install
To install namaco, simply run:

```
$ pip install namaco
```

## Data format
The data must be in the following format(tsv).
The following snippet is an example:

```
安	B-PERSON
倍	E-PERSON
首	O
相	O
が	O
訪	O
米	S-LOC
し	O
た	O
 
本	B-DATE
日	E-DATE
```

## Get Started
### Import
First, import the necessary modules:
```python
import os
import namaco
from namaco.data.reader import load_data_and_labels
from namaco.data.preprocess import prepare_preprocessor
from namaco.config import ModelConfig, TrainingConfig
```
They include loading modules, a preprocessor and configs.


And set parameters to use later:
```python
DATA_ROOT = 'data/ja/ner'
SAVE_ROOT = './models'  # trained model
LOG_ROOT = './logs'     # checkpoint, tensorboard
model_config = ModelConfig()
training_config = TrainingConfig()
```

### Loading data

After importing the modules, read data for training, validation and test:
```python
train_path = os.path.join(DATA_ROOT, 'train.txt')
valid_path = os.path.join(DATA_ROOT, 'valid.txt')
test_path = os.path.join(DATA_ROOT, 'test.txt')
x_train, y_train = load_data_and_labels(train_path)
x_valid, y_valid = load_data_and_labels(valid_path)
x_test, y_test = load_data_and_labels(test_path)
```

After reading the data, prepare preprocessor:
```python
p = prepare_preprocessor(x_train, y_train)
model_config.vocab_size = len(p.vocab)
```

Now we are ready for training :)


### Training a model
Let's train a model. For training a model, we can use ***Trainer***. 
Trainer manages everything about training.
Prepare an instance of Trainer class and give train data and valid data to train method:
```
trainer = namaco.Trainer(model_config, training_config, checkpoint_path=LOG_ROOT, save_path=SAVE_ROOT,
                        preprocessor=p)
trainer.train(x_train, y_train, x_valid, y_valid)
```

If training is progressing normally, progress bar will be displayed as follows:

```commandline
...
Epoch 3/15
702/703 [============================>.] - ETA: 0s - loss: 60.0129 - f1: 89.70
703/703 [==============================] - 319s - loss: 59.9278   
Epoch 4/15
702/703 [============================>.] - ETA: 0s - loss: 59.9268 - f1: 90.03
703/703 [==============================] - 324s - loss: 59.8417   
Epoch 5/15
702/703 [============================>.] - ETA: 0s - loss: 58.9831 - f1: 90.67
703/703 [==============================] - 297s - loss: 58.8993   
...
```


### Evaluating a model
To evaluate the trained model, we can use ***Evaluator***.
Evaluator performs evaluation.
Prepare an instance of Evaluator class and give test data to eval method:

```
weights = 'model_weights.h5'

evaluator = namaco.Evaluator(model_config, weights, save_path=SAVE_ROOT, preprocessor=p)
evaluator.eval(x_test, y_test)
```

After evaluation, F1 value is output:
```commandline
- f1: 90.67
```

### Tagging a sentence
To tag any text, we can use ***Tagger***.
Prepare an instance of Tagger class and give text to tag method:
```
weights = 'model_weights.h5'
tagger = namaco.Tagger(model_config, weights, save_path=SAVE_ROOT, preprocessor=p)
```

Let's try tagging a sentence, `安倍首相が訪米した`
We can do it as follows:
```python
>>> sent = '安倍首相が訪米した'
>>> print(tagger.tag(sent))
[('安', 'B-PERSON'), ('倍', 'E-PERSON'), ('首', 'O'),
 ('相', 'O'), ('が', 'O'), ('訪', 'O'),
 ('米', 'S-LOCATION'), ('し', 'O'), ('た', 'O')]
>>> print(tagger.get_entities(sent))
{'PERSON': ['安倍'], 'LOCATION': ['米']}
```


