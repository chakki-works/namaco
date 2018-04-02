"""
Tagging example.
"""
import argparse
import os
from pprint import pprint

from namaco.tagger import Tagger
from namaco.models import CharModel
from namaco.preprocess import StaticPreprocessor, DynamicPreprocessor, tokenize


def main(args):
    print('Loading objects...')
    model = CharModel.load(args.weights_file, args.params_file)
    sp = StaticPreprocessor.load(args.preprocessor_file)
    dp = DynamicPreprocessor(n_labels=len(sp.label_dic))
    tagger = Tagger(model, preprocessor=sp, dynamic_preprocessor=dp, tokenizer=tokenize)

    print('Tagging a sentence...')
    res = tagger.analyze(args.sent)
    pprint(res)


if __name__ == '__main__':
    SAVE_DIR = os.path.join(os.path.dirname(__file__), 'models')
    parser = argparse.ArgumentParser(description='Tagging a sentence.')
    parser.add_argument('--sent', default='太郎は東京に出かけた。')
    parser.add_argument('--save_dir', default=SAVE_DIR)
    parser.add_argument('--weights_file', default=os.path.join(SAVE_DIR, 'model_weights.h5'))
    parser.add_argument('--params_file', default=os.path.join(SAVE_DIR, 'params.json'))
    parser.add_argument('--preprocessor_file', default=os.path.join(SAVE_DIR, 'preprocessor.json'))
    args = parser.parse_args()
    main(args)
