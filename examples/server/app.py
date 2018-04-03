import json
import os
import tornado.ioloop
import tornado.web
import tornado.httpserver
from tornado.web import url

from namaco.tagger import Tagger
from namaco.models import CharModel
from namaco.preprocess import StaticPreprocessor, DynamicPreprocessor, tokenize

# Files related to the model
SAVE_DIR = os.path.join(os.path.dirname(__file__), '../models')
weights_file = os.path.join(SAVE_DIR, 'model_weights_14_0.83.h5')
params_file = os.path.join(SAVE_DIR, 'params.json')
preprocessor_file = os.path.join(SAVE_DIR, 'preprocessor.json')

print('Loading objects...')
model = CharModel.load(weights_file, params_file)
sp = StaticPreprocessor.load(preprocessor_file)
dp = DynamicPreprocessor(n_labels=len(sp.label_dic))
tagger = Tagger(model, preprocessor=sp, dynamic_preprocessor=dp, tokenizer=tokenize)


class NERHandler(tornado.web.RequestHandler):

    def get(self):
        self.render('index2.html', sent='', entities=[])

    def post(self):
        sent = self.get_argument('sent')
        entities = tagger.analyze(sent)
        print(entities)
        if entities:
            self.render('index2.html', sent=sent, entities=entities['entities'])
            # self.write(json.dumps(entities))


BASE_DIR = os.path.dirname(__file__)


def main():
    application = tornado.web.Application([
        url(r'/named-entity-recognition', NERHandler, name='ner'),
    ],
        template_path=os.path.join(BASE_DIR, 'templates'),
        static_path=os.path.join(BASE_DIR, 'static'),
    )
    http_server = tornado.httpserver.HTTPServer(application)
    port = int(os.environ.get('PORT', 8000))
    http_server.listen(port)
    tornado.ioloop.IOLoop.instance().start()


if __name__ == '__main__':
    main()