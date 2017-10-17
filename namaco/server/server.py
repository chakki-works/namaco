import json
import os
import tornado.ioloop
import tornado.web

import namaco
from namaco.data.preprocess import Preprocessor

SAVE_ROOT = os.path.join(os.path.dirname(__file__), '../../tests/models')
p = Preprocessor.load(os.path.join(SAVE_ROOT, 'preprocessor.pkl'))
model_path = os.path.join(SAVE_ROOT, 'model.h5')
tagger = namaco.Tagger(model_path, preprocessor=p, tokenizer=list)


class MainHandler(tornado.web.RequestHandler):

    def get(self):
        self.render('index.html', sent='')

    def post(self):
        sent = self.get_argument('sent')
        entities = tagger.analyze(sent)
        print(entities)
        if entities:
            self.write(json.dumps(entities))


BASE_DIR = os.path.dirname(__file__)

application = tornado.web.Application([
    (r'/', MainHandler),
    ],
    template_path=os.path.join(BASE_DIR, 'templates'),
    static_path=os.path.join(BASE_DIR, 'static'),
)

if __name__ == '__main__':
    application.listen(8888)
    tornado.ioloop.IOLoop.current().start()
