from werkzeug.middleware import dispatcher
from flask import Flask, redirect
from werkzeug import serving
from threading import Thread
import tensorboard as tb
from lib.ml.ml import Ml
import os

PATH = os.path.dirname(__file__)
WORK_PATH = os.path.join(PATH, 'ml')

HOST = "0.0.0.0"
PORT = 7070


app = Flask(__name__)


@app.route('/')
def default(*args, **kwargs):
    return redirect('/tensorboard')


class CustomServer(tb.program.TensorBoardServer):
    def __init__(self, tensorboard_app, flags):
        del flags  # unused
        self._app = dispatcher.DispatcherMiddleware(
            app, {"/tensorboard": tensorboard_app}
        )

    def serve_forever(self):
        serving.run_simple(HOST, PORT, self._app)

    def get_url(self):
        return "http://%s:%s" % (HOST, PORT)

    def print_serving_message(self):
        pass  # Werkzeug's `serving.run_simple` handles this


def run_ml(uuid: str):
    ml = Ml(
        uuid=uuid,
        work_dir=WORK_PATH
    )

    ml.create()
    ml.train()


UUID = '59f5db7a-a41d-5166-90d8-207ca87fecc6'
if __name__ == "__main__":

    th1 = Thread(target=run_ml, args=[UUID])
    th1.start()

    program = tb.program.TensorBoard(server_class=CustomServer)
    program.configure(logdir=os.path.join(WORK_PATH, 'runs', UUID), load_fast=True)
    program.main()



