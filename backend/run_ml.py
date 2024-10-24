from werkzeug.middleware import dispatcher
from flask import Flask, redirect
from werkzeug import serving
from threading import Thread
from numba import jit, cuda
import tensorboard as tb
from lib.ml.ml import Ml
from lib.ml.predict import Predict
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


UUID = '59f5db7a-a41d-5166-90d8-207ca87fecc6'
#UUID ='d1e83c9d-caa9-5bf0-89df-4238df995d06'
#UUID = 'f22a4d55-0655-5bb4-923d-ea1dbec39d58'
#UUID = '9e33b9f5-5fcb-54dc-abf2-7039a17dea05'


if __name__ == "__main__":
    ml = Ml( root=WORK_PATH, uuid=UUID)
    ml.train()

    pr = Predict( root=WORK_PATH, uuid=UUID)
    pr.predict()

    exit(0)

    th1 = Thread(target=main)
    th1.start()

    program = tb.program.TensorBoard(server_class=CustomServer)
    program.configure(logdir=ml.tensorboard_path, load_fast=True)
    program.main()



