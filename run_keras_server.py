import yaml
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from threading import Thread
from PIL import Image
import numpy as np
import base64
import flask
import redis
import uuid
import time
import json
import sys
import io
import logging.config

# initialize constants used to control image spatial dimensions and data type
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANS = 3
IMAGE_DTYPE = "float32"

# initialize constants used for server queuing
IMAGE_QUEUE = "image_queue"
BATCH_SIZE = 32
SERVER_SLEEP = 0.25
CLIENT_SLEEP = 0.25

# initialize our Flask application, Redis server, and Keras model
app = flask.Flask(__name__)
db = redis.Redis(host="localhost", port=6379, db=0)
model = None


def base64_encode_image(a):
    # base64 encode the input NumPy array
    return base64.b64encode(a).decode("utf-8")


def base64_decode_image(a, dtype, shape):
    # if this is Python 3, we need the extra step of encoding the
    # serialized NumPy string as a byte object
    if sys.version_info.major == 3:
        a = bytes(a, encoding="utf-8")

    # convert the string to a NumPy array using the supplied data
    # type and target shape
    a = np.frombuffer(base64.decodestring(a), dtype=dtype)
    a = a.reshape(shape)

    # return the decoded image
    return a


def prepare_image(image, target):

    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    return image


def classify_process():
    app.logger.debug("* Loading model...")
    model = ResNet50(weights="imagenet")  # IDEA: own models
    app.logger.debug("* Model loaded")

    while True:

        queue = db.lrange(IMAGE_QUEUE, 0, BATCH_SIZE - 1)
        imageIDs = []
        batch = None

        for q in queue:
            q = json.loads(q.decode("utf-8"))
            image = base64_decode_image(q["image"], IMAGE_DTYPE,
                                        (1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANS))
            app.logger.debug("Deserialized image")

            if batch is None:
                batch = image
            else:
                batch = np.vstack([batch, image])

            imageIDs.append(q["id"])

            if len(imageIDs) > 0:

                print("* Batch size: {}".format(batch.shape))
                preds = model.predict(batch)
                results = imagenet_utils.decode_predictions(preds)
                app.logger.debug("Classified a batch of images and decoded predictions")

                for (imageID, resultSet) in zip(imageIDs, results):

                    output = []

                    for (imagenetID, label, prob) in resultSet:
                        r = {"label": label, "probability": float(prob)}
                        output.append(r)
                        app.logger.debug("Broke down predictions for the image {}".format(imagenetID))

                    db.set(imageID, json.dumps(output))
                    app.logger.debug("Stored predictions in the db")

                db.ltrim(IMAGE_QUEUE, len(imageIDs), -1)
                app.logger.debug("Removed the set of img from the db")

            time.sleep(SERVER_SLEEP)


@app.route("/classify", methods=["POST"])
def predict():
    app.logger.debug('entered classify endpoint, ImageNet')

    data = {"success": False}

    if flask.request.method == "POST":
        app.logger.debug('POST method check successful')

        image = flask.request.files["image"].read()
        image = Image.open(io.BytesIO(image))
        image = prepare_image(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        app.logger.debug('image converted to PIL')

        image = image.copy(order="C")
        app.logger.debug('NumPy array is C-contiguous for serialization')

        k = str(uuid.uuid4())
        app.logger.debug('ID is generated and equals {}'.format(k))

        d = {"id": k, "image": base64_encode_image(image)}
        app.logger.debug('ID assigned to the image')

        db.rpush(IMAGE_QUEUE, json.dumps(d))
        app.logger.debug('image pushed to the queue')

        while True:

            classified = db.get(k)
            app.logger.debug('getting predictions from the queue')
            app.logger.debug('classified = {} '.format(classified))

            if classified is not None:
                app.logger.debug('predictions exist')

                classified = classified.decode("utf-8")
                data["predictions"] = json.loads(classified)

                db.delete(k)
                app.logger.debug('image is removed from the queue')

                break

            time.sleep(CLIENT_SLEEP)

        data["success"] = True
        app.logger.debug('successfully predicted')

        return flask.jsonify(data)


if __name__ == "__main__":
    print("* Setting up logger...")
    logging.config.dictConfig(yaml.load(open('logging/logging.conf')))
    logfile = logging.getLogger('file')
    logconsole = logging.getLogger('console')
    logfile.debug("Debug FILE")
    logconsole.debug("Debug CONSOLE")

    print("* Starting model service...")
    t = Thread(target=classify_process, args=())
    t.daemon = True
    t.start()

    print("* Starting web service...")
    app.run()