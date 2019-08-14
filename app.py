from keras.models import load_model
from flask import Flask, request, json
from PIL import Image
import tensorflow as tf
import numpy as np

graph = tf.get_default_graph()

app = Flask(__name__)

model = load_model('./classifier_digit.h5')

@app.route('/predict_digit', methods=['POST'])
def predict_digit():
    im = Image.open(request.files['image'])
    im2arr = np.array(im).reshape((1, 28, 28, 1))
    with graph.as_default():
        return json.dumps(str(np.argmax(model.predict(im2arr))))

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=5000)
