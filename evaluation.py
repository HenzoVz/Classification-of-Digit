from keras.models import load_model
from PIL import Image
import tensorflow as tf
import numpy as np
import json

graph = tf.get_default_graph()
model = load_model('./classifier_digit.h5')

im = Image.open("7949.png")
im2arr = np.array(im).reshape((1, 28, 28, 1))
with graph.as_default():
     a = np.argmax(model.predict(im2arr))

