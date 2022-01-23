import matplotlib
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

IMAGE_SIZE=256
BATCH_SIZE=32

data = tf.keras.preprocessing.image_dataset_from_directory(
    "data", 
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE
)