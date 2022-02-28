import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input, Dense, Conv1D
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.losses import MeanSquaredError as mse

class Segmentation():

    def __init__(self):
        self.model = None

    def new_model(self, project_name = "segmentation", tam = 512):
        input   = Input(shape = (tam, 1))
        x       = Conv1D(filters=32, kernel_size=13, padding="same", activation = "sigmoid", name = "kernel13")(input)
        x       = tf.keras.layers.MaxPooling1D(32, data_format = "channels_first")(x)
        x       = tf.keras.layers.Flatten()(x)
        outputs = Dense(tam, activation = 'sigmoid', name = "sigmoid")(x)
        self.model   = tf.keras.Model(inputs=input, outputs=outputs, name=project_name)

        opt = optimizers.Adam(learning_rate = 0.001, beta_1=0.9, beta_2 = 0.999, epsilon=1.e-7)

        loss = tf.keras.losses.BinaryCrossentropy()
        metric = tf.keras.metrics.BinaryAccuracy()
        self.model.compile(loss = loss, optimizer = opt, metrics = [metric])