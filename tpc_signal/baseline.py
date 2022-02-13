import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input, Dense, Conv1D
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.losses import MeanSquaredError as mse

class Baseline():
    def __init__(self):
        self.model = None

    def new_model(self, project_name = "baseline", tam = 512):
        self.model = Sequential(name = project_name)
        self.model.add(Input(shape = (tam, 1)))
        self.model.add(Conv1D(filters = 32, kernel_size = 19, padding = "valid", activation = None, name = "kernel19", use_bias = False))
        self.model.add(Conv1D(filters = 32, kernel_size = 17, padding = "valid", activation = None, name = "kernel17", use_bias = False))
        self.model.add(tf.keras.layers.MaxPooling1D(pool_size = 16, name = "MaxPool1D_1", data_format = "channels_first"))
        # model.add(Conv1D(filters = 32, kernel_size = 19, padding = "valid", activation = None, name = "kernel19_2", use_bias = False))
        # model.add(Conv1D(filters = 32, kernel_size = 17, padding = "valid", activation = None, name = "kernel17_2", use_bias = False))
        # model.add(tf.keras.layers.MaxPooling1D(pool_size = 32, name = "MaxPool1D_2", data_format = "channels_first"))
        self.model.add(tf.keras.layers.Flatten(name = "flatten"))
        self.model.add(Dense(tam, activation = 'relu', name = "relu"))
        # model.add(Dense(tam, activation = 'linear', name = "linear"))

        # opt = optimizers.Adam(learning_rate = 0.0005, beta_1=0.9, beta_2 = 0.999, clipnorm = 0.45, epsilon=1.e-7)
        opt = optimizers.Adam(learning_rate = 0.0005, beta_1=0.9, beta_2 = 0.999, clipnorm = 0.45, epsilon=1.e-7)
        # opt = optimizers.SGD(learning_rate = 0.001, momentum=0.0, nesterov=False, name='SGD', clipnorm = 0.001)
        # opt = optimizers.Adamax(learning_rate = 0.001, beta_1=0.9, beta_2=0.999, name='Adamax', clipnorm = 0.001)

        loss = mse()
        # loss = mae()
        # metric1 = tf.keras.metrics.CategoricalAccuracy()
        metric2 = MeanAbsoluteError()
        self.model.compile(loss = loss, optimizer = opt, metrics = [metric2])
        return self.model