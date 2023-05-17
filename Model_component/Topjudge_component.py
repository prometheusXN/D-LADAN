import tensorflow as tf
from tensorflow.keras.models import Model
import tensorflow.keras as keras
from Model_component.CNN_cell import CNNEncoder
from Model_component.LSTM_cell import LSTMDecoder


class Topjudge(Model):
    def __init__(self, config, law_num, accu_num, trainable=True):
        self.trainable = trainable
        super(Topjudge, self).__init__()
        self.config = config
        self.encoder = CNNEncoder(self.config, trainable=self.trainable)
        self.decoder = LSTMDecoder(self.config, law_num, accu_num, trainable=self.trainable)
        self.trans_linear = keras.layers.Dense(self.decoder.feature_len, name="trans_fc", trainable=self.trainable)
        self.num1 = law_num
        self.num2 = accu_num

    def call(self, inputs, mask=None, dropout=None, **kwargs):
        x = self.encoder(inputs)
        if self.encoder.feature_len != self.decoder.feature_len:
            x = self.trans_linear(x)
        if dropout is not None:
            x = tf.nn.dropout(x, rate=dropout)

        x = self.decoder(x)

        return x

    def compute_output_shape(self, input_shape):

        return [(input_shape[0], self.num1), (input_shape[0], self.num2), (input_shape[0], 12)]

