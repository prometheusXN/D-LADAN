import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras import activations
import tensorflow.keras as keras
import tensorflow.keras.regularizers as regularizers
import tensorflow.keras.backend as K


class GraphRecorder(Layer):
    def __init__(self, node_num, feature_dim, trainable=False, **kwargs):
        self.trainable = trainable
        self.node_num =node_num
        self.feature_dim = feature_dim
        super(GraphRecorder, self).__init__(trainable=self.trainable, **kwargs)

    def build(self, input_shape):
        self.node_features = self.add_weight(shape=(self.node_num, self.feature_dim),
                                             initializer='zeros',
                                             trainable=self.trainable,
                                             name='PosteriorLawAdjFeatures',
                                             dtype=tf.float32)
        super(GraphRecorder, self).build(input_shape)

    def call(self, inputs, **kwargs):
        feature_norm = K.l2_normalize(self.node_features, axis=-1)
        cosine_matrix = feature_norm @ tf.transpose(feature_norm)
        self_loop = tf.eye(num_rows=self.node_num, dtype=tf.float32)
        adj_matrix = cosine_matrix - self_loop
        return self.node_features * 1.0, adj_matrix

    def compute_output_shape(self, input_shape):
        return [(self.node_num, self.feature_dim), (self.node_num, self.node_num)]


class MemoryMomentum(Layer):
    def __init__(self, node_num, feature_dim, trainable=False, **kwargs):
        self.trainable = trainable
        self.node_num =node_num
        self.feature_dim = feature_dim
        super(MemoryMomentum, self).__init__(trainable=self.trainable, **kwargs)

    def build(self, input_shape):
        self.memory = self.add_weight(shape=(self.node_num, self.feature_dim),
                                      initializer='zeros', trainable=self.trainable,
                                      name='PosteriorMemory', dtype=tf.float32)
        super(MemoryMomentum, self).build(input_shape)

    def momentum_update(self, proto, keep_coefficient=1.0,
                        synchronize_memory=False, warming_up=True, momentum_flag=False):
        if synchronize_memory:
            self.memory.assign(proto)
        elif warming_up:
            pass
        elif momentum_flag:
            memory_updated = (1 - keep_coefficient) * proto + keep_coefficient * self.memory
            self.memory.assign(memory_updated)
        else:
            pass

    def call(self, inputs, classifier=None, keep_coefficient=1.0, training=False,
             synchronize_memory=False, warming_up=True, momentum_flag=False, **kwargs):
        if classifier is not None:
            _, proto = classifier(self.memory)
        else:
            proto = inputs
        if training:
            self.momentum_update(proto, keep_coefficient=keep_coefficient, warming_up=warming_up,
                                 synchronize_memory=synchronize_memory, momentum_flag=momentum_flag)
        return self.memory * 1.0

    def compute_output_shape(self, input_shape):
        return (self.node_num, self.feature_dim)
