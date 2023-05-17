from tensorflow.keras.layers import Layer
import tensorflow as tf
import tensorflow.keras.backend as K


class CosClassifier(Layer):
    def __init__(self, units, trainable=True, with_proto=False, **kwargs):
        self.output_dim = units
        self.trainable = trainable
        self.with_proto = with_proto
        super(CosClassifier, self).__init__(trainable=self.trainable, **kwargs)

    def build(self, input_shape):
        batch_dim, feature_dim = input_shape
        self.W = self.add_weight(name='W',
                                 shape=(self.output_dim, feature_dim),  # 假设输入tensor只有一个维度（不算batch的维度）
                                 initializer='uniform',
                                 trainable=self.trainable)  # 如果要定义可训练参数这里一定要选择True
        super(CosClassifier, self).build(input_shape)  # 这行代码一定要加上，super主要是调用MyLayer的父类（Layer）的build方法。

    def call(self, inputs, **kwargs):
        inputs_norm = K.l2_normalize(inputs, axis=-1)
        weight_norm = K.l2_normalize(self.W, axis=-1)
        score = inputs_norm @ tf.transpose(weight_norm) # 该层要实现的功能
        if self.with_proto:
            prototypes = tf.stop_gradient(self.W)
            return score, prototypes
        else:
            return score

    def compute_output_shape(self, input_shape):
        batch_dim, feature_dim = input_shape
        if self.with_proto:
            return [(input_shape[0], self.output_dim), (self.output_dim, feature_dim)]
        else:
            return (input_shape[0], self.output_dim)


class CosClassifierInter(Layer):
    def __init__(self, units, trainable=True, **kwargs):
        self.output_dim = units
        self.trainable = trainable
        super(CosClassifierInter, self).__init__(trainable=self.trainable, **kwargs)

    def build(self, input_shape):
        batch_dim, class_num, feature_dim = input_shape
        self.W = self.add_weight(name='W',
                                 shape=(self.output_dim, feature_dim),  # 假设输入tensor只有一个维度（不算batch的维度）
                                 initializer='uniform',
                                 trainable=self.trainable)  # 如果要定义可训练参数这里一定要选择True
        super(CosClassifierInter, self).build(input_shape)  # 这行代码一定要加上，super主要是调用MyLayer的父类（Layer）的build方法。

    def call(self, inputs, **kwargs):
        inputs_norm = K.l2_normalize(inputs, axis=-1)
        weight_norm = K.l2_normalize(self.W, axis=-1)
        score = tf.reduce_sum(inputs_norm * tf.expand_dims(weight_norm, axis=0), axis=-1) # 该层要实现的功能
        return score

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


class MultiplyScalar(Layer):
    def __init__(self, trainable=True, **kwargs):
        self.trainable = trainable
        super(MultiplyScalar, self).__init__(trainable=self.trainable, **kwargs)

    def build(self, input_shape):
        self.scalar = self.add_weight(name='Scalar',
                                 initializer='ones',
                                 trainable=self.trainable)  # 如果要定义可训练参数这里一定要选择True
        super(MultiplyScalar, self).build(input_shape)  # 这行代码一定要加上，super主要是调用MyLayer的父类（Layer）的build方法。

    def call(self, inputs, **kwargs):
        inputs = inputs
        return self.scalar * inputs * 10.0  # 该层要实现的功能

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        return output_shape


