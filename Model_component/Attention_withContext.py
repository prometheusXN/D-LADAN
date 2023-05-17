import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras import activations


class AttentionWithContext(Layer):
    def __init__(self, context_size, trainable=True, **kwargs):
        self.trainable = trainable
        self.context_size = context_size
        self.epsilon = 1e-32
        super(AttentionWithContext, self).__init__(trainable=self.trainable, **kwargs)

    def build(self, input_shape):
        self.key_encoder = Dense(units=self.context_size, name='key_encoder', trainable=self.trainable)
        self.context = self.add_weight(shape=(self.context_size, ), name='u_context', trainable=self.trainable)
        super(AttentionWithContext, self).build(input_shape)

    def get_safe_shift(self, logits, mask):
        """
        :param logits: A tf.Tensor of shape [B, TQ, TK] of dtype tf.float32
        :param mask: A tf.Tensor of shape [B, TQ, TK] of dtype tf.float32
        where TQ, TK are the maximum lengths of the queries resp. the keys in the batch
        """

        # Determine minimum
        K_shape = logits.get_shape().as_list()
        mask_shape = mask.get_shape().as_list()
        if mask_shape != K_shape:
            mask = tf.tile(mask, [1] + [K_shape[1] // mask_shape[1]] + [1] * (len(K_shape) - 2))

        logits_min = tf.reduce_min(logits, axis=-1, keepdims=True)  # [B, TQ, 1]
        logits_min = tf.tile(logits_min, multiples=[1] * (len(K_shape) - 1) + [K_shape[-1]])  # [B, TQ, TK]

        logits = tf.where(condition=mask > .5, x=logits, y=logits_min)

        # Determine maximum
        logits_max = tf.reduce_max(logits, axis=-1, keepdims=True, name="logits_max")  # [B, TQ, 1]
        logits_shifted = tf.subtract(logits, logits_max, name="logits_shifted")  # [B, TQ, TK]

        return logits_shifted

    def padding_aware_softmax(self, logits, key_mask, query_mask=None):

        logits_shifted = self.get_safe_shift(logits, key_mask)

        # Apply exponential
        weights_unscaled = tf.exp(logits_shifted)

        # Apply mask
        weights_unscaled = tf.multiply(key_mask, weights_unscaled)  # [B, TQ, TK]

        # Derive total mass
        weights_total_mass = tf.reduce_sum(weights_unscaled, axis=-1, keepdims=True)  # [B, TQ, 1]

        # Avoid division by zero
        if query_mask:
            weights_total_mass = tf.where(condition=tf.equal(query_mask, 1),
                                          x=weights_total_mass,
                                          y=tf.ones_like(weights_total_mass))

        # Normalize weights
        weights = tf.divide(weights_unscaled, weights_total_mass + self.epsilon)  # [B, TQ, TK]

        return weights

    def call(self, inputs: tf.Tensor, mask=None, dropout=None, **kwargs):
        input_shape = inputs.get_shape().as_list()
        key = self.key_encoder(inputs)
        Values = inputs

        atten_scores = tf.reduce_sum(key * self.context, -1)/ tf.sqrt(tf.cast(input_shape[-1],tf.float32))
        if mask is not None:
            scores = self.padding_aware_softmax(atten_scores, mask)
            output = tf.reduce_sum(tf.expand_dims(scores, -1) * Values, -2)
        else:
            scores = tf.nn.softmax(atten_scores, -1)
            output = tf.reduce_sum(tf.expand_dims(scores, -1) * Values, -2)

        if dropout is not None:
            output = tf.nn.dropout(output, rate=dropout)
        return output, scores  # size: [x, z]

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[:-2] + [input_shape[-1]]
        output_shape = tuple(output_shape)
        return output_shape


class AttentionWithExtraContext(Layer):
    def __init__(self, context_size, trainable=True, activation='tanh', **kwargs):
        self.trainable = trainable
        self.context_size = context_size
        self.epsilon = 1e-32
        self.activation = activations.get(activation)
        super(AttentionWithExtraContext, self).__init__(trainable=self.trainable, **kwargs)

    def build(self, input_shape):
        self.key_encoder = Dense(units=self.context_size, name='key_encoder', trainable=self.trainable)
        super(AttentionWithExtraContext, self).build(input_shape)

    def get_safe_shift(self, logits, mask):
        """
        :param logits: A tf.Tensor of shape [B, TQ, TK] of dtype tf.float32
        :param mask: A tf.Tensor of shape [B, TQ, TK] of dtype tf.float32
        where TQ, TK are the maximum lengths of the queries resp. the keys in the batch
        """

        # Determine minimum
        K_shape = logits.get_shape().as_list()
        mask_shape = mask.get_shape().as_list()
        if mask_shape != K_shape:
            mask = tf.tile(mask, [1] + [K_shape[1] // mask_shape[1]] + [1] * (len(K_shape) - 2))

        logits_min = tf.reduce_min(logits, axis=-1, keepdims=True)  # [B, TQ, 1]
        logits_min = tf.tile(logits_min, multiples=[1] * (len(K_shape) - 1) + [K_shape[-1]])  # [B, TQ, TK]

        logits = tf.where(condition=mask > .5, x=logits, y=logits_min)

        # Determine maximum
        logits_max = tf.reduce_max(logits, axis=-1, keepdims=True, name="logits_max")  # [B, TQ, 1]
        logits_shifted = tf.subtract(logits, logits_max, name="logits_shifted")  # [B, TQ, TK]

        return logits_shifted

    def padding_aware_softmax(self, logits, key_mask, query_mask=None):

        logits_shifted = self.get_safe_shift(logits, key_mask)

        # Apply exponential
        weights_unscaled = tf.exp(logits_shifted)

        # Apply mask
        weights_unscaled = tf.multiply(key_mask, weights_unscaled)  # [B, TQ, TK]

        # Derive total mass
        weights_total_mass = tf.reduce_sum(weights_unscaled, axis=-1, keepdims=True)  # [B, TQ, 1]

        # Avoid division by zero
        if query_mask:
            weights_total_mass = tf.where(condition=tf.equal(query_mask, 1),
                                          x=weights_total_mass,
                                          y=tf.ones_like(weights_total_mass))

        # Normalize weights
        weights = tf.divide(weights_unscaled, weights_total_mass + self.epsilon)  # [B, TQ, TK]

        return weights

    def call(self, inputs: [tf.Tensor, tf.Tensor], mask=None, Value_weight=None, dropout=None, **kwargs):
        feature_input, context_vector = inputs
        input_shape = feature_input.get_shape().as_list()

        key = self.key_encoder(feature_input)

        if Value_weight is not None:
            Values = feature_input @ Value_weight
            # Values = self.activation(Values)
        else:
            Values = feature_input

        atten_scores = tf.reduce_sum(key * context_vector, -1)/ tf.sqrt(tf.cast(input_shape[-1],tf.float32))
        if mask is not None:
            scores = self.padding_aware_softmax(atten_scores, mask)
        else:
            scores = tf.nn.softmax(atten_scores, -1)

        output = tf.reduce_sum(tf.expand_dims(scores, -1) * Values, -2)
        if dropout is not None:
            output = tf.nn.dropout(output, rate=dropout)
        return output, scores  # size: [x, z]

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[:-2] + [input_shape[-1]]
        output_shape = tuple(output_shape)
        return output_shape
