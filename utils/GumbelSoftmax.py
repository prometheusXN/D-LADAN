import tensorflow.keras.backend as K
import tensorflow as tf


def sample_gumbel(shape, eps=1e-20):
    U = tf.random.uniform(shape)
    return -tf.math.log(-tf.math.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    shape = tf.shape(logits)
    y = logits + sample_gumbel(shape=shape)
    return K.softmax(axis=-1, x=(y/temperature))


def gumbel_softmax(logits, temperature=1.0, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    # shape = y.get_shape()
    # ind = tf.argmax(y, dim=-1)
    # y_hard = tf.reshape(tf.zeros_like(y), (-1, shape[-1]))
    # y_hard.scatter_(1, tf.reshape(ind, (-1, 1)), 1)
    # y_hard = y_hard.view(*shape)
    # # Set gradients w.r.t. y_hard gradients w.r.t. y
    # y_hard = (y_hard - y).detach() + y
    return y


def softmax_with_mask(logits, masks=None, axis=-1):
    if masks is not None:
        logits = logits + (1-masks) * (-1e32)
    score = K.softmax(logits, axis=axis)
    return score

