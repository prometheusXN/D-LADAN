import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Flatten
import tensorflow.keras as keras
from Model_component.Attention_withContext import AttentionWithContext


class Han(Model):
    def __init__(self, config, trainable=True, **kwargs):
        super(Han, self).__init__(**kwargs)

        self.config = config
        self.han_size = self.config.getint('net', 'han_size')

        self.word_encoder = Sequential([keras.layers.Bidirectional(keras.layers.GRU(self.han_size, return_sequences=True), merge_mode='concat', trainable=trainable)], name='word_BiGRU')
        self.word_attention = AttentionWithContext(context_size=self.han_size * 2, trainable=trainable, name='word_attention')

        self.sentence_encoder = Sequential([keras.layers.Bidirectional(keras.layers.GRU(self.han_size, return_sequences=True), merge_mode='concat', trainable=trainable)], name='sentence_BIGRU')
        self.sentence_attention = AttentionWithContext(context_size=self.han_size * 2, trainable=trainable, name='sentence_attention')

    def run_model(self, inputs, mask, model:Model):
        input_shape = inputs.get_shape().as_list()
        mask = tf.expand_dims(mask, -1)
        mask_shape = mask.get_shape().as_list()
        inputs = tf.reshape(inputs, [-1] + input_shape[-2:])
        mask = tf.cast(tf.reshape(mask, [-1] + mask_shape[-2:]), dtype=tf.bool)
        # out = model(inputs, mask=mask)
        output = model(inputs)
        rep = tf.reshape(output, shape=[-1] + input_shape[1:-1] + [self.han_size *2])
        return rep

    def call(self, inputs, training=None, word_mask=None, sentence_mask=None):
        rep_word_level = self.run_model(inputs, word_mask, self.word_encoder)
        rep_sentences, _ = self.word_attention(rep_word_level, mask=word_mask)

        rep_sentence_level = self.run_model(rep_sentences, sentence_mask, self.sentence_encoder)
        rep_doc, _ = self.sentence_attention(rep_sentence_level, mask=sentence_mask)

        return rep_doc

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.han_size * 2)





