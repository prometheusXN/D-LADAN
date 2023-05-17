import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense
import tensorflow.keras as keras
from Model_component.Attention_withContext import AttentionWithContext, AttentionWithExtraContext
from Model_component.GraphDistillOperator import GraphDistillOperator
import tensorflow.keras.backend as K
from utils.GumbelSoftmax import gumbel_softmax


class Ladan(Model):
    def __init__(self, config, group_num, trainable=True, **kwargs):
        super(Ladan, self).__init__(**kwargs)

        self.config = config
        self.group_num = group_num
        self.han_size = self.config.getint('net', 'han_size')
        self.num_distill_layers = self.config.getint('net', 'num_distill_layers')

        self.word_encoder = keras.layers.Bidirectional(keras.layers.GRU(self.han_size, return_sequences=True, trainable=trainable), merge_mode='concat', trainable=trainable, name='word_BiGRU')
        self.word_attention = AttentionWithContext(context_size=self.han_size * 2, trainable=trainable, name='word_attention')
        self.sentence_encoder = keras.layers.Bidirectional(keras.layers.GRU(self.han_size, return_sequences=True, trainable=trainable), merge_mode='concat', trainable=trainable, name='sentence_BIGRU')
        self.sentence_attention = AttentionWithContext(context_size=self.han_size * 2, trainable=trainable, name='sentence_attention')

        self.graph_distillers = []
        for i in range(self.num_distill_layers):
            distill_layer = GraphDistillOperator(out_dim=self.han_size*2, trainable=trainable, activation='tanh', name='distill_layer_'+str(i))
            self.graph_distillers.append(distill_layer)

        self.group_chosen = Dense(self.group_num, trainable=trainable, name='group_chosen')

        self.context_generation_w = Dense(self.han_size * 2, trainable=trainable,
                                          name='context_generation_w', activation='tanh')
        self.context_generation_s = Dense(self.han_size * 2, trainable=trainable,
                                          name='context_generation_s', activation='tanh')

        self.word_reattention = AttentionWithExtraContext(context_size=self.han_size * 2, trainable=trainable, name='word_reattention')
        self.sentence_re_encoder = Sequential([keras.layers.Bidirectional(keras.layers.GRU(self.han_size, return_sequences=True, trainable=trainable), merge_mode='concat', trainable=trainable)], name='re_sentence_BIGRU')
        self.sentence_reattention = AttentionWithExtraContext(context_size=self.han_size * 2, trainable=trainable, name='sentence_reattention')

    def run_model(self, inputs, mask, model:Model):
        input_shape = inputs.get_shape().as_list()
        mask = tf.expand_dims(mask, -1)
        mask_shape = mask.get_shape().as_list()
        inputs = tf.reshape(inputs, [-1] + input_shape[-2:])
        mask = tf.cast(tf.reshape(mask, [-1] + mask_shape[-2:]), dtype=tf.bool)
        output = model.call(inputs, mask=mask)
        # output = model(inputs)
        rep = tf.reshape(output, shape=[-1] + input_shape[1:-1] + [self.han_size *2])
        return rep

    def distill_polling(self, features, group_index):
        feature_shape = features.get_shape().as_list()
        features_grouped = tf.dynamic_partition(features, group_index, num_partitions=self.group_num)

        group_contexts = []
        for i in range(self.group_num):
            u = tf.reduce_max(features_grouped[i], 0)  # law_representation[i]: [n, law_size]
            u_2 = tf.reduce_min(features_grouped[i], 0)
            group_contexts.append(tf.concat([u, u_2], -1))
            # size: [graph_num, 2*law_size] whether this u can use attention to get
        group_contexts = tf.reshape(tf.concat(group_contexts, 0), [-1, 2 * feature_shape[-1]])
        return group_contexts

    def aggregate_polling(self, features, group_index):
        feature_shape = features.get_shape().as_list()
        features_grouped = tf.dynamic_partition(features, group_index, num_partitions=self.group_num)

        group_contexts = []
        for i in range(self.group_num):
            u = tf.reduce_mean(features_grouped[i], 0)  # law_representation[i]: [n, law_size]
            group_contexts.append(u)
            # size: [graph_num, 2*law_size] whether this u can use attention to get
        group_contexts = tf.reshape(tf.concat(group_contexts, 0), [self.group_num, feature_shape[-1]])
        return group_contexts

    def call(self, inputs, law_information=None, training=None, dropout=None, word_mask=None, sentence_mask=None, word_mask_law=None, sentence_mask_law=None):
        fact_inputs = inputs
        law_inputs, adj_matrix_law, group_indexes = law_information

        with tf.name_scope('law_encoding_base'):
            law_word_level = self.run_model(law_inputs, word_mask_law, self.word_encoder)
            law_rep_sentences, _ = self.word_attention(law_word_level, mask=word_mask_law)
            law_sentence_level = self.run_model(law_rep_sentences, sentence_mask_law, self.sentence_encoder)
            law_base, _ = self.sentence_attention(law_sentence_level, mask=sentence_mask_law)
            if dropout is not None:
                law_base = tf.nn.dropout(law_base, rate=dropout)

        with tf.name_scope('ComputeGroupBasedContext'):
            with tf.name_scope('GenerateGroupInformation'):
                distilled_law = law_base
                for i in range(self.num_distill_layers):
                    distilled_law, _ = self.graph_distillers[i]([distilled_law, adj_matrix_law])

                context_list = self.distill_polling(features=distilled_law, group_index=group_indexes)

            with tf.name_scope('ComputeContextForLaw'):
                context_law = tf.gather(context_list, group_indexes)
                context_word_l = tf.reshape(self.context_generation_w(context_law), shape=(-1, 1, 1, self.han_size * 2))
                context_sentence_l = tf.reshape(self.context_generation_s(context_law), shape=(-1, 1, self.han_size * 2))

        with tf.name_scope('Re-encodeLaw'):
            re_law_rep_sentences, _ = self.word_reattention([law_word_level, context_word_l], mask=word_mask_law)
            re_law_sentence_level = self.run_model(re_law_rep_sentences, sentence_mask_law, self.sentence_re_encoder)
            law_distill, _ = self.sentence_reattention([re_law_sentence_level, context_sentence_l], mask=sentence_mask_law)
            if dropout is not None:
                law_distill = tf.nn.dropout(law_distill, rate=dropout)

        with tf.name_scope('EncodeFact'):
            with tf.name_scope('BaseEncodingFact'):
                fact_word_level = self.run_model(fact_inputs, word_mask, self.word_encoder)
                fact_rep_sentences, _ = self.word_attention(fact_word_level, mask=word_mask)
                fact_sentence_level = self.run_model(fact_rep_sentences, sentence_mask, self.sentence_encoder)
                fact_base, _ = self.sentence_attention(fact_sentence_level, mask=sentence_mask)
                if dropout is not None:
                    fact_base = tf.nn.dropout(fact_base, rate=dropout)

            with tf.name_scope('ComputeContextForFact'):
                group_pred = self.group_chosen(fact_base)

                group_pred = K.softmax(group_pred, -1)
                group_choose = tf.one_hot(tf.argmax(group_pred, axis=-1), self.group_num) # import Gumbol-softmax
                re_context = group_choose @ context_list

                context_word = tf.reshape(self.context_generation_w(re_context), shape=(-1, 1, 1, self.han_size * 2))
                context_sentence = tf.reshape(self.context_generation_s(re_context), shape=(-1, 1, self.han_size * 2))

            with tf.name_scope('DistillEncodingFact'):
                re_fact_rep_sentences, _ = self.word_reattention([fact_word_level, context_word], mask=word_mask)
                re_fact_sentence_level = self.run_model(re_fact_rep_sentences, sentence_mask, self.sentence_re_encoder)
                fact_distill, _ = self.sentence_reattention([re_fact_sentence_level, context_sentence], mask=sentence_mask)
                if dropout is not None:
                    fact_distill = tf.nn.dropout(fact_distill, rate=dropout)

        fact_rep = tf.concat([fact_base, fact_distill], axis=-1)
        law_rep = tf.concat([law_base, law_distill], axis=-1)

        return fact_rep, law_rep, group_pred

    def compute_output_shape(self, input_shape):
        fact_inputs_shape, law_inputs_shape, _, _ = input_shape
        return [(fact_inputs_shape[0], self.han_size*4), (law_inputs_shape[0], self.han_size*4), (fact_inputs_shape[0], self.group_num)]

