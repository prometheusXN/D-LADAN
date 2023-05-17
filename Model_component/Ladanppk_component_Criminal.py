import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
import tensorflow.keras as keras
from Model_component.Attention_withContext import AttentionWithContext, AttentionWithExtraContext
from Model_component.GraphDistillOperator import GraphDistillOperator, GraphDistillOperatorWithEdgeWeight
from Model_component.TransformerLayer import TransformerFeatureWithLabel
import tensorflow.keras.backend as K
from utils.GumbelSoftmax import gumbel_softmax
import numpy as np


class LadanPPK_Criminal(Model):
    def __init__(self, config, group_num, trainable=True, **kwargs):
        super(LadanPPK_Criminal, self).__init__(**kwargs)

        self.config = config
        self.group_num = group_num
        self.han_size = self.config.getint('net', 'han_size')
        self.num_distill_layers = self.config.getint('net', 'num_distill_layers')

        with tf.name_scope('define_encoder_base'):
            self.base_encoder_0 = keras.layers.Bidirectional(
                keras.layers.GRU(self.han_size, return_sequences=True, trainable=trainable), merge_mode='concat',
                trainable=trainable, name='base_BiGRU_0')
            self.base_encoder_1 = keras.layers.Bidirectional(
                keras.layers.GRU(self.han_size, return_sequences=True, trainable=trainable), merge_mode='concat',
                trainable=trainable, name='base_BiGRU_1')
            self.base_attention = AttentionWithContext(context_size=self.han_size * 2, trainable=trainable,
                                                       name='base_attention')

        with tf.name_scope('define_distiller_prior'):
            self.graph_distillers_prior = []
            for i in range(self.num_distill_layers):
                distill_layer = GraphDistillOperator(out_dim=self.han_size*2, trainable=trainable,
                                                     activation='tanh', name='distiller_prior_'+str(i))
                self.graph_distillers_prior.append(distill_layer)

        with tf.name_scope('define_distiller_posterior'):
            self.graph_distillers_posterior = []
            for i in range(self.num_distill_layers):
                distill_layer = GraphDistillOperatorWithEdgeWeight(out_dim=self.han_size*2, trainable=trainable,
                                                                   withAgg=False, activation='tanh',
                                                                   name='distiller_posterior_'+str(i))  # with edge weight
                self.graph_distillers_posterior.append(distill_layer)

        with tf.name_scope('context_generator_prior'):
            self.group_chosen_hidden = Dense(256, trainable=trainable, name='group_chosen_hidden',
                                             activation='tanh')
            self.group_chosen = Dense(self.group_num, trainable=trainable, name='group_chosen')
            self.context_prior = Dense(self.han_size * 2, trainable=trainable,
                                       name='context_prior', activation='tanh')

        with tf.name_scope('define_encoder_prior'):
            self.prior_encoder_0 = keras.layers.Bidirectional(
                keras.layers.GRU(self.han_size, return_sequences=True, trainable=trainable),merge_mode='concat',
                trainable=trainable, name='prior_BiGRU_0')
            self.prior_encoder_1 = keras.layers.Bidirectional(
                keras.layers.GRU(self.han_size, return_sequences=True, trainable=trainable), merge_mode='concat',
                trainable=trainable, name='prior_BiGRU_1')
            self.prior_attention = AttentionWithExtraContext(context_size=self.han_size * 2, trainable=trainable,
                                                             name='prior_attention')

        with tf.name_scope('define_encoder_posterior'):
            self.posterior_encoder_0 = keras.layers.Bidirectional(
                keras.layers.GRU(self.han_size, return_sequences=True, trainable=trainable), merge_mode='concat',
                trainable=trainable, name='posterior_BiGRU_0')
            self.posterior_encoder_1 = keras.layers.Bidirectional(
                keras.layers.GRU(self.han_size, return_sequences=True, trainable=trainable), merge_mode='concat',
                trainable=trainable, name='posterior_BiGRU_1')
            self.posterior_attention = AttentionWithExtraContext(context_size=self.han_size * 2, trainable=trainable,
                                                                 name='posterior_attention')

        with tf.name_scope('context_generator_posterior'):
            self.Transformer_posterior = TransformerFeatureWithLabel(head_num=4, trainable=trainable,
                                                                 name='Transformer_posterior')
            self.matching_law_posterior = Dense(self.han_size * 2, trainable=trainable,
                                                name='matching_accu_posterior', activation='tanh')
            self.matching_fact_posterior = Dense(self.han_size * 2, trainable=trainable,
                                                 name='matching_fact_posterior', activation='tanh')

            self.context_posterior = Dense(self.han_size * 2, trainable=trainable,
                                           name='context_posterior', activation='tanh')

        self.posterior_mask = tf.Variable(initial_value=np.zeros([1, self.han_size * 2]), dtype=tf.float32,
                                          trainable=False, name='posterior_mask')
        self.posterior_maskF = tf.Variable(initial_value=np.zeros([1, self.han_size * 2]), dtype=tf.float32,
                                           trainable=False, name='posterior_maskF')

    def run_model(self, inputs, mask, model:Model):
        input_shape = inputs.get_shape().as_list()
        mask = tf.expand_dims(mask, -1)
        mask_shape = mask.get_shape().as_list()
        inputs = tf.reshape(inputs, [-1] + input_shape[-2:])
        mask = tf.cast(tf.reshape(mask, [-1] + mask_shape[-2:]), dtype=tf.bool)
        output = model(inputs, mask=mask)
        rep = tf.reshape(output, shape=[-1] + input_shape[1:-1] + [self.han_size * 2])
        return rep

    def distill_polling(self, features, group_index, dropout=None):
        feature_shape = features.get_shape().as_list()
        features_grouped = tf.dynamic_partition(features, group_index, num_partitions=self.group_num)

        group_contexts = []
        for i in range(self.group_num):
            u = tf.reduce_max(features_grouped[i], 0)  # law_representation[i]: [n, law_size]
            u_2 = tf.reduce_min(features_grouped[i], 0)
            group_contexts.append(tf.concat([u, u_2], -1))
            # size: [graph_num, 2*law_size] whether this u can use attention to get
        group_contexts = tf.reshape(tf.concat(group_contexts, 0), [-1, 2 * feature_shape[-1]])
        if dropout is not None:
            group_contexts = tf.nn.dropout(group_contexts, rate=dropout)
        return group_contexts

    def aggregate_polling(self, features, group_index, dropout=None):
        feature_shape = features.get_shape().as_list()
        features_grouped = tf.dynamic_partition(features, group_index, num_partitions=self.group_num)

        group_contexts = []
        for i in range(self.group_num):
            u = tf.reduce_mean(features_grouped[i], 0)  # law_representation[i]: [n, law_size]
            group_contexts.append(u)
            # size: [graph_num, 2*law_size] whether this u can use attention to get
        group_contexts = tf.reshape(tf.concat(group_contexts, 0), [self.group_num, feature_shape[-1]])
        if dropout is not None:
            group_contexts = tf.nn.dropout(group_contexts, rate=dropout)
        return group_contexts

    def re_encoding_fact(self, name, inputs, dense_funcs, context_funcs, encoder_funcs, masks, dropout=None):

        fact_base, key_list, context_list, fact_word_level, fact_hiddens = inputs
        law_Dense, fact_Dense = dense_funcs
        group_chosen, Transformer, context_generation = context_funcs
        reattention = encoder_funcs
        mask_hidden, mask_words = masks

        with tf.name_scope(name + 'ContextForFact'):
            with tf.name_scope(name + '_context_choose'):
                with tf.name_scope(name + '_Transformer_input'): # the context chosen function and the context source need to be rewrite
                    if name=="Posterior":
                        matching_accu_prior = law_Dense(key_list)
                        macthing_fact_prior = fact_Dense(fact_word_level)
                        label_mask = tf.ones([tf.shape(macthing_fact_prior)[0], tf.shape(key_list)[0]], dtype=tf.float32)
                        CLS_mask = tf.ones([tf.shape(macthing_fact_prior)[0], 1], dtype=tf.float32)
                        matching_mask = tf.concat([CLS_mask, label_mask, mask_words], axis=-1)
                        matching_mask = tf.reshape(matching_mask,
                                                   shape=[-1, macthing_fact_prior.get_shape()[1] + key_list.get_shape()[0] + 1])
                        CLS_input = tf.ones(shape=[tf.shape(macthing_fact_prior)[0], 1, macthing_fact_prior.get_shape()[-1]],
                                            dtype=tf.float32)
                        label_input = tf.tile(tf.expand_dims(matching_accu_prior, axis=0),
                                              [tf.shape(macthing_fact_prior)[0], 1, 1])

                        CLS_output, label_output, fact_output = Transformer([CLS_input, label_input, macthing_fact_prior],
                                                                        mask=matching_mask)
                        fact_key = tf.reduce_sum(fact_output, axis=1) / tf.reduce_sum(mask_words, axis=-1, keepdims=True)
                        fact_key = K.l2_normalize(fact_key, axis=-1)
                        label_output = K.l2_normalize(label_output, axis=-1)
                        group_pred = tf.reduce_sum(fact_key @ tf.transpose(label_output, [0, 2, 1]), axis=1) * 10.0
                        group_pred = K.softmax(group_pred)
                        re_context = group_pred @ context_list
                    else:
                        group_pred = group_chosen(self.group_chosen_hidden(fact_base))
                        group_pred = K.softmax(group_pred)
                        re_context = group_pred @ context_list

                context = tf.reshape(context_generation(re_context), shape=(-1, 1, self.han_size * 2))

        with tf.name_scope('Encoding'):
            fact_prior, _ = reattention([fact_hiddens, context], mask=mask_hidden, dropout=dropout)
        return fact_prior, group_pred

    def re_encodering_law(self, inputs, group_indexes, masks, context_list, name, context_funcs, encoder_funcs, dropout=None):
        law_hiddens = inputs
        word_mask_law = masks
        context_generation = context_funcs
        reattention = encoder_funcs

        with tf.name_scope(name+'ContextForLaw'):
            context_law = tf.gather(context_list, group_indexes)
            # context_law = tf.gather(self.context, group_indexes)
            context_l = tf.reshape(context_generation(context_law), shape=(-1, 1, 1, self.han_size * 2))

        with tf.name_scope(name+'Encoding'):
            law_prior, _ = reattention([law_hiddens, context_l], mask=word_mask_law, dropout=dropout)

        return law_prior

    def call(self, inputs, law_information=None, training=None, warming_up=False, dropout=None,
             word_mask=None, word_mask_law=None):

        fact_inputs = inputs
        law_inputs, adj_matrix_law, group_indexes, accu_inputs_posterior, adj_matrix_posterior = law_information

        with tf.name_scope('law_encoding_base'):
            law_hidden_0 = self.run_model(law_inputs, word_mask_law, self.base_encoder_0)
            law_hidden_1 = self.run_model(law_hidden_0, word_mask_law, self.base_encoder_1)
            law_base, _ = self.base_attention(law_hidden_1, mask=word_mask_law, dropout=dropout)

        with tf.name_scope('GeneratePriorGroupInformation'):
            distilled_law_prior = law_base
            for i in range(self.num_distill_layers):
                distilled_law_prior, aggregate_law_prior = self.graph_distillers_prior[i]([distilled_law_prior, adj_matrix_law])
            context_list_prior = self.distill_polling(features=distilled_law_prior, group_index=group_indexes, dropout=dropout)
            key_list_prior = self.aggregate_polling(features=aggregate_law_prior, group_index=group_indexes, dropout=dropout)

        with tf.name_scope('GeneratePosteriorGroupInformation'):
            distilled_law_posterior, aggregate_law_posterior = accu_inputs_posterior, accu_inputs_posterior
            for i in range(self.num_distill_layers):
                distilled_law_posterior, aggregate_law_posterior = self.graph_distillers_posterior[i]([distilled_law_posterior, aggregate_law_posterior, adj_matrix_posterior])

            if warming_up:
                distilled_law_posterior *= self.posterior_mask
                aggregate_law_posterior *= self.posterior_mask

            key_list_posterior = aggregate_law_posterior
            context_list_posterior = distilled_law_posterior

        with tf.name_scope('EncodeFact'):
            with tf.name_scope('BaseEncodingFact'):
                fact_hidden_0 = self.run_model(fact_inputs, word_mask, self.base_encoder_0)
                fact_hidden_1 = self.run_model(fact_hidden_0, word_mask, self.base_encoder_1)
                fact_base, _ = self.base_attention(fact_hidden_1, mask=word_mask, dropout=dropout)

            with tf.name_scope('PriorEncodingFact'):
                fact_hidden_prior_0 = self.run_model(fact_inputs, word_mask, self.prior_encoder_0)
                fact_hidden_prior_1 = self.run_model(fact_hidden_prior_0, word_mask, self.prior_encoder_1)

                fact_prior, group_pred_prior = \
                    self.re_encoding_fact(inputs=[fact_base, key_list_prior, context_list_prior,  fact_hidden_1, fact_hidden_prior_1],
                                          dense_funcs=[None, None],
                                          context_funcs=[self.group_chosen, None, self.context_prior],
                                          encoder_funcs=self.prior_attention,
                                          masks=[word_mask, word_mask], dropout=dropout, name='Prior')

            with tf.name_scope('PosteriorEncodingFact_Law'):
                fact_sentence = tf.concat([fact_hidden_1, fact_hidden_prior_1], axis=-1)
                new_sentence_mask = word_mask
                fact_hidden_posterior_0 = self.run_model(fact_inputs, word_mask, self.posterior_encoder_0)
                fact_hidden_posterior_1 = self.run_model(fact_hidden_posterior_0, word_mask, self.posterior_encoder_1)
                fact_posterior, group_pred_posterior = \
                    self.re_encoding_fact(inputs=[None, accu_inputs_posterior, context_list_posterior, fact_sentence, fact_hidden_posterior_1],
                                          dense_funcs=[self.matching_law_posterior, self.matching_fact_posterior],
                                          context_funcs=[None, self.Transformer_posterior, self.context_posterior],
                                          encoder_funcs=self.posterior_attention,
                                          masks=[word_mask, new_sentence_mask], dropout=dropout, name='Posterior')

            if warming_up:
                fact_posterior *= self.posterior_maskF

        fact_rep = tf.concat([fact_base, fact_prior, fact_posterior], axis=-1)
        return fact_rep, group_pred_prior, group_pred_posterior
