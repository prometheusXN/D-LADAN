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


class LadanPPK(Model):
    def __init__(self, config, group_num, trainable=True, accu_relation=None, **kwargs):
        super(LadanPPK, self).__init__(**kwargs)

        self.config = config
        self.group_num = group_num
        self.han_size = self.config.getint('net', 'han_size')
        self.num_distill_layers = self.config.getint('net', 'num_distill_layers')
        self.accu_relation = accu_relation

        with tf.name_scope('define_encoder_base'):
            self.word_encoder = keras.layers.Bidirectional(keras.layers.GRU(self.han_size, return_sequences=True, trainable=trainable), merge_mode='concat', trainable=trainable, name='word_BiGRU')
            self.word_attention = AttentionWithContext(context_size=self.han_size * 2, trainable=trainable,
                                                       name='word_attention')
            self.sentence_encoder = keras.layers.Bidirectional(keras.layers.GRU(self.han_size, return_sequences=True, trainable=trainable), merge_mode='concat', trainable=trainable, name='sentence_BIGRU')
            self.sentence_attention = AttentionWithContext(context_size=self.han_size * 2, trainable=trainable,
                                                           name='sentence_attention')

        with tf.name_scope('define_distiller_prior'):
            self.graph_distillers_prior = []
            for i in range(self.num_distill_layers):
                distill_layer = GraphDistillOperator(out_dim=self.han_size*2, trainable=trainable,
                                                     activation='tanh', name='distiller_prior_'+str(i))
                self.graph_distillers_prior.append(distill_layer)

        with tf.name_scope('define_distiller_posterior'):
            self.graph_distillers_posterior = []
            for i in range(self.num_distill_layers):
                distill_layer = GraphDistillOperatorWithEdgeWeight(out_dim=self.han_size*2, trainable=trainable, withAgg=False,
                                                     activation='tanh', name='distiller_posterior_'+str(i))  # with edge weight
                self.graph_distillers_posterior.append(distill_layer)

        with tf.name_scope('context_generator_prior'):
            self.group_chosen_hidden = Dense(256, trainable=trainable, name='group_chosen_hidden', activation='tanh')
            self.group_chosen = Dense(self.group_num, trainable=trainable, name='group_chosen')
            self.context_w_prior = Dense(self.han_size * 2, trainable=trainable,
                                              name='context_w_prior', activation='tanh')
            self.context_s_prior = Dense(self.han_size * 2, trainable=trainable,
                                              name='context_s_prior', activation='tanh')

        with tf.name_scope('define_encoder_prior'):
            self.word_attention_prior = AttentionWithExtraContext(context_size=self.han_size * 2, trainable=trainable,
                                                              name='word_reattention')
            self.word_encoder_prior = keras.layers.Bidirectional(
                keras.layers.GRU(self.han_size, return_sequences=True, trainable=trainable), merge_mode='concat',
                trainable=trainable, name='word_encoder_prior')
            self.sentence_encoder_prior = keras.layers.Bidirectional(
                keras.layers.GRU(self.han_size, return_sequences=True, trainable=trainable), merge_mode='concat',
                trainable=trainable, name='sentence_encoder_prior')
            self.sentence_attention_prior = AttentionWithExtraContext(context_size=self.han_size * 2, trainable=trainable,
                                                              name='sentence_reattention')

        with tf.name_scope('define_encoder_posterior'):
            self.word_attention_posterior = AttentionWithExtraContext(context_size=self.han_size * 2, trainable=trainable,
                                                              name='word_attention_posterior')
            self.word_encoder_posterior = keras.layers.Bidirectional(
                keras.layers.GRU(self.han_size, return_sequences=True, trainable=trainable), merge_mode='concat',
                trainable=trainable, name='word_encoder_posterior')
            self.sentence_encoder_posterior = keras.layers.Bidirectional(
                keras.layers.GRU(self.han_size, return_sequences=True, trainable=trainable), merge_mode='concat',
                trainable=trainable, name='sentence_encoder_posterior')
            self.sentence_attention_posterior = AttentionWithExtraContext(context_size=self.han_size * 2, trainable=trainable,
                                                                  name='sentence_attention_posterior')

        with tf.name_scope('context_generator_posterior'):
            self.Transformer_posterior = TransformerFeatureWithLabel(head_num=4, trainable=trainable,
                                                                 name='Transformer_posterior')
            self.matching_law_posterior = Dense(self.han_size * 2, trainable=trainable,
                                            name='matching_law_posterior', activation='tanh')
            self.matching_fact_posterior = Dense(self.han_size * 2, trainable=trainable,
                                             name='matching_fact_posterior', activation='tanh')

            self.context_w_posterior = Dense(self.han_size * 2, trainable=trainable,
                                         name='context_w_posterior', activation='tanh')
            self.context_s_posterior = Dense(self.han_size * 2, trainable=trainable,
                                         name='context_s_posterior', activation='tanh')

        if self.accu_relation is not None:
            print('build posterior of charge')
            with tf.name_scope('define_distiller_posterior_accu'):
                self.graph_distillers_posterior_accu = []
                for i in range(self.num_distill_layers):
                    distill_layer = GraphDistillOperatorWithEdgeWeight(out_dim=self.han_size * 2, trainable=trainable,
                                                                       withAgg=False,
                                                                       activation='tanh',
                                                                       name='distiller_posterior_accu_' + str(i))
                    # with edge weight
                    self.graph_distillers_posterior_accu.append(distill_layer)

            with tf.name_scope('define_encoder_posterior_accu'):
                self.word_attention_posterior_A = AttentionWithExtraContext(context_size=self.han_size * 2,
                                                                          trainable=trainable,
                                                                          name='word_attention_posterior_accu')
                self.word_encoder_posterior_A = keras.layers.Bidirectional(
                    keras.layers.GRU(self.han_size, return_sequences=True, trainable=trainable), merge_mode='concat',
                    trainable=trainable, name='word_encoder_posterior_accu')
                self.sentence_encoder_posterior_A = keras.layers.Bidirectional(
                    keras.layers.GRU(self.han_size, return_sequences=True, trainable=trainable), merge_mode='concat',
                    trainable=trainable, name='sentence_encoder_posterior_accu')
                self.sentence_attention_posterior_A = AttentionWithExtraContext(context_size=self.han_size * 2,
                                                                              trainable=trainable,
                                                                              name='sentence_attention_posterior_accu')

            with tf.name_scope('context_generator_posterior_accu'):
                self.Transformer_posterior_A = TransformerFeatureWithLabel(head_num=4, trainable=trainable,
                                                                         name='Transformer_posterior_accu')
                self.matching_law_posterior_A = Dense(self.han_size * 2, trainable=trainable,
                                                    name='matching_law_posterior_accu', activation='tanh')
                self.matching_fact_posterior_A = Dense(self.han_size * 2, trainable=trainable,
                                                     name='matching_fact_posterior_accu', activation='tanh')

                self.context_w_posterior_A = Dense(self.han_size * 2, trainable=trainable,
                                                 name='context_w_posterior_accu', activation='tanh')
                self.context_s_posterior_A = Dense(self.han_size * 2, trainable=trainable,
                                                 name='context_s_posterior_accu', activation='tanh')

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
        # output = model(inputs)
        rep = tf.reshape(output, shape=[-1] + input_shape[1:-1] + [self.han_size * 2])
        return rep

    def distill_pooling(self, features, group_index, dropout=None):
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

    def distill_pooling_normal(self, features, adj_matrix, dropout=None):
        return

    def aggregate_pooling_normal(self, features, adj_matrix, dropout=None):
        group_contexts = (features + adj_matrix @ features) / (tf.reduce_sum(adj_matrix, axis=-1, keepdims=True)+1.0)
        if dropout is not None:
            group_contexts = tf.nn.dropout(group_contexts, rate=dropout)
        return group_contexts

    def aggregate_pooling(self, features, group_index, dropout=None):
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

        fact_base, key_list, context_list, fact_word_level, fact_sentence_level = inputs
        law_Dense, fact_Dense = dense_funcs
        group_chosen, Transformer, context_generation_w, context_generation_s = context_funcs
        word_reattention, sentence_re_encoder, sentence_reattention = encoder_funcs
        word_mask, sentence_mask, sentence_mask_1 = masks

        with tf.name_scope(name + 'ContextForFact'):
            with tf.name_scope(name + '_context_choose'):
                with tf.name_scope(name + '_Transformer_input'):
                    # the context chosen function and the context source need to be rewrite
                    if name == "Posterior":
                        matching_law_prior = law_Dense(key_list)    # part of source input
                        matching_fact_prior = fact_Dense(fact_sentence_level)
                        label_mask = tf.ones([tf.shape(fact_sentence_level)[0], tf.shape(key_list)[0]], dtype=tf.float32)
                        CLS_mask = tf.ones([tf.shape(fact_sentence_level)[0], 1], dtype=tf.float32)
                        matching_mask = tf.concat([CLS_mask, label_mask, sentence_mask_1], axis=-1)
                        matching_mask = tf.reshape(matching_mask,
                                                   shape=[-1, fact_sentence_level.get_shape()[1] + key_list.get_shape()[0] + 1])
                        CLS_input = tf.ones(shape=[tf.shape(fact_sentence_level)[0], 1, fact_sentence_level.get_shape()[-1]],
                                            dtype=tf.float32)
                        label_input = tf.tile(tf.expand_dims(matching_law_prior, axis=0),
                                              [tf.shape(fact_sentence_level)[0], 1, 1])

                        CLS_output, label_output, fact_output = Transformer([CLS_input, label_input, matching_fact_prior],
                                                                        mask=matching_mask)
                        fact_key = tf.reduce_sum(fact_output, axis=1) / tf.reduce_sum(sentence_mask_1, axis=-1, keepdims=True)
                        fact_key = K.l2_normalize(fact_key, axis=-1)
                        label_output = K.l2_normalize(label_output, axis=-1)
                        CLS_output = tf.reduce_sum(CLS_output, axis=1)
                        group_pred = tf.reduce_sum(fact_key @ tf.transpose(label_output, [0, 2, 1]), axis=1) * 10.0
                        group_pred = K.softmax(group_pred)
                        re_context = group_pred @ context_list
                    else:
                        group_pred = group_chosen(self.group_chosen_hidden(fact_base))
                        group_pred = K.softmax(group_pred)
                        group_choose = tf.one_hot(tf.argmax(group_pred, axis=-1), self.group_num)
                        re_context = group_pred @ context_list

                context_word = tf.reshape(context_generation_w(re_context), shape=(-1, 1, 1, self.han_size * 2))
                context_sentence = tf.reshape(context_generation_s(re_context), shape=(-1, 1, self.han_size * 2))

        with tf.name_scope('Encoding'):
            re_fact_rep_sentences, score_word = word_reattention([fact_word_level, context_word], mask=word_mask, dropout=dropout)
            re_fact_sentence_level = self.run_model(re_fact_rep_sentences, sentence_mask, sentence_re_encoder)
            fact_prior, score_sentence = sentence_reattention([re_fact_sentence_level, context_sentence], mask=sentence_mask, dropout=dropout)
            # fact_prior = tf.concat([fact_prior, CLS_output], axis=-1)
        return fact_prior, re_fact_sentence_level, group_pred, score_word, score_sentence

    def re_encodering_law(self, inputs, group_indexes, masks, context_list, name, context_funcs, encoder_funcs, dropout=None):
        law_word_level = inputs
        word_mask_law, sentence_mask_law = masks
        context_generation_w, context_generation_s = context_funcs
        word_reattention, sentence_re_encoder, sentence_reattention = encoder_funcs

        with tf.name_scope(name+'ContextForLaw'):
            context_law = tf.gather(context_list, group_indexes)
            # context_law = tf.gather(self.context, group_indexes)
            context_word_l = tf.reshape(context_generation_w(context_law), shape=(-1, 1, 1, self.han_size * 2))
            context_sentence_l = tf.reshape(context_generation_s(context_law), shape=(-1, 1, self.han_size * 2))

        with tf.name_scope(name+'Encoding'):
            re_law_rep_sentences, _ = word_reattention([law_word_level, context_word_l], mask=word_mask_law, dropout=dropout)
            re_law_sentence_level = self.run_model(re_law_rep_sentences, sentence_mask_law, sentence_re_encoder)
            law_prior, _ = sentence_reattention([re_law_sentence_level, context_sentence_l], mask=sentence_mask_law, dropout=dropout)

        return law_prior

    def call(self, inputs, law_information=None, training=None, warming_up=False, dropout=None,
             word_mask=None, sentence_mask=None, accu_information=None, time_information=None,
             word_mask_law=None, sentence_mask_law=None):
        fact_inputs = inputs
        law_inputs, adj_matrix_law, group_indexes, law_inputs_posterior, adj_matrix_posterior = law_information

        with tf.name_scope('law_encoding_base'):
            law_word_level = self.run_model(law_inputs, word_mask_law, self.word_encoder)
            law_rep_sentences, _ = self.word_attention(law_word_level, mask=word_mask_law, dropout=dropout)
            law_sentence_level = self.run_model(law_rep_sentences, sentence_mask_law, self.sentence_encoder)
            law_base, _ = self.sentence_attention(law_sentence_level, mask=sentence_mask_law, dropout=dropout)

        with tf.name_scope('GeneratePriorGroupInformation'):
            distilled_law_prior = law_base
            for i in range(self.num_distill_layers):
                distilled_law_prior, aggregate_law_prior = self.graph_distillers_prior[i]([distilled_law_prior, adj_matrix_law])
            context_list_prior = self.distill_pooling(features=distilled_law_prior, group_index=group_indexes, dropout=dropout)
            # key_list_prior = self.aggregate_pooling(features=aggregate_law_prior, group_index=group_indexes, dropout=dropout)

        with tf.name_scope('EncodingLaw'):
            with tf.name_scope('PriorEncodingLaw'):
                law_prior = self.re_encodering_law(inputs=law_word_level, group_indexes=group_indexes, dropout=dropout,
                                                   masks=[word_mask_law, sentence_mask_law],
                                                   context_list=context_list_prior,
                                                   encoder_funcs=[self.word_attention_prior, self.sentence_encoder_prior, self.sentence_attention_prior],
                                                   context_funcs=[self.context_w_prior, self.context_s_prior],
                                                   name='Prior')

            with tf.name_scope('GeneratePosteriorGroupInformation'):
                distilled_law_posterior, aggregate_law_posterior = law_inputs_posterior, law_inputs_posterior
                for i in range(self.num_distill_layers):
                    distilled_law_posterior, aggregate_law_posterior = self.graph_distillers_posterior[i]([distilled_law_posterior, aggregate_law_posterior, adj_matrix_posterior])

                if warming_up:
                    distilled_law_posterior *= self.posterior_mask
                    aggregate_law_posterior *= self.posterior_mask

                context_list_posterior = distilled_law_posterior
                if dropout is not None:
                    context_list_posterior = tf.nn.dropout(context_list_posterior, rate=dropout)

            with tf.name_scope('PosteriorEncodingLaw'):
                law_posterior = self.re_encodering_law(inputs=law_word_level, group_indexes=group_indexes,
                                                       masks=[word_mask_law, sentence_mask_law], dropout=dropout,
                                                       context_list=context_list_posterior,
                                                       encoder_funcs=[self.word_attention_posterior, self.sentence_encoder_posterior, self.sentence_attention_posterior],
                                                       context_funcs=[self.context_w_posterior, self.context_s_posterior],
                                                       name='Posterior')

        with tf.name_scope('EncodeFact'):
            with tf.name_scope('BaseEncodingFact'):
                fact_word_level = self.run_model(fact_inputs, word_mask, self.word_encoder)
                fact_rep_sentences, score_w_base = self.word_attention(fact_word_level, mask=word_mask, dropout=dropout)
                fact_sentence_level = self.run_model(fact_rep_sentences, sentence_mask, self.sentence_encoder)
                fact_base, score_s_base = self.sentence_attention(fact_sentence_level, mask=sentence_mask, dropout=dropout)

            with tf.name_scope('PriorEncodingFact'):
                fact_word_level_1 = self.run_model(fact_inputs, word_mask, self.word_encoder_prior)
                fact_prior, fact_prior_sentence, group_pred_prior, score_w_prior, score_s_prior = \
                    self.re_encoding_fact(inputs=[fact_base, None, context_list_prior, fact_word_level_1, fact_sentence_level],
                                          dense_funcs=[None, None],
                                          # dense_funcs=[self.matching_law_prior, self.matching_fact_prior],
                                          context_funcs=[self.group_chosen, None, self.context_w_prior, self.context_s_prior],
                                          encoder_funcs=[self.word_attention_prior, self.sentence_encoder_prior, self.sentence_attention_prior],
                                          masks=[word_mask, sentence_mask, None], dropout=dropout, name='Prior')

            with tf.name_scope('PosteriorEncodingFact_Law'):
                fact_sentence = tf.concat([fact_sentence_level, fact_prior_sentence], axis=1)
                new_sentence_mask = tf.concat([sentence_mask, sentence_mask], axis=-1)
                fact_word_level_2 = self.run_model(fact_inputs, word_mask, self.word_encoder_posterior)
                fact_posterior, fact_posterior_sentence, group_pred_posterior, score_w_posterior, score_s_posterior = \
                    self.re_encoding_fact(inputs=[None, law_inputs_posterior, context_list_posterior, fact_word_level_2, fact_sentence],
                                          dense_funcs=[self.matching_law_posterior, self.matching_fact_posterior],
                                          context_funcs=[None, self.Transformer_posterior, self.context_w_posterior, self.context_s_posterior],
                                          encoder_funcs=[self.word_attention_posterior, self.sentence_encoder_posterior, self.sentence_attention_posterior],
                                          masks=[word_mask, sentence_mask, new_sentence_mask], dropout=dropout, name='Posterior')

            if warming_up:
                fact_posterior *= self.posterior_maskF

        if self.accu_relation is not None:
            accu_inputs_posterior, accu_adj_matrix_posterior = accu_information
            with tf.name_scope('GeneratePosteriorGroupInformation_A'):
                distilled_accu_posterior, aggregate_accu_posterior = accu_inputs_posterior, accu_inputs_posterior
                for i in range(self.num_distill_layers):
                    distilled_accu_posterior, aggregate_accu_posterior = self.graph_distillers_posterior_accu[i]([distilled_accu_posterior, aggregate_accu_posterior, accu_adj_matrix_posterior])
                if warming_up:
                    distilled_accu_posterior *= self.posterior_mask
                    aggregate_accu_posterior *= self.posterior_mask

                context_list_posterior_A = distilled_accu_posterior
                if dropout is not None:
                    context_list_posterior_A = tf.nn.dropout(context_list_posterior_A, rate=dropout)

            with tf.name_scope('PosteriorEncodingFact_Law'):
                fact_sentence = tf.concat([fact_sentence_level, fact_prior_sentence], axis=1)
                new_sentence_mask = tf.concat([sentence_mask, sentence_mask], axis=-1)
                fact_word_level_3 = self.run_model(fact_inputs, word_mask, self.word_encoder_posterior_A)
                fact_posterior_A, fact_posterior_sentence_A, group_pred_posterior_A, score_w_posterior_A, score_s_posterior_A = \
                    self.re_encoding_fact(inputs=[None, accu_inputs_posterior, context_list_posterior_A, fact_word_level_3, fact_sentence],
                                          dense_funcs=[self.matching_law_posterior_A, self.matching_fact_posterior_A],
                                          context_funcs=[None, self.Transformer_posterior_A, self.context_w_posterior_A, self.context_s_posterior_A],
                                          encoder_funcs=[self.word_attention_posterior_A, self.sentence_encoder_posterior_A, self.sentence_attention_posterior_A],
                                          masks=[word_mask, sentence_mask, new_sentence_mask], dropout=dropout, name='Posterior')

            fact_rep = tf.concat([fact_base, fact_prior, fact_posterior, fact_posterior_A], axis=-1)
            law_rep = tf.concat([law_base, law_prior, law_posterior], axis=-1)

            return [fact_rep, law_rep, group_pred_prior, group_pred_posterior, group_pred_posterior_A,
                    score_w_base, score_s_base,
                    score_w_prior, score_s_prior,
                    score_w_posterior, score_s_posterior,
                    score_w_posterior_A, score_s_posterior_A]

        fact_rep = tf.concat([fact_base, fact_prior, fact_posterior], axis=-1)
        law_rep = tf.concat([law_base, law_prior, law_posterior], axis=-1)
        return [fact_rep, law_rep, group_pred_prior, group_pred_posterior,
                score_w_base, score_s_base,
                score_w_prior, score_s_prior,
                score_w_posterior, score_s_posterior]





