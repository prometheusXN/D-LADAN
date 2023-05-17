import tensorflow as tf
from tensorflow.keras.layers import Embedding, Input, Lambda, Dense
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import numpy as np
import tensorflow.keras as keras
from Model_component.Han_component import Han
from parser import ConfigParser
from tensorflow.keras.optimizers import Adam, SGD


def han_model(config: ConfigParser, word_dict_size, embedding_dim, emb_path, law_num,
              accu_num, time_num, word2id_dict, mode='training'):
    '''
    :param config:
    :param word_dict_size:
    :param embedding_dim:
    :param emb_path:
    :param law_num:
    :param accu_num:
    :param time_num:
    :param word2id_dict:
    :param mode:
    :return:
    '''
    learning_rate = config.getfloat('train', 'learning_rate')
    word_num = config.getint('data', 'sentence_num')
    sent_num = config.getint('data', 'sentence_len')

    with tf.name_scope('define_input'):
        fact_input = Input(shape=(sent_num, word_num), name='fact_description')
        fact_mask = tf.cast(tf.cast(fact_input - word2id_dict['BLANK'], tf.bool), tf.float32)
        fact_sent_len = tf.reduce_sum(fact_mask, -1)
        fact_doc_mask = tf.cast(tf.cast(fact_sent_len, tf.bool), tf.float32)

    with tf.name_scope('define_model'):
        with tf.name_scope('define_Embeddings'):
            embedding_matrix = np.cast[np.float](np.load(emb_path))
            embedding_layer = Embedding(word_dict_size, embedding_dim,
                                        embeddings_initializer=keras.initializers.Constant(embedding_matrix),
                                        trainable=False,
                                        mask_zero=True)

        with tf.name_scope('define_FeatureExtracter'):
            Han_model = Han(config=config, trainable=True, name='Han_model')
            Law_decoder = Dense(units=law_num, name='Law_decoder')
            Accu_decoder = Dense(units=accu_num, name='Accu_decoder')
            Time_decoder = Dense(units=time_num, name='Time_decoder')

        with tf.name_scope('model_process'):
            fact_description = embedding_layer(fact_input)
            fact_represents = Han_model(fact_description,
                                        word_mask=fact_mask,
                                        sentence_mask=fact_doc_mask)

            output_law, output_accu, output_time = \
                Law_decoder(fact_represents), Accu_decoder(fact_represents), Time_decoder(fact_represents)

            pred_law = Lambda(lambda x: K.softmax(x, axis=-1), name='output_law')(output_law)
            pred_accu = Lambda(lambda x: K.softmax(x, axis=-1), name='output_accu')(output_accu)
            pred_time = Lambda(lambda x: K.softmax(x, axis=-1), name='output_time')(output_time)

        model = Model([fact_input], [pred_law, pred_accu, pred_time])
        model.compile(optimizer=Adam(learning_rate=learning_rate, beta_1=.9, beta_2=.999, epsilon=1e-7),
                      loss={'output_law': "categorical_crossentropy",
                            'output_accu': "categorical_crossentropy",
                            'output_time': "categorical_crossentropy"},
                      metrics={'output_law': "accuracy",
                               'output_accu': "accuracy",
                               'output_time': "accuracy"},
                      run_eagerly=True)

        model.summary()
        # print(model.get_layer('Han_model').word_attention.key_encoder.get_weights())
        return model