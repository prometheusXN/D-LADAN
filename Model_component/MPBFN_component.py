import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Dense, Lambda
import tensorflow.keras.backend as K
import tensorflow.keras as keras
import tensorflow.keras.regularizers as regularizers


def normalize(x):
    x /= (K.sum(x, axis=-1, keepdims=True) + K.epsilon())
    return x


class MPBFNDecoder(Model):

    def __init__(self, config, law_num, accu_num, term_num, trainable, with_proto=False, **kwargs):
        super(MPBFNDecoder, self).__init__(**kwargs)
        self.trainable = trainable
        self.law_num = law_num
        self.accu_num = accu_num
        self.term_num = term_num
        self.with_proto = with_proto
        self.hidden_size = config.getint("net", "hidden_size")

        self.law_embedding_layer = Embedding(self.law_num, self.hidden_size,
                                             trainable=True, mask_zero=False, name='law_embedding')
        self.accu_embedding_layer = Embedding(self.accu_num, self.hidden_size,
                                              trainable=True, mask_zero=False, name='accu_embedding')
        self.term_embedding_layer = Embedding(self.term_num, self.hidden_size,
                                              trainable=True, mask_zero=False, name='term_embedding')

        self.law_input = tf.constant([i for i in range(self.law_num)], dtype=tf.int32)
        self.accu_input = tf.constant([i for i in range(self.accu_num)], dtype=tf.int32)
        self.term_input = tf.constant([i for i in range(self.term_num)], dtype=tf.int32)

        self.Fact_trans = Dense(units=self.hidden_size, activation='tanh', name='Fact_trans')

        self.Term_Pred_Law = Dense(units=self.term_num, activation='softmax', name='Term_Pred_Law')
        self.Term_Pred_Accu = Dense(units=self.term_num, activation='softmax', name='Term_Pred_Accu')

        self.Law_pred_Term = Dense(units=self.law_num, activation='sigmoid', name='Law_pred_Term')
        self.Law_pred_Accu = Dense(units=self.law_num, activation='sigmoid', name='Law_pred_Accu')
        self.Accu_pred_Term = Dense(units=self.accu_num, activation='sigmoid', name='Accu_pred_Term')

        self.Law_sementic = Dense(units=self.hidden_size, activation='elu', use_bias=False, name='Law_sementic')
        self.Accu_sementic = Dense(units=self.hidden_size, activation='elu', use_bias=False, name='Accu_sementic')
        self.Term_sementic = Dense(units=self.hidden_size, activation='elu', use_bias=False, name='Term_sementic')

    def call(self, inputs, training=None, mask=None):
        fact, Law_decoder, Law_Scalar, Accu_decoder, Accu_Scalar = inputs
        fact = self.Fact_trans(fact)
        if self.with_proto:
            law_pred_1 = Law_Scalar(Law_decoder(fact)[0])
        else:
            law_pred_1 = Law_Scalar(Law_decoder(fact))    # size [batch_size, law_num]
        law_pred_1 = Lambda(lambda x: K.softmax(x, axis=-1), name='output_law')(law_pred_1)

        law_embedded_sequence = self.law_embedding_layer(self.law_input)    # size [law_num, 256]
        law_merge = law_pred_1 @ law_embedded_sequence
        law_merge_semantic = self.Law_sementic(law_merge)
        law_fix = fact * law_merge_semantic

        if self.with_proto:
            accu_pred_1 = Accu_Scalar(Accu_decoder(law_fix)[0])
        else:
            accu_pred_1 = Accu_Scalar(Accu_decoder(law_fix))
        accu_pred_1 = Lambda(lambda x: K.softmax(x, axis=-1), name='output_law')(accu_pred_1)

        accu_embedded_sequence = self.accu_embedding_layer(self.accu_input)
        accu_merge = accu_pred_1 @ accu_embedded_sequence
        accu_merge_semantic = self.Accu_sementic(accu_merge)
        accu_fix = fact * accu_merge_semantic

        term_pred_law = self.Term_Pred_Law(law_fix)
        term_pred_accu = self.Term_Pred_Accu(accu_fix)

        term_pred_1 = term_pred_law * term_pred_accu
        term_pred_1 = Lambda(normalize, name="term_preds")(term_pred_1)
        term_embedded_sequence = self.term_embedding_layer(self.term_input)
        term_merge = term_pred_1 @ term_embedded_sequence

        term2law = self.Law_pred_Term(term_merge)
        term2accu = self.Accu_pred_Term(term_merge)

        accu2law = self.Law_pred_Accu(accu_merge)

        law_pred_2 = law_pred_1 * term2law * accu2law
        law_pred_2 = Lambda(normalize, name="law_preds")(law_pred_2)

        accu_pred_2 = accu_pred_1 * term2accu
        accu_pred_2 = Lambda(normalize, name="accu_preds")(accu_pred_2)

        return [law_pred_2, accu_pred_2, term_pred_1]

    def compute_output_shape(self, input_shape):
        fact_shape = input_shape[0]
        return [(fact_shape[0], self.law_num), (fact_shape[0], self.law_num), (fact_shape[0], self.term_num)]
