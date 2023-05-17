import sys
sys.path.append("..")
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Input, Lambda, Dense
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import numpy as np
import tensorflow.keras as keras
from Model_component.Ladan_component import Ladan
from Model_component.CosinClassifier import CosClassifier, MultiplyScalar
from tensorflow.python.keras.engine.training import _minimize, _keras_api_gauge, _disallow_inside_tf_function, _minimum_control_deps
from tensorflow.python.keras.engine.training import *


def negative_log_likehood(y_true, y_pred):
    log_y = K.log(y_pred) * y_true
    loss = -K.mean(K.sum(log_y, axis=-1))
    return loss


class ladan_model(Model):
    def __init__(self, config, word2id_dict, emb_path, group_num, law_input, law_adj_matrix, group_indexes, trainable, **kwargs):
        self.trainable = trainable
        super(ladan_model, self).__init__(**kwargs)
        self.word2id_dict = word2id_dict
        self.word_dict_size = len(word2id_dict)
        self.emb_path = emb_path

        self.config = config
        self.word_num = config.getint('data', 'sentence_num')
        self.sent_num = config.getint('data', 'sentence_len')
        self.law_relation_threshold = config.getfloat('data', 'graph_threshold')
        self.embedding_dim = config.getint('data', 'vec_size')
        self.law_num = config.getint('num_class_small', 'law_num')
        self.accu_num = config.getint('num_class_small', 'accu_num')
        self.time_num = config.getint('num_class_small', 'time_num')
        self.group_num = group_num

        self.law_input = tf.Variable(law_input, dtype=tf.int32, name='law_input', trainable=False)
        self.law_adj_matrix = tf.Variable(law_adj_matrix, dtype=tf.int32, name='law_adj_matrix', trainable=False)
        self.group_indexes = tf.Variable(group_indexes, dtype=tf.int32, name='group_indexes', trainable=False)
        self.build_components()
        # self.build((None, self.sent_num, self.word_num))
        # self.out = self.call(Input(shape=(self.sent_num, self.word_num)))

    def build_components(self):
        with tf.name_scope('define_model'):
            with tf.name_scope('define_Embeddings'):
                embedding_matrix = np.cast[np.float](np.load(self.emb_path))
                self.embedding_layer = Embedding(self.word_dict_size, self.embedding_dim,
                                            embeddings_initializer=keras.initializers.Constant(embedding_matrix),
                                            trainable=False,
                                            mask_zero=True)

            with tf.name_scope('define_FeatureExtracter'):
                self.Ladan = Ladan(config=self.config, group_num=self.group_num, trainable=self.trainable, name='Ladan_model')

            with tf.name_scope('define_Decoder'):
                self.Law_decoder_m = Dense(units=256, name='Law_decoder_m', trainable=self.trainable, activation='relu')
                self.Accu_decoder_m = Dense(units=256, name='Accu_decoder_m', trainable=self.trainable, activation='relu')
                self.Time_decoder_m = Dense(units=256, name='Time_decoder_m', trainable=self.trainable, activation='relu')

                self.Law_decoder = Dense(units=self.law_num, name='Law_decoder', trainable=self.trainable, use_bias=False)
                self.Accu_decoder = Dense(units=self.accu_num, name='Accu_decoder', trainable=self.trainable, use_bias=False)
                self.Time_decoder = Dense(units=self.time_num, name='Time_decoder', trainable=self.trainable, use_bias=False)

    def call(self, inputs, training=None, dropout=None, **kwargs):
        fact_input = inputs

        with tf.name_scope('compute_mask'):
            fact_mask = tf.cast(tf.cast(fact_input - self.word2id_dict['BLANK'], tf.bool), tf.float32)
            fact_sent_len = tf.reduce_sum(fact_mask, -1)
            fact_doc_mask = tf.cast(tf.cast(fact_sent_len, tf.bool), tf.float32)

            law_mask = tf.cast(tf.cast(self.law_input - self.word2id_dict['BLANK'], tf.bool), tf.float32)
            law_sent_len = tf.reduce_sum(law_mask, -1)
            law_doc_mask = tf.cast(tf.cast(law_sent_len, tf.bool), tf.float32)

        with tf.name_scope('model_process'):
            fact_description = self.embedding_layer(fact_input)
            law_description = self.embedding_layer(self.law_input)

            fact_rep, law_rep, group_pred = self.Ladan(fact_description, training=training, dropout=dropout,
                                                       law_information=[law_description, self.law_adj_matrix,
                                                                        self.group_indexes],
                                                       word_mask=fact_mask, sentence_mask=fact_doc_mask,
                                                       word_mask_law=law_mask, sentence_mask_law=law_doc_mask)

            output_law = self.Law_decoder(self.Law_decoder_m(fact_rep))
            output_accu = self.Accu_decoder(self.Accu_decoder_m(fact_rep))
            output_time = self.Time_decoder(self.Time_decoder_m(fact_rep))

            pred_law = Lambda(lambda x: K.softmax(x, axis=-1), name='output_law')(output_law)
            pred_accu = Lambda(lambda x: K.softmax(x, axis=-1), name='output_accu')(output_accu)
            pred_time = Lambda(lambda x: K.softmax(x, axis=-1), name='output_time')(output_time)

        return {'law': pred_law,
                'accu': pred_accu,
                'time': pred_time,
                'group_prior': group_pred,
                }

    def train_step(self, data, **kwargs):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        # tf.print(tf.argmax(y['law'], axis=-1), summarize=-1)
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True, dropout=None, **kwargs)
            loss = self.compiled_loss(
                y, y_pred, sample_weight, regularization_losses=self.losses)

        _minimize(self.distribute_strategy, tape, self.optimizer, loss,
                  self.trainable_variables)

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):

        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        y_pred = self(x, training=False)
        # Updates stateful loss metrics.
        self.compiled_loss(
            y, y_pred, sample_weight, regularization_losses=self.losses)

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):

        data = data_adapter.expand_1d(data)
        x, _, _ = data_adapter.unpack_x_y_sample_weight(data)
        return self(x, training=False)

    def make_train_function(self):
        if self.train_function is not None:
            return self.train_function

        def step_function(model, iterator, **kwargs):
            """Runs a single training step."""

            def run_step(data, **kwargs):
                outputs = model.train_step(data, **kwargs)
                # Ensure counter is updated only if `train_step` succeeds.
                with ops.control_dependencies(_minimum_control_deps(outputs)):
                    model._train_counter.assign_add(1)  # pylint: disable=protected-access
                return outputs

            data = next(iterator)

            outputs = model.distribute_strategy.run(run_step, args=(data, ), kwargs=kwargs)
            outputs = reduce_per_replica(
                outputs, self.distribute_strategy, reduction='first')
            write_scalar_summaries(outputs, step=model._train_counter)  # pylint: disable=protected-access
            return outputs

        if self._steps_per_execution.numpy().item() == 1:

            def train_function(iterator, **kwargs):
                """Runs a training execution with one step."""
                return step_function(self, iterator, **kwargs)

        else:

            def train_function(iterator, **kwargs):
                """Runs a training execution with multiple steps."""
                outputs = step_function(self, iterator, **kwargs)
                for _ in math_ops.range(self._steps_per_execution - 1):
                    outputs = step_function(self, iterator, **kwargs)
                return outputs

        if not self.run_eagerly:
            train_function = def_function.function(
                train_function, experimental_relax_shapes=True)

        self.train_function = train_function
        return self.train_function

    @enable_multi_worker
    def fit_(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None,
             validation_split=0., validation_data=None, shuffle=True, class_weight=None, sample_weight=None,
             initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_batch_size=None,
             validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False,
             ave_acc=True):

        _keras_api_gauge.get_cell('fit').set(True)
        # Legacy graph support is contained in `training_v1.Model`.
        version_utils.disallow_legacy_graph('Model', 'fit')
        self._assert_compile_was_called()
        self._check_call_args('fit')
        _disallow_inside_tf_function('fit')

        if validation_split:
            # Create the validation data using the training data. Only supported for
            # `Tensor` and `NumPy` input.
            (x, y, sample_weight), validation_data = (
                data_adapter.train_validation_split(
                    (x, y, sample_weight), validation_split=validation_split))

        if validation_data:
            val_x, val_y, val_sample_weight = (
                data_adapter.unpack_x_y_sample_weight(validation_data))

        with self.distribute_strategy.scope(), \
             training_utils.RespectCompiledTrainableState(self):
            # Creates a `tf.data.Dataset` and handles batch and epoch iteration.
            data_handler = data_adapter.DataHandler(
                x=x,
                y=y,
                sample_weight=sample_weight,
                batch_size=batch_size,
                steps_per_epoch=steps_per_epoch,
                initial_epoch=initial_epoch,
                epochs=epochs,
                shuffle=shuffle,
                class_weight=class_weight,
                max_queue_size=max_queue_size,
                workers=workers,
                use_multiprocessing=use_multiprocessing,
                model=self,
                steps_per_execution=self._steps_per_execution)

            # Container that configures and calls `tf.keras.Callback`s.
            if not isinstance(callbacks, callbacks_module.CallbackList):
                callbacks = callbacks_module.CallbackList(
                    callbacks,
                    add_history=True,
                    add_progbar=verbose != 0,
                    model=self,
                    verbose=verbose,
                    epochs=epochs,
                    steps=data_handler.inferred_steps)

            self.stop_training = False
            train_function = self.make_train_function()
            self._train_counter.assign(0)
            callbacks.on_train_begin()
            training_logs = None
            # Handle fault-tolerance for multi-worker.
            # TODO(omalleyt): Fix the ordering issues that mean this has to
            # happen after `callbacks.on_train_begin`.
            data_handler._initial_epoch = (  # pylint: disable=protected-access
                self._maybe_load_initial_epoch_from_ckpt(initial_epoch))
            for epoch, iterator in data_handler.enumerate_epochs():
                self.reset_metrics()
                callbacks.on_epoch_begin(epoch)
                with data_handler.catch_stop_iteration():
                    for step in data_handler.steps():
                        with trace.Trace(
                                'TraceContext',
                                graph_type='train',
                                epoch_num=epoch,
                                step_num=step,
                                batch_size=batch_size):
                            callbacks.on_train_batch_begin(step)
                            tmp_logs = train_function(iterator)
                            if data_handler.should_sync:
                                context.async_wait()
                            logs = tmp_logs  # No error, now safe to assign to logs.
                            end_step = step + data_handler.step_increment
                            callbacks.on_train_batch_end(end_step, logs)
                epoch_logs = copy.copy(logs)

                # Run validation.
                if validation_data and self._should_eval(epoch, validation_freq):
                    # Create data_handler for evaluation and cache it.
                    if getattr(self, '_eval_data_handler', None) is None:
                        self._eval_data_handler = data_adapter.DataHandler(
                            x=val_x,
                            y=val_y,
                            sample_weight=val_sample_weight,
                            batch_size=validation_batch_size or batch_size,
                            steps_per_epoch=validation_steps,
                            initial_epoch=0,
                            epochs=1,
                            max_queue_size=max_queue_size,
                            workers=workers,
                            use_multiprocessing=use_multiprocessing,
                            model=self,
                            steps_per_execution=self._steps_per_execution)
                    val_logs = self.evaluate(
                        x=val_x,
                        y=val_y,
                        sample_weight=val_sample_weight,
                        batch_size=validation_batch_size or batch_size,
                        steps=validation_steps,
                        callbacks=callbacks,
                        max_queue_size=max_queue_size,
                        workers=workers,
                        use_multiprocessing=use_multiprocessing,
                        return_dict=True)
                    if ave_acc:
                        acc = (val_logs['accu_accuracy'] + val_logs['law_accuracy'] + val_logs[
                            'time_accuracy'])/3.0
                        ave_acc = {'val_ave_accuracy': acc}
                        epoch_logs.update(ave_acc)
                    val_logs = {'val_' + name: val for name, val in val_logs.items()}
                    epoch_logs.update(val_logs)

                callbacks.on_epoch_end(epoch, epoch_logs)
                training_logs = epoch_logs
                if self.stop_training:
                    break

            # If eval data_hanlder exists, delete it after all epochs are done.
            if getattr(self, '_eval_data_handler', None) is not None:
                del self._eval_data_handler
            callbacks.on_train_end(logs=training_logs)
            return self.history
