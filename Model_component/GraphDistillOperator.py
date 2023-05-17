import tensorflow as tf
import numpy as np
import networkx as nx
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras import activations
from utils.GumbelSoftmax import softmax_with_mask
import tensorflow.keras.regularizers as regularizers


def matrix_to_graph(adjacency_matrix):
    adjacency_matrix = adjacency_matrix.numpy()
    graph = nx.from_numpy_matrix(adjacency_matrix)

    def get_edge_index(graph):
        edge_list = nx.edges(graph)
        row = []
        col = []
        for i in edge_list:
            row.append(i[0])
            col.append(i[1])
        return [np.array(row), np.array(col)]
    edge_index = get_edge_index(graph)
    row, col = edge_index[0], edge_index[1]
    return row, col


class GraphDistillOperator(Layer):
    def __init__(self, trainable, out_dim, activation='tanh', withAgg=False, **kwargs):
        self.trainable = trainable
        self.activation = activations.get(activation)
        self.out_dim =out_dim
        self.withAgg = withAgg
        super(GraphDistillOperator, self).__init__(trainable=self.trainable, **kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        shape_proto, shape_matrix = input_shape
        # self.LayerNormalization = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.distill_dense = Dense(units=shape_proto[-1], name='distill_encoder', trainable=self.trainable)
        self.distll_out_dense = Dense(units=self.out_dim, name='aggregarate_encoder', trainable=self.trainable)
        if self.withAgg:
            self.aggregate_dense = Dense(units=shape_proto[-1], name='aggregate_encoder', trainable=self.trainable)
            self.aggregate_out_dense = Dense(units=self.out_dim, name='aggregate_out_encoder', trainable=self.trainable)

        super(GraphDistillOperator, self).build(input_shape)

    def call(self, inputs, mode='concat', dropout=None, **kwargs):
        features, adj_matrix = inputs
        adj_matrix = tf.cast(adj_matrix, dtype=tf.float32)
        node_num, feature_dim = features.get_shape()  # size [node_num, feature_dim]
        head_features = tf.tile(tf.expand_dims(features, axis=1), [1, node_num, 1])
        tail_features = tf.tile(tf.expand_dims(features, axis=0), [node_num, 1, 1])
        neigh_features = tf.concat([head_features, tail_features], axis=-1)

        neight_features_sum = tf.reduce_sum(tf.expand_dims(adj_matrix, axis=-1) * neigh_features, axis=1)
        neigh_mask = tf.reduce_max(adj_matrix, axis=-1, keepdims=True)
        neigh_num = tf.reduce_sum(adj_matrix, axis=-1, keepdims=True) + (1 - neigh_mask) * 1
        neight_features_ave = neight_features_sum / neigh_num

        neighbor_features = self.distill_dense(neight_features_ave)
        feature_updated = self.distll_out_dense((features - neighbor_features))
        feature_updated = tf.reshape(feature_updated, shape=[node_num, self.out_dim])
        feature_updated = self.activation(feature_updated)
        if dropout is not None:
            feature_updated = tf.nn.dropout(feature_updated, rate=dropout)

        if self.withAgg:
            neighbor_features_aggregate = self.aggregate_dense(adj_matrix @ features)
            feature_aggregate = self.aggregate_out_dense((features + neighbor_features_aggregate))
            feature_aggregate = tf.reshape(feature_aggregate, shape=[node_num, self.out_dim])
            feature_aggregate = self.activation(feature_aggregate)
            if dropout is not None:
                feature_aggregate = tf.nn.dropout(feature_aggregate, rate=dropout)
            return feature_updated, feature_aggregate
        else:
            return feature_updated, feature_updated

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_proto, shape_matrix = input_shape

        return [(shape_proto[0], self.out_dim), (shape_proto[0], self.out_dim)]


class GraphDistillOperatorWithEdgeWeight(Layer):
    def __init__(self, trainable, out_dim, activation='tanh', withAgg=False, **kwargs):
        self.trainable = trainable
        self.activation = activations.get(activation)
        self.out_dim =out_dim
        self.withAgg = withAgg
        super(GraphDistillOperatorWithEdgeWeight, self).__init__(trainable=self.trainable, **kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        shape_proto, key_proto, shape_matrix = input_shape
        # self.LayerNormalization_features = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # self.LayerNormalization_key = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.distill_dense = Dense(units=shape_proto[-1], name='distill_encoder', trainable=self.trainable)
        self.distll_out_dense = Dense(units=self.out_dim, name='distll_out_encoder', trainable=self.trainable)

        if self.withAgg:
            self.aggregate_dense = Dense(units=shape_proto[-1], name='aggregate_encoder', trainable=self.trainable)
            self.aggregate_out_dense = Dense(units=self.out_dim, name='aggregate_out_encoder', trainable=self.trainable)

        super(GraphDistillOperatorWithEdgeWeight, self).build(input_shape)

    def call(self, inputs:[tf.Tensor, tf.Tensor, tf.Tensor], mode='concat', dropout=None, **kwargs):
        features, key_features, adj_matrix = inputs
        adj_matrix = tf.cast(adj_matrix, dtype=tf.float32)
        # features = self.LayerNormalization_features(features)
        # key_features = self.LayerNormalization_key(key_features)
        node_num, feature_dim = features.get_shape()  # size [node_num, feature_dim]
        head_features = tf.tile(tf.expand_dims(features, axis=1), [1, node_num, 1])
        tail_features = tf.tile(tf.expand_dims(features, axis=0), [node_num, 1, 1])
        neigh_features_distill = tf.concat([head_features, tail_features], axis=-1)

        self_loop_mask = 1.0 - tf.eye(num_rows=node_num, dtype=tf.float32)
        adj_matrix_soft = softmax_with_mask(adj_matrix * 5.0, masks=self_loop_mask, axis=-1)

        neight_features_norm = tf.reduce_sum(tf.expand_dims(adj_matrix_soft, axis=-1) * neigh_features_distill, axis=1)

        neighbor_features_distill = self.distill_dense(neight_features_norm)
        feature_updated = self.distll_out_dense((features - neighbor_features_distill))
        feature_updated = tf.reshape(feature_updated, shape=[node_num, self.out_dim])
        feature_updated = self.activation(feature_updated)

        if dropout is not None:
            feature_updated = tf.nn.dropout(feature_updated, rate=dropout)

        if self.withAgg:
            neighbor_features_aggregate = self.aggregate_dense(adj_matrix_soft @ key_features)
            feature_aggregate = self.aggregate_out_dense((key_features + neighbor_features_aggregate))
            feature_aggregate = tf.reshape(feature_aggregate, shape=[node_num, self.out_dim])
            feature_aggregate = self.activation(feature_aggregate)
            if dropout is not None:
                feature_aggregate = tf.nn.dropout(feature_aggregate, rate=dropout)

            return feature_updated, feature_aggregate
        else:
            return feature_updated, feature_updated

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_proto, key_proto, shape_matrix = input_shape
        return [(shape_proto[0], self.out_dim), (shape_proto[0], self.out_dim)]


class GraphAggregateOperator(Layer):
    def __init__(self, trainable, out_dim, activation='tanh', withAgg=False, **kwargs):
        self.trainable = trainable
        self.activation = activations.get(activation)
        self.out_dim =out_dim
        self.withAgg = withAgg
        super(GraphAggregateOperator, self).__init__(trainable=self.trainable, **kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        shape_proto, shape_matrix = input_shape
        # self.LayerNormalization = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.aggregate_dense = Dense(units=shape_proto[-1], name='aggregate_encoder', trainable=self.trainable)
        self.aggregate_out_dense = Dense(units=self.out_dim, name='aggregate_out_encoder', trainable=self.trainable)

        super(GraphAggregateOperator, self).build(input_shape)

    def call(self, inputs, mode='concat', dropout=None, **kwargs):
        features, adj_matrix = inputs
        adj_matrix = tf.cast(adj_matrix, dtype=tf.float32)
        node_num, feature_dim = features.get_shape()  # size [node_num, feature_dim]

        neighbor_features_aggregate = self.aggregate_dense(adj_matrix @ features)
        feature_aggregate = self.aggregate_out_dense((features + neighbor_features_aggregate))
        feature_aggregate = tf.reshape(feature_aggregate, shape=[node_num, self.out_dim])
        feature_aggregate = self.activation(feature_aggregate)

        if dropout is not None:
            feature_aggregate = tf.nn.dropout(feature_aggregate, rate=dropout)

        return feature_aggregate

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_proto, shape_matrix = input_shape

        return (shape_proto[0], self.out_dim)