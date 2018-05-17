import numpy as np
import tensorflow as tf
from base.base_model import BaseModel

class CNNClassifier(BaseModel):
    def __init__(self, config, sequence_length, vocab_size):
        super(CNNClassifier, self).__init__(config)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)
        self.inputs_placeholder = tf.placeholder(tf.int32, [None, self.sequence_length], name='x')
        self.labels_placeholder = tf.placeholder(tf.float32, [None, self.config.num_class], name='y')
        self.dropout_placeholder = tf.placeholder(tf.float32, name='dropout')
        self.embedded_text_expanded = self.add_embedding()
        self.logits = self.add_prediction_op()
        self.loss, self.accuracy = self.add_loss_op(self.logits)
        self.train_op = self.add_training_op(self.loss)

    def add_embedding(self):
        with tf.variable_scope('embedding'):
            self.embeddings = tf.Variable(
                tf.random_uniform([self.vocab_size, self.config.embedding_size], -1.0, 1.0))
            embedded_text = tf.nn.embedding_lookup(self.embeddings, self.inputs_placeholder)
            embedded_text_expanded = tf.expand_dims(embedded_text, axis=-1)
            return embedded_text_expanded

    def add_prediction_op(self):
        outputs = []
        for i, filter_size in enumerate(self.config.filter_sizes):
            with tf.variable_scope('conv-%s' % filter_size):
                # [filter_height, filter_width, in_channels, out_channels]
                filter_shape = [filter_size, self.config.embedding_size, 1, self.config.num_filters]
                W = tf.get_variable("W", filter_shape, tf.float32,
                        initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable('b', [self.config.num_filters], tf.float32, tf.constant_initializer(0.1))
                conv = tf.nn.conv2d(self.embedded_text_expanded, W, strides=[1,1,1,1], padding="VALID")
                h = tf.nn.relu(conv + b)
                output = tf.nn.max_pool(
                            h,
                            ksize=[1,self.sequence_length - filter_size + 1,1,1],
                            strides=[1,1,1,1],
                            padding='VALID')
                outputs.append(output)

        feature_map = tf.concat(outputs, axis=3)
        num_filters_total = len(self.config.filter_sizes) * self.config.num_filters
        feature_map = tf.reshape(feature_map, [-1, num_filters_total])

        with tf.variable_scope('dropout'):
            feature_map = tf.nn.dropout(feature_map, self.dropout_placeholder)

        with tf.variable_scope('output'):
            W = tf.get_variable('W', [num_filters_total, self.config.num_class], tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b', [self.config.num_class], tf.float32, initializer=tf.constant_initializer(0))
            logits = tf.nn.softmax(tf.matmul(feature_map, W) + b)

        return logits

    def add_loss_op(self, logits):
        with tf.variable_scope('loss'):
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.labels_placeholder)
            loss = tf.reduce_mean(loss)

            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(self.labels_placeholder, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return loss, accuracy

    def add_training_op(self, loss):
        train_op = tf.train.AdamOptimizer(self.config.learning_rate).minimize(loss, global_step=self.global_step_tensor)
        return train_op

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
