#!/usr/bin/python3.7 -u
import tensorflow as tf
import numpy as np
import crf

from collections import namedtuple


def char_embeddings(chars, len_chars, n_chars, config):
    batch_size = tf.shape(chars)[0]
    max_word = tf.shape(chars)[1]
    max_char = tf.shape(chars)[2]

    char_embeddings = tf.get_variable(
        'char_embeddings',
        [n_chars, config.char_embed_size],
        initializer = tf.variance_scaling_initializer(
            distribution = "uniform"
        ),
        trainable = True
    )

    char_ids = tf.reshape(chars, [-1])
    embedded_chars = tf.reshape(
        tf.nn.embedding_lookup(char_embeddings, tf.reshape(chars, [-1])),
        [batch_size * max_word, max_char, config.char_embed_size]
    )

    embed_mask = tf.expand_dims(
        tf.sequence_mask(
            tf.reshape(len_chars, [-1]),
            max_char,
            dtype = tf.float32
        ),
        axis = -1
    )

    return embedded_chars * embed_mask

def conv_max_pool(embeddings, len_words, config):
    pad_shape = [config.conv_kernel - 1] * 2
    conv = tf.layers.conv1d(
        tf.pad(
            embeddings,
            [[0, 0], [0, 0], pad_shape],
            constant_values = 1 # <PAD>
        ),
        filters = config.pool_size,
        kernel_size = config.conv_kernel,
        strides = 1,
        padding = 'SAME',
        use_bias = True,
        activation = 'relu'
    )

    max_pool = tf.reduce_max(tf.matrix_transpose(conv), axis = 2)
    pool_mask = tf.reshape(tf.sequence_mask(len_words, dtype = tf.float32), [-1, 1])

    return max_pool * pool_mask


Config = namedtuple(
    'Config',
    [
        'batch_size',
        'char_embed_size',
        'conv_kernel',
        'depth',
        'dropout',
        'h_size',
        'learning_rate',
        'pool_size',
    ]
)


class Network:

    def __init__(self, name, batch, config, v_shape):
        words, pos, gazetteer, chars, len_chars, labels, len_words = batch
        n_words, n_pos, n_categories, n_chars, n_tags = v_shape

        batch_size = tf.shape(words)[0]
        max_words = tf.shape(words)[1]

        embeddings = char_embeddings(chars, len_chars, n_chars, config)
        embedding_pool = tf.reshape(
            conv_max_pool(embeddings, len_words, config),
            [batch_size, max_words, config.pool_size]
        )

        fw = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.LSTMCell(config.h_size) for _ in range(config.depth)]
        )
        bw = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.LSTMCell(config.h_size) for _ in range(config.depth)]
        )

        self.dropout = tf.placeholder(tf.float32, [])

        features = tf.nn.dropout(
            tf.concat([words, pos, gazetteer, embedding_pool], axis = 2),
            keep_prob = 1 - self.dropout
        )

        output, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw = fw,
            cell_bw = bw,
            inputs = features,
            sequence_length = len_words,
            dtype = tf.float32
        )
        output = tf.concat(output, axis = 2)
        output = tf.layers.dense(
            tf.nn.dropout(output, keep_prob = 1 - self.dropout),
            units = n_tags,
            name = "output"
        )

        log_likelihood, transition = crf.crf_log_likelihood(
            output,
            tag_indices = labels,
            sequence_lengths = len_words
        )

        # Viterbi decode
        self.predict, self.score = crf.crf_decode(
            output,
            transition_params = transition,
            sequence_length = len_words
        )

        # Cross-entropy loss
        self.loss = tf.reduce_mean(-log_likelihood, name = "loss")

        tvars = tf.trainable_variables()
        gradients, _ = tf.clip_by_global_norm(
            tf.gradients(self.loss, tvars),
            clip_norm = 5.0
        )

        optimizer = tf.train.AdamOptimizer(config.learning_rate, epsilon = 0.1)
        self.train = optimizer.apply_gradients(zip(gradients, tvars))




if __name__ == '__main__':

    from data import Vocab

    vocab = Vocab('etc/nynorsk.300.cbow.bin')

    config = Config(
        batch_size = 100,
        char_embed_size = 25,
        conv_kernel = 3,
        depth = 1,
        dropout = 0.5,
        h_size = 256,
        learning_rate = 0.01,
        pool_size = 53
    )

    output_types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.int32)
    output_shapes = (
        [None, vocab.n_words], # Word embeddings for each word in the sentence
        [None, vocab.n_pos],   # one_hot encoded PoS for each word
        [None, None],          # The characters for each word
        [None],                # The number of characters pr word
        [None],                # The labels for each word
        []                     # the number of words in sentence
    )

    data = tf.data.Dataset.from_generator(
        vocab.examples('data/nynorsk/no_nynorsk-ud-test.bioes'),
        output_types = output_types,
        output_shapes = output_shapes
    ).padded_batch(
        batch_size = config.batch_size,
        padded_shapes = output_shapes
    )

    iterator = tf.data.Iterator.from_structure(
        output_types = output_types,
        output_shapes = data.output_shapes
    )

    data_init = iterator.make_initializer(
        data,
        name = "data"
    )

    tf.set_random_seed(0)

    batch = iterator.get_next(name = "batch")

    net = Network(
        name = "test",
        batch = batch,
        config = config,
        v_shape = vocab.shape
    )

    batch_size = tf.shape(batch[0])[0]
    max_words = tf.shape(batch[0])[1]

    embed_size = 30
    embeddings = char_embeddings(batch[2], batch[3], vocab.n_chars, embed_size)
    embedding_pool = conv_max_pool(embeddings, embed_size, batch[5])

    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)

        session.run(data_init)
        val = session.run(embedding_pool)
        print(val)


        # timestep = 0
        # while True:

        #     total_loss = 0
        #     step = 0
        #     session.run(data_init)
        #     try:
        #         while True:
        #             _, loss = session.run(
        #                 (net.train, net.loss),
        #                 feed_dict = {
        #                     net.dropout : 0.0
        #                 }
        #             )

        #             prediction = session.run(
        #                 net.predict,
        #                 feed_dict = {
        #                     net.dropout : 0.0
        #                 }
        #             )

        #             print(prediction)
        #             print(f"Step {step} : {loss}")
        #             total_loss += loss
        #             step += 1
        #     except tf.errors.OutOfRangeError:
        #         pass

        #     print(f"Timestep {timestep}")
        #     print(f"loss = {total_loss / step}")
        #     print("==============")
        #     timestep += 1
