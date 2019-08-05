import sys
import numpy as np
import tensorflow as tf

from pathlib import Path
from model import Network, Config
from data import Vocab
from itertools import chain




if __name__ == '__main__':

    config = Config(
        batch_size = 1,
        char_embed_size = 25,
        conv_kernel = 3,
        depth = 1,
        dropout = 0.5,
        h_size = 256,
        learning_rate = 0.01,
        pool_size = 53
    )

    vocab = Vocab(f'etc/samnorsk.300.skipgram.bin', 'etc/gazetteer.txt')

    output_types = (tf.float32, tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.int32)
    output_shapes = (
        [None, vocab.n_words],      # Word embeddings for each word in the sentence
        [None, vocab.n_pos],        # one_hot encoded PoS for each word
        [None, vocab.n_categories], # NE category memberships
        [None, None],               # The characters for each word
        [None],                     # The number of characters pr word
        [None],                     # The labels for each word
        []                          # the number of words in sentence
    )


    examples = tf.data.Dataset.from_generator(
        vocab.examples(sys.argv[2]),
        output_types = output_types,
        output_shapes = output_shapes
    ).padded_batch(
        batch_size = config.batch_size,
        padded_shapes = output_shapes
    )

    iterator = tf.data.Iterator.from_structure(
        output_types = output_types,
        output_shapes = examples.output_shapes
    )

    example_init = iterator.make_initializer(
        examples
    )

    batch = iterator.get_next(name = "batch")

    net = Network(
        name = 'samnorsk',
        batch = batch,
        config = config,
        v_shape = vocab.shape
    )

    meta_graph = Path(sys.argv[1])
    model = meta_graph.with_name(meta_graph.stem)
    saver = tf.train.Saver()

    with tf.Session() as session:

        saver.restore(session, str(model))

        session.run(example_init)
        try:
            while True:
                predicted = session.run(
                    net.predict,
                    feed_dict = {
                        net.dropout: 0.0
                    }
                )

                for label in chain(*predicted):
                    print(vocab.idx_tag[label])
                print()

        except tf.errors.OutOfRangeError:
            pass
