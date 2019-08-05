import re
import sys
import numpy as np
import fastText as ft

from pathlib import Path
from itertools import groupby, islice, dropwhile


def parse_file(name_file):
    with name_file.open() as f:
        lines = f.readlines()
    return parse_lines(lines)


def parse_lines(lines):
    data = (line.strip() for line in lines if line)
    data = groupby(data, bool)

    return (parse_sentence(g) for k, g in data if k)

def parse_sentence(lines):
    if not lines:
        return

    lines = dropwhile(lambda w: w.startswith('#'), lines)

    return [word.split('\t') for word in lines]

def one_hot(idx, n):
    a = np.zeros(n, dtype=np.float32)
    a[idx] = 1
    return a

class Vocab:

    def __init__(self, fasttext_model, gazetteer):
        print(f"Loading {fasttext_model}", file=sys.stderr)

        self.word_embedding = ft.load_model(fasttext_model)
        self._m = self.word_embedding.get_input_matrix()
        self.n_words = self.word_embedding.get_dimension()

        self.idx_pos = [
            'X', 'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ',
            'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ',
            'SYM', 'VERB'
        ]
        self.pos_idx = { p:idx for idx, p in enumerate(self.idx_pos) }
        self.n_pos = len(self.idx_pos)

        self.idx_char =  ['<UNK>', '<PAD>'] + list('''
         !"#$%&\'()*+,--./0123456789:;<=>?@
        ABCDEFGHIJKLMNOPQRSTUVWXYZÆØÅ
        [\\\\]^_`
        abcdefghijklmnopqrstuvwxyzæøå
        {|}~¡¢£¤¥¦§¨©ª«¬®¯°´µ¶·¸»¿
        ÀÁÂÃÄÇÈÉÊËÌÍÎÏÐÒÓÔÕÖÚÛÜÝÞßàáâãäçèéêëìíîïðñòóôõö÷øùúûüýþÿĆřŠšžȈ
        '''.replace('\n', ''))
        self.char_idx = { char:idx for idx, char in enumerate(self.idx_char) }
        self.n_chars = len(self.idx_char)

        self.idx_tag = [
            '<UNKT>', 'O',
            'B-GEO', 'B-ORG', 'B-OTH', 'B-PRS',
            'E-GEO', 'E-ORG', 'E-OTH', 'E-PRS',
            'I-GEO', 'I-ORG', 'I-OTH', 'I-PRS',
            'S-GEO', 'S-ORG', 'S-OTH', 'S-PRS',
        ]
        self.tag_idx = { tag:idx for idx, tag in enumerate(self.idx_tag) }
        self.n_tags = len(self.idx_tag)

        self.gazetteer = {"GEO" : set(), "ORG" : set(), "OTH" : set(), "PRS" : set()}
        with open(gazetteer) as f:
            for line in f:
                part, category = line.strip().split('\t')
                self.gazetteer[category].add(part)
        self.n_categories = len(self.gazetteer)

        self.shape = (self.n_words, self.n_pos, self.n_categories, self.n_chars, self.n_tags)


    def sentences(self, lines):
        for sentence in parse_lines(lines):
            words = []
            pos = []
            tags = []
            characters = []
            gazetteer = []
            for idx, word, _, p, tag in sentence:

                char_id = [self.char_idx.get(c, 0) for c in word]

                word = word.lower()
                _, subwords = self.word_embedding.get_subwords(word)
                if 0 < len(subwords):
                    word_id = sum(self._m[subwords])
                else:
                    word_id = np.zeros(self.n_words, dtype = np.float32)
                words.append(word_id)

                pos_id = one_hot(self.pos_idx.get(p, 0), self.n_pos)
                pos.append(pos_id)

                tag_id = self.tag_idx.get(tag, 0)
                tags.append(tag_id)


                characters.append(char_id)

                gazetteer.append(
                    [1 if word in category else 0 for category in self.gazetteer.values()]
                )

            lengths = list(map(len, characters))
            chars = np.zeros((len(words), max(lengths)), dtype = np.int32)
            for i, char in enumerate(characters):
                chars[i, :len(char)] = char

            yield (
                np.stack(words),
                np.stack(pos),
                np.stack(gazetteer),
                chars,
                lengths,
                np.stack(tags)
            )


    def examples(self, name_file):
        def gen():
            with open(name_file) as f:
                lines = f.readlines()
                for words, pos, gazetteer, chars, lengths, labels in self.sentences(lines):
                    yield words, pos, gazetteer, chars, lengths, labels, len(labels)
        return gen


    def chunk(self, tag_seq):
        tags = [self.idx_tag[idx] if idx < len(tag_seq) else '<UNKT>' for idx in tag_seq]

        chunks = []
        i = 0
        while i < len(tags):
            start = i
            end = start + 1
            if tags[i].startswith('B') and i+1 < len(tags):
                tag = tags[start][-3:]
                while tags[end] == "I-" + tag:
                    end += 1
                if tags[end] == "E-" + tag:
                    chunks.append((tag, start, end))
                    i = end
            elif tags[i].startswith('S'):
                tag = tags[i][-3:]
                chunks.append((tag, i, i))
            i += 1

        return set(chunks)



if __name__ == '__main__':

    import tensorflow as tf
    import sys

    batch_size = 1

    print('Importing vocab...')
    vocab = Vocab(f'etc/cc.bokmaal.300.bin', 'etc/gazetteer.txt')

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

    test_data = tf.data.Dataset.from_generator(
        vocab.examples(sys.argv[1]),
        output_types = output_types,
        output_shapes = output_shapes
    ).padded_batch(
        batch_size = batch_size,
        padded_shapes = output_shapes
    )

    iterator = tf.data.Iterator.from_structure(
        output_types = output_types,
        output_shapes = test_data.output_shapes
    )

    test_data_init = iterator.make_initializer(test_data)

    batch = iterator.get_next(name = "batch")

    with tf.Session() as session:



        session.run(test_data_init)
        while True:
            val = session.run(batch)
            print(val)
