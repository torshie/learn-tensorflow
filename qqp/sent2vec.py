#!/usr/bin/env python3

import argparse
import zipfile
import pickle
import os.path

import numpy as np
import tensorflow as tf


def parse_cmdline():
    p = argparse.ArgumentParser()
    p.add_argument('--vector', required=False)
    p.add_argument('--qqp', required=True)
    p.add_argument('--bin', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--unknown', required=True)
    return p.parse_args()


def load_word2vec(filename):
    zipped = zipfile.ZipFile(filename)

    with zipped.open('crawl-300d-2M.vec', 'r') as f:
        size, dim = map(int, f.readline().split())
        dictionary = {}
        vector = np.zeros([size, dim], dtype=np.float32)
        for i, line in enumerate(f):
            tokens = line.rstrip().split(b' ')
            dictionary[tokens[0].decode('utf-8')] = i
            vector[i][:] = list(map(float, tokens[1:]))

    zipped.close()

    return dictionary, vector


def strip_quote(sentence):
    if len(sentence) <= 1:
        return sentence
    if not (sentence[0] == '"' and sentence[-1] == '"'):
        return sentence
    p = sentence[1:-1]
    return p.replace('""', '"')


def load_question_pair(filename):
    with open(filename) as f:
        result = []
        for i, line in enumerate(f):
            if i == 0:
                continue
            x = line.rstrip().split('\t')
            if len(x) != 6:
                continue
            result.append((strip_quote(x[3]), strip_quote(x[4]), int(x[5])))

    length = len(result)
    train = int(length * 0.75)
    dev = int(length * 0.05)

    return result[:train], result[train:train+dev], result[train+dev:]


def cleanup_word(w):
    if len(w) > 1 and (w.startswith("'") or w.startswith('“')):
        w = w[1:]
    if len(w) > 1 and (w.endswith("'") or w.endswith('”')):
        w = w[:-1]
    return w


def word2vec(dictionary, vector, unknown, word):
    if word not in dictionary:
        word = cleanup_word(word)
    offset = dictionary.get(word, None)
    if offset is None:
        unknown.add(word)
        return np.zeros([vector.shape[1]], dtype=np.float32)

    return vector[offset]


def sent2vec(dictionary, vector, unknown, sentence):
    dim = vector.shape[1]
    result = np.zeros([dim * 2], dtype=np.float32)
    words = tf.keras.preprocessing.text.text_to_word_sequence(sentence)
    if len(words) == 0:
        return result

    if len(words) == 1:
        words = words * 2

    previous = word2vec(dictionary, vector, unknown, words[0])
    for i in range(1, len(words) - 1):
        current = word2vec(dictionary, vector, unknown, words[i])
        result += np.concatenate([previous, current])
        previous = current
    return result


def dataset_sent2vec(dictionary, vector, unknown, dataset):
    result = []
    for i, d in enumerate(dataset):
        result.append((
            sent2vec(dictionary, vector, unknown, d[0]),
            sent2vec(dictionary, vector, unknown, d[1]),
            d[2],
            d[0], d[1]))
        if i % 10000 == 0:
            print(i)
    return result


def main():
    cmdline = parse_cmdline()
    if os.path.isfile(cmdline.bin):
        with open(cmdline.bin, 'rb') as f:
            dictionary, vector = pickle.load(f)
    else:
        if not cmdline.vector:
            raise Exception("Wrong")
        dictionary, vector = load_word2vec(cmdline.vector)
        with open(cmdline.bin, 'wb') as f:
            pickle.dump((dictionary, vector), f)

    unknown = set()

    print("Dictionary size:", len(dictionary))
    train, dev, test = load_question_pair(cmdline.qqp)
    print("Question pair size:", len(train), len(dev), len(test))

    result = {
        "train": dataset_sent2vec(dictionary, vector, unknown, train),
        "test": dataset_sent2vec(dictionary, vector, unknown, test),
        "dev": dataset_sent2vec(dictionary, vector, unknown, dev)
    }
    with open(cmdline.output, 'wb') as f:
        pickle.dump(result, f)
    with open(cmdline.unknown, 'wb') as f:
        pickle.dump(unknown, f)


if __name__ == '__main__':
    main()
