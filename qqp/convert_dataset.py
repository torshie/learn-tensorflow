#!/usr/bin/env python3

import argparse
import pickle

import numpy as np
import nltk

from WordVector import WordVector


def parse_cmdline():
    p = argparse.ArgumentParser()
    p.add_argument('--word2vec', required=True)
    p.add_argument('--qqp', required=True)
    p.add_argument('--output', required=True)
    return p.parse_args()


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


def pad(data, size):
    if len(data) < size:
        data.extend([0] * (size - len(data)))


def sentence_to_id(sentence, word2vec):
    result = []
    for w in nltk.word_tokenize(sentence):
        id_ = word2vec.to_id(w)
        if id_ == 0:
            continue
        result.append(id_)
    if len(result) == 0:
        result = [0]
    return result


def split_dataset(word2vec, ds, limit):
    category = {}
    for l in limit:
        category[l] = []
    for a, b, y in ds:
        x0 = sentence_to_id(a, word2vec)
        x1 = sentence_to_id(b, word2vec)
        for l in limit:
            if len(x0) <= l and len(x1) <= l:
                pad(x0, l)
                pad(x1, l)
                category[l].append((x0, x1, y))
                break
        else:
            print(len(x0), len(x1), a, b)
            raise Exception("Whoops")

    for l in limit:
        if len(category[l]) == 0:
            continue
        x0, x1, y = zip(*category[l])
        category[l] = (
            np.array(x0, dtype=np.float32),
            np.array(x1, dtype=np.float32),
            np.array(y, dtype=np.int32)
        )

    return category


def main():
    cmdline = parse_cmdline()
    word2vec = WordVector()
    word2vec.load_bin(cmdline.word2vec)

    print("Dictionary size:", word2vec.vocab, word2vec.dim)
    train, dev, test = load_question_pair(cmdline.qqp)
    print("Question pair size:", len(train), len(dev), len(test))

    limit = [8, 16, 24, 32, 64, 128, 256, 300]
    result = {
        "train": split_dataset(word2vec, train, limit),
        "test": split_dataset(word2vec, test, limit),
        "dev": split_dataset(word2vec, dev, limit)
    }
    with open(cmdline.output, 'wb') as f:
        pickle.dump(result, f)


if __name__ == '__main__':
    main()
