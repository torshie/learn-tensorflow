import zipfile
import multiprocessing
import pickle

import numpy as np


def _parse_line(i, line):
    tokens = line.rstrip().split(b' ')
    word = tokens[0].decode('utf-8').strip()
    vector = np.array(list(map(float, tokens[1:])), dtype=np.float32)
    return i, word, vector


class WordVector:
    def __init__(self):
        self.__vocab = 0
        self.__dim = 0
        self.__dictionary = {}
        self.__weight = np.array([0], dtype=np.float32)

    def load_zip(self, zipname, filename):
        zipped = zipfile.ZipFile(zipname)

        pool = multiprocessing.Pool()
        with zipped.open(filename, 'r') as f:
            self.__vocab, self.__dim = map(int, f.readline().split())
            self.__weight = np.zeros([(self.__vocab + 1), self.__dim],
                dtype=np.float32)
            for i, word, vec in \
                    pool.starmap(_parse_line, enumerate(f), chunksize=100):
                self.__dictionary[word] = i + 1
                self.__weight[i + 1][:] = vec
        pool.close()

        zipped.close()

    def load_bin(self, filename):
        with open(filename, 'rb') as f:
            self.__vocab, self.__dim, self.__dictionary, self.__weight = \
                pickle.load(f)

    def save(self, filename):
        with open(filename, 'wb') as f:
            r = (self.__vocab, self.__dim, self.__dictionary, self.__weight)
            pickle.dump(r, f)

    def to_id(self, word):
        r = self.__dictionary.get(word, None)
        if r:
            return r
        return self.__dictionary.get(word.lower(), 0)

    @property
    def dim(self):
        return self.__dim

    @property
    def dictionary(self):
        return self.__dictionary

    @property
    def weight(self):
        return self.__weight

    @property
    def vocab(self):
        return self.__vocab
