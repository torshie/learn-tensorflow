#!/usr/bin/env python3

import argparse
import pickle

import tensorflow as tf
import numpy as np


def parse_cmdline():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    return p.parse_args()


def convert(dataset):
    x, y = [], []
    for a, b, c in dataset:
        vec = np.zeros([len(a) * 2], dtype=np.float32)
        vec[:len(a)] = a
        vec[len(a):] = b
        x.append(vec)
        y.append(c)

    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)


def load_dataset(filename):
    with open(filename, 'rb') as f:
        x = pickle.load(f)
        train, dev, test = x['train'], x['dev'], x['test']

    return tuple(map(convert, (train, dev, test)))


def main():
    cmdline = parse_cmdline()

    train, dev, test = load_dataset(cmdline.dataset)
    size = len(train[0][0])
    print("Input X size: ", size)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(size, activation=tf.nn.relu),
        tf.keras.layers.Dense(100, activation=tf.nn.relu),
        tf.keras.layers.Dense(2, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    model.fit(train[0], train[1], epochs=5)
    model.evaluate(test[0], test[1])


if __name__ == '__main__':
    main()
