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
    x, y, q1, q2 = [], [], [], []
    for a, b, c, d, e in dataset:
        vec = np.zeros([len(a) * 2], dtype=np.float32)
        vec[:len(a)] = a
        vec[len(a):] = b
        x.append(vec)
        y.append(c)
        q1.append(d)
        q2.append(e)

    return np.array(x, dtype=np.float32), np.array(y, dtype=np.int32), q1, q2


def load_dataset(filename):
    with open(filename, 'rb') as f:
        x = pickle.load(f)
        train, dev, test = x['train'], x['dev'], x['test']

    return tuple(map(convert, (train, dev, test)))


def main():
    cmdline = parse_cmdline()

    train, dev, test = load_dataset(cmdline.dataset)
    size = len(train[0][0])
    print("Input X shape: ", train[0].shape)
    print("Input Y shape: ", train[1].shape)
    print("Test X shape: ", test[0].shape)
    print("Test Y shape: ", test[1].shape)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(size, activation=tf.nn.relu),
        tf.keras.layers.Dense(100, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    for i in range(10):
        model.fit(train[0], train[1], epochs=1)
        print("Evaluating on dev set ...")
        r = model.evaluate(dev[0], dev[1])
        print(r)

    print("Evaluating model ...")
    r = model.evaluate(test[0], test[1])
    print(r)


if __name__ == '__main__':
    main()
