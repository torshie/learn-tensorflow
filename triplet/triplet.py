#!/usr/bin/env python3

import random

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class Network:
    def __init__(self):
        self.image = tf.placeholder(shape=[None, 28 * 28],
            dtype=tf.float32)
        self.label = tf.placeholder(shape=[None], dtype=tf.int32)

        self.w0 = tf.Variable(tf.random_normal(shape=[28 * 28, 64], stddev=0.1))
        self.b0 = tf.Variable(tf.zeros(shape=[64]))

        self.w1 = tf.Variable(tf.random_normal(shape=[64, 64], stddev=0.1))
        self.b1 = tf.Variable(tf.zeros(shape=[64]))

        self.w2 = tf.Variable(tf.random_normal(shape=[64, 32], stddev=0.1))
        self.b2 = tf.Variable(tf.zeros(shape=[32]))

        h0 = tf.nn.relu(tf.matmul(self.image, self.w0) + self.b0)
        h1 = tf.nn.relu(tf.matmul(h0, self.w1) + self.b1)

        h2 = tf.matmul(h1, self.w2) + self.b2
        self.embedding = tf.nn.l2_normalize(h2, axis=-1)
        print(self.embedding.shape)

        self.loss = tf.contrib.losses.metric_learning.triplet_semihard_loss(
            self.label, self.embedding)

        optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.0001)

        self.training_ops = optimizer.minimize(self.loss)


def filter_dataset(x, y, valid):
    x_result, y_result = [], []
    for a, b in zip(x, y):
        if b not in valid:
            continue
        x_result.append(a)
        y_result.append(b)

    return np.array(x_result), np.array(y_result)


def calc_center(net, dataset, session):
    result = []
    for i in range(10):
        x, y = filter_dataset(*dataset, [i])
        tmp = session.run(net.embedding, feed_dict={net.image: x})
        result.append(np.mean(tmp, axis=0))
    return np.array(result)


def get_predict(embedding, center):
    min_dist = None
    result = 0
    for i in range(10):
        dist = np.sum(np.square(embedding - center[i]))
        if min_dist is None or dist < min_dist:
            min_dist = dist
            result = i
    return result


def load_mnist():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape([len(x_train), 28 * 28])
    x_test = x_test.reshape([len(x_test), 28 * 28])

    return (x_train, y_train), (x_test, y_test)


def shuffle_dataset(dataset):
    tmp = list(zip(*dataset))
    random.shuffle(tmp)
    x, y = zip(*tmp)
    return np.array(x), np.array(y)


def main():
    training_set, test_set = load_mnist()
    partial_training = filter_dataset(*training_set, {0, 1, 2, 3, 4, 5, 6, 7, 8})
    partial_test = filter_dataset(*test_set, [9])

    net = Network()
    init = tf.global_variables_initializer()
    session = tf.Session()
    session.run(init)
    batch_size = 100

    for epoch in range(1):
        x_train, y_train = shuffle_dataset(partial_training)
        cost = 0.0
        for j in range(0, len(x_train), batch_size):
            batch_x, batch_y = x_train[j:j+batch_size], y_train[j:j+batch_size]
            _, c = session.run([net.training_ops, net.loss],
                feed_dict={net.image: batch_x, net.label: batch_y})
            cost += c
        print("epoch %s, cost %s" % (epoch, cost))

    center = calc_center(net, training_set, session)
    print("center shape: %s" % (center.shape,))

    error = 0
    x_test, _ = partial_test
    encoding = session.run(net.embedding, feed_dict={net.image: x_test})
    print("Total: %s" % len(encoding))
    for v in encoding:
        if get_predict(v, center) != 9:
            error += 1
    print("Wrong prediction: %s, %s" % (error, error / len(encoding)))


if __name__ == '__main__':
    main()
