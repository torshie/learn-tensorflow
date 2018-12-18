#!/usr/bin/env python3

import random

import tensorflow as tf
import numpy as np


def sort_by_label(image, label):
    tmp = {}
    for i in range(10):
        tmp[i] = []

    for x, y in zip(image, label):
        tmp[y].append(x.reshape([28 * 28]))

    result = {}
    for k, v in tmp.items():
        result[k] = np.array(v, dtype=np.float32)
    return result


def load_mnist():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    train = sort_by_label(x_train, y_train)
    test = sort_by_label(x_test, y_test)

    return train, test


MARGIN = 1.0

anchor_placeholder = tf.placeholder(shape=[None, 28 * 28], dtype=tf.float32)
positive_placeholder = tf.placeholder(shape=[None, 28 * 28], dtype=tf.float32)
negative_placeholder = tf.placeholder(shape=[None, 28 * 28], dtype=tf.float32)
center_placeholder = tf.placeholder(shape=[10, 64], dtype=tf.float32)


w0 = tf.Variable(tf.random_normal(shape=[28 * 28, 64], stddev=0.1))
b0 = tf.Variable(tf.zeros(shape=[64]))

w1 = tf.Variable(tf.random_normal(shape=[64, 64], stddev=0.1))
b1 = tf.Variable(tf.zeros(shape=[64]))

w2 = tf.Variable(tf.random_normal(shape=[64, 64], stddev=0.1))
b2 = tf.Variable(tf.zeros(shape=[64]))


def encode_image(input_tensor):
    h0 = tf.nn.relu(tf.matmul(input_tensor, w0) + b0)
    h1 = tf.nn.relu(tf.matmul(h0, w1) + b1)

    h2 = tf.matmul(h1, w2) + b2

    return tf.nn.l2_normalize(h2, axis=-1)


def sample_triplet(dataset, size):
    labels = list(range(9))
    anchor_image = []
    positive_image = []
    negative_image = []
    for i in range(size):
        positive_label, negative_label = random.choices(labels, k=2)
        a, p = random.choices(dataset[positive_label], k=2)
        n = random.choice(dataset[negative_label])
        anchor_image.append(a)
        positive_image.append(p)
        negative_image.append(n)
    return np.array(anchor_image), np.array(positive_image), np.array(negative_image)


anchor_output = encode_image(anchor_placeholder)
positive_output = encode_image(positive_placeholder)
negative_output = encode_image(negative_placeholder)

d_pos = tf.reduce_sum(tf.square(anchor_output - positive_output), axis=-1)
d_neg = tf.reduce_sum(tf.square(anchor_output - negative_output), axis=-1)

triplet_loss = tf.maximum(0.0, MARGIN + d_pos - d_neg)
regularizer = tf.nn.l2_loss(w0) + tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)
loss = tf.reduce_mean(triplet_loss) + 0.00001 * regularizer

optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)
train_ops = optimizer.minimize(loss)

init = tf.global_variables_initializer()


def calc_center(dataset, session):
    result = []
    for i in range(10):
        tmp = session.run(anchor_output, feed_dict={anchor_placeholder: dataset[i]})
        result.append(np.mean(tmp, axis=0))
    return np.array(result)


def calc_encoding(dataset, session):
    result = []
    for i in range(10):
        tmp = session.run(anchor_output, feed_dict={anchor_placeholder: dataset[i]})
        result.append(tmp)
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


def main():
    session = tf.Session()
    session.run(init)

    training_set, test_set = load_mnist()
    batch_size = 100

    cost = 0.0
    for i in range(5):
        a, p, n = sample_triplet(training_set, batch_size)

        _, c = session.run([train_ops, loss],
            feed_dict={
                anchor_placeholder: a,
                positive_placeholder: p,
                negative_placeholder: n}
            )
        cost += c / batch_size
        if i % 1000 == 0:
            print('%d, cost: %s' % (i, cost))
            cost = 0.0

    center = calc_center(training_set, session)
    print("shape %s, value: %s" % (center[0].shape, center[0]))

    error = 0
    test_encoding = calc_encoding(test_set, session)
    for embedding in test_encoding[9]:
        if get_predict(embedding, center) != 9:
            error += 1
    print("Wrong prediction: %s, %s" % (error, error / len(test_encoding[9])))


if __name__ == '__main__':
    main()
