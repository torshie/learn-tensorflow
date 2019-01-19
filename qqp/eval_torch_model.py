#!/usr/bin/env python3

import argparse
import pickle

import torch

from torch_model import SentenceEncoder
from WordVector import WordVector

from torch_model import evaluate, make_data_loader, create_encoder


def parse_cmdline():
    p = argparse.ArgumentParser()
    p.add_argument('--w2v', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--model', required=True)
    return p.parse_args()


def main():
    cmdline = parse_cmdline()

    with open(cmdline.dataset, 'rb') as f:
        dataset = pickle.load(f)

    loader = make_data_loader(dataset['test'], False)

    word2vec = WordVector()
    word2vec.load_bin(cmdline.w2v)
    encoder = create_encoder(word2vec)
    encoder.load_state_dict(torch.load(cmdline.model))

    r = evaluate(loader, encoder, None)
    print(r)


if __name__ == '__main__':
    main()
