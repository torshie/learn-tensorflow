#!/usr/bin/env python3

import argparse
import os.path

from WordVector import WordVector


def parse_cmdline():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    return p.parse_args()


def main():
    cmdline = parse_cmdline()
    w2v = WordVector()

    base = os.path.basename(cmdline.input)
    base = base.replace('.zip', '')
    w2v.load_zip(cmdline.input, base)
    w2v.save(cmdline.output)


if __name__ == '__main__':
    main()
