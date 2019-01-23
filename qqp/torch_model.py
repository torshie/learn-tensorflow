#!/usr/bin/env python3

import argparse
import pickle
import time
import itertools
import os.path
import shutil

import torch
import torch.nn.functional
import torch.optim
import torch.utils.data

import tensorboardX


from WordVector import WordVector


margin = None
log_writer = None


class SentenceEncoder(torch.nn.Module):
    def __init__(self, vocab, dim, kernel):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab, dim, sparse=True)

        tmp = []
        for feature in kernel:
            tmp.append(torch.nn.Conv1d(dim, 300, feature))
        self.convolution = torch.nn.ModuleList(tmp)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(300 * len(kernel), 128)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = x.cuda()

        tmp = [
            torch.nn.functional.relu(c(x).max(-1)[0])
            for c in self.convolution
        ]
        x = torch.cat(tmp, 1)
        x = self.dropout(x)
        return torch.nn.functional.normalize(self.fc1(x))

    def cuda(self, device=None):
        self.convolution.cuda(device)
        self.fc1.cuda(device)
        self.dropout.cuda(device)

    def dense_parameters(self):
        return itertools.chain(self.convolution.parameters(),
            self.dropout.parameters(), self.fc1.parameters())

    def sparse_parameters(self):
        return self.embedding.parameters()


def parse_cmdline():
    p = argparse.ArgumentParser()
    p.add_argument('--w2v', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--log', required=False, default='log')
    p.add_argument('--margin', required=False, default=0.95, type=float)
    return p.parse_args()


def make_data_loader(ds, shuffle):
    result = []
    for _, (x0, x1, y) in ds.items():
        # Make the label suitable for torch.nn.CosineEmbeddingLoss
        y = torch.cuda.FloatTensor(y)
        y = y * 2.0 - 1.0

        tmp = torch.utils.data.TensorDataset(torch.LongTensor(x0),
            torch.LongTensor(x1), y)
        loader = torch.utils.data.DataLoader(tmp, shuffle=shuffle,
            batch_size=128)
        result.append(loader)
    return result


def eval_output(r0, r1, target):
    predict = torch.nn.functional.cosine_similarity(r0, r1)
    correct = 0
    m = margin
    for y_, y in zip(predict.cpu(), target.cpu()):
        if y_ < m and y < 0:
            correct += 1
            continue
        if y_ > m and y > 0:
            correct += 1
            continue
    return correct, predict.sum()


def evaluate(loader, model, step, quiet=False):
    start = time.time()
    with torch.no_grad():
        total = 0
        correct = 0
        model.eval()
        total_loss = 0
        output_sum = 0
        for x0, x1, expected in itertools.chain(*loader):
            total += len(expected)
            r0 = model(x0)
            r1 = model(x1)
            total_loss += torch.nn.functional.cosine_embedding_loss(r0, r1,
                expected, margin=margin, reduction='sum')
            n, s = eval_output(r0, r1, expected)
            correct += n
            output_sum += s

    if step is not None:
        log_writer.add_scalars('loss', {'dev': total_loss / total}, step)
        log_writer.add_scalars('accuracy', {'dev': correct / total}, step)
    if not quiet:
        print("Eval() time: ", time.time() - start)
        print("Output mean: ", output_sum / total)
    return total, correct, correct / total


def create_encoder(word2vec):
    model = SentenceEncoder(len(word2vec.weight), word2vec.dim, [1, 2, 3, 4, 5])
    model.cuda()
    model.embedding.weight.data = torch.Tensor(word2vec.weight)
    return model


def check_dataset(*ds):
    for d in ds:
        positive = 0
        total = 0
        for a, b, y in itertools.chain(*d):
            total += len(y)
            for v in y:
                if v > 0:
                    positive += 1
        print(f"Total {total}, positive {positive}")


def main():
    cmdline = parse_cmdline()
    global margin
    margin = cmdline.margin

    word2vec = WordVector()
    print("Loading word2vec")
    word2vec.load_bin(cmdline.w2v)
    with open(cmdline.dataset, 'rb') as f:
        dataset = pickle.load(f)
        train_loader = make_data_loader(dataset['train'], True)
        test_loader = make_data_loader(dataset['test'], False)
        dev_loader = make_data_loader(dataset['dev'], False)

    check_dataset(train_loader, dev_loader, test_loader)

    print("Building model")
    model = create_encoder(word2vec)

    r = evaluate(dev_loader, model, None)
    print("Initial dev set evaluation:", r)

    global log_writer
    if os.path.isdir(cmdline.log):
        shutil.rmtree(cmdline.log)
    log_writer = tensorboardX.SummaryWriter(cmdline.log)

    sparse_optimizer = torch.optim.SparseAdam(model.sparse_parameters())
    dense_optimizer = torch.optim.Adam(model.dense_parameters())
    criterion = torch.nn.CosineEmbeddingLoss(margin=margin)

    epoch = 15

    for e in range(epoch):
        print("Epoch %d ===============" % e)
        total_loss = 0.0
        sample = 0

        start = time.time()
        model.train()
        correct = 0
        for x0, x1, y in itertools.chain(*train_loader):
            sample += len(y)
            r0 = model(x0)
            r1 = model(x1)
            loss = criterion(r0, r1, y)

            dense_optimizer.zero_grad()
            sparse_optimizer.zero_grad()
            loss.backward()
            dense_optimizer.step()
            sparse_optimizer.step()

            total_loss += loss.item() * len(y)
            correct += eval_output(r0, r1, y)[0]

        print("Epoch size:", sample, "average loss:", total_loss / sample,
            "time", time.time() - start)
        log_writer.add_scalars('loss', {'train': total_loss / sample}, e + 1)
        print("Training accuracy", correct / sample)
        log_writer.add_scalars('accuracy', {'train': correct / sample}, e + 1)
        r = evaluate(dev_loader, model, e + 1)
        print("Dev set evaluation", r)

    print("Searching for best margin")
    end = 0.999
    best_accuracy = 0
    best_margin = 0
    while margin <= end:
        r = evaluate(dev_loader, model, None, quiet=True)
        if r[2] > best_accuracy:
            best_accuracy = r[2]
            best_margin = margin
            print("Updating margin to", best_margin, "accuracy", best_accuracy)
        margin += 0.001
    print("Best accuracy:", best_accuracy, "corresponding margin", best_margin)

    margin = best_margin
    r = evaluate(test_loader, model, None)
    print("Test set evaluation:", r)

    print("Saving model to", cmdline.model)
    torch.save(model.state_dict(), cmdline.model)


if __name__ == '__main__':
    main()
