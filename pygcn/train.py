from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from pygcn.utils import load_data, accuracy
from pygcn.models import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--train', type=str, default='training',
                    help='Training dataset.')
parser.add_argument('--test', type=str, default='testing',
                    help='Test dataset, used for prediction.')
parser.add_argument('--path', type=str, default='../data/GCNs/',
                    help='Path where data is stored.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

train_dataset = args.train
test_dataset = args.test
path = args.path

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data(path, train_dataset)
print('number of nodes in training set: ', len(labels))
# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

features, labels = Variable(features), Variable(labels)


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    class_weights = Variable(torch.FloatTensor([0.005, 1]).cuda())
    loss_train = F.nll_loss(output[idx_train], labels[idx_train], weight=class_weights)
    # loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    precision, recall, F1, acc = evaluate(output[idx_train], labels[idx_train])
    print('Epoch: {:04d}'.format(epoch+1),
         'loss_train: {:.4f}'.format(loss_train.data[0]),
         'acc_train: {:.4f}'.format(acc_train.data[0]),
         'loss_val: {:.4f}'.format(loss_val.data[0]),
         'acc_val: {:.4f}'.format(acc_val.data[0]),
         'time: {:.4f}s'.format(time.time() - t),
         'recall: {:4f}s'.format(recall),
         'precision: {:4f}s'.format(precision))


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data[0]),
          "accuracy= {:.4f}".format(acc_test.data[0]))
    precision, recall, F1, acc = evaluate(output[idx_test], labels[idx_test])
#     with open("predictions/test_evaluation.txt", "a") as f:
#         f.write('%s %f %f %f %f\n' % (train_dataset, precision, recall, F1, acc))

def predict():
    model.eval()
    adj, features, labels, _, _, _ = load_data(path, test_dataset)
    print('number of nodes in test set: ', len(features))
    if args.cuda:
        features = features.cuda()
        adj = adj.cuda()
    features, labels = Variable(features), Variable(labels)
    output = model(features, adj)
    acc = accuracy(output, labels)
    precision, recall, F1, acc = evaluate(output, labels)
    with open("predictions/evaluation.txt", "a") as f:
        f.write('%s %s %f %f %f %f\n' % (train_dataset, test_dataset, precision, recall, F1, acc))
#         f.write('%s %s %f\n' % (train_dataset, test_dataset, acc.data[0]))

def evaluate(output, labels):
    preds = output.max(1)[1].type_as(labels)
    TP = ((preds == 1) & (labels == 1)).double().sum().data[0]
    TN = ((preds == 0) & (labels == 0)).double().sum().data[0]
    FP = ((preds == 1) & (labels == 0)).double().sum().data[0]
    FN = ((preds == 0) & (labels == 1)).double().sum().data[0]
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
#     print(TP, FN, FP, TN)
    return precision, recall, F1, acc

# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
# print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()

# Prediction
# predict()
