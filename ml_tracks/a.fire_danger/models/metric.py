import torch
import torch.nn as nn


def accuracy(pred, target):
    with torch.no_grad():
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct, len(target)


def precision(output, labels):
    true_positives_fire = 0
    false_positives_fire = 0
    for j in range(output.size()[0]):
        if output[j] == 1 and labels[j] == 1:
            true_positives_fire += 1
        if output[j] == 1 and labels[j] == 0:
            false_positives_fire += 1
    if false_positives_fire + true_positives_fire == 0:
        true_positives_fire += 1
    return true_positives_fire, false_positives_fire + true_positives_fire


def recall(output, labels):
    true_positives_fire = 0
    false_negatives_fire = 0
    for j in range(output.size()[0]):
        if output[j] == 1 and labels[j] == 1:
            true_positives_fire += 1
        if output[j] == 0 and labels[j] == 1:
            false_negatives_fire += 1
    if false_negatives_fire + true_positives_fire == 0:
        true_positives_fire += 1
    return true_positives_fire, false_negatives_fire + true_positives_fire


def f1_score(output, labels):
    true_positives_fire = 0
    false_negatives_fire = 0
    false_positives_fire = 0
    for j in range(output.size()[0]):
        if output[j] == 1 and labels[j] == 1:
            true_positives_fire += 1
        if output[j] == 0 and labels[j] == 1:
            false_negatives_fire += 1
        if output[j] == 1 and labels[j] == 0:
            false_positives_fire += 1
    return true_positives_fire, true_positives_fire + (1/2)*(false_positives_fire + false_negatives_fire)


def aucpr(preds, labels):
    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    return preds, labels
