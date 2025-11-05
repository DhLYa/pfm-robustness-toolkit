# This file has been modified from the original reetoolbox project.
# Additional changes by Dhyey Yajnik, 2025.

from sklearn.metrics import accuracy_score
import torch
import numpy as np


def accuracy(results):
    labels = results["labels"]
    outputs = results["outputs"]
    _, predictions = torch.max(outputs, 1)
    return accuracy_score(labels, predictions)


def adversarial_accuracy(results):
    labels = results["labels"]
    outputs = results["adversarial_outputs"]
    _, predictions = torch.max(outputs, 1)
    return accuracy_score(labels, predictions)


def rmse(a):
    return torch.sqrt(torch.mean(torch.pow(a, 2)))


def input_sensitivity(results):
    outputs = torch.exp(results["outputs"])
    adv_outputs = torch.exp(results["adversarial_outputs"])
    mean_out_diff = torch.mean(torch.abs(outputs - adv_outputs))
    return mean_out_diff.item()


def normalised_input_sensitivity(results):
    in_sens = input_sensitivity(results)

    pert_measures = results["perturbation_measures"]
    avg_pert = torch.mean(pert_measures)

    norm_in_sens = in_sens / avg_pert
    return norm_in_sens.item()


def fooling_ratio(results):
    acc = accuracy(results)
    adv_acc = adversarial_accuracy(results)
    return (acc - adv_acc) / acc


def fooling_rate(results):
    return fooling_ratio(results) * 100

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize
import torch

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize
import torch
import numpy as np

def roc_auc(results):
    labels = results["labels"].cpu().numpy()
    adv_probs = torch.softmax(results["outputs"], dim=1).cpu().numpy()
    n_classes = adv_probs.shape[1]

    # if only one class in labels, undefined
    unique = np.unique(labels)
    if unique.size < 2:
        return float("nan")

    if n_classes == 2:
        # binary: use p(class=1)
        return roc_auc_score(labels, adv_probs[:, 1])
    # multiclass: one-vs-rest macro
    y_true = label_binarize(labels, classes=range(n_classes))
    return roc_auc_score(y_true, adv_probs, average="macro", multi_class="ovr")

def pr_auc(results):
    labels = results["labels"].cpu().numpy()
    adv_probs = torch.softmax(results["outputs"], dim=1).cpu().numpy()
    n_classes = adv_probs.shape[1]

    unique = np.unique(labels)
    if unique.size < 2:
        return float("nan")

    if n_classes == 2:
        return average_precision_score(labels, adv_probs[:, 1])
    y_true = label_binarize(labels, classes=range(n_classes))
    return average_precision_score(y_true, adv_probs, average="macro")

def adversarial_roc_auc(results):
    labels = results["labels"].cpu().numpy()
    adv_probs = torch.softmax(results["adversarial_outputs"], dim=1).cpu().numpy()
    n_classes = adv_probs.shape[1]

    # if only one class in labels, undefined
    unique = np.unique(labels)
    if unique.size < 2:
        return float("nan")

    if n_classes == 2:
        # binary: use p(class=1)
        return roc_auc_score(labels, adv_probs[:, 1])
    # multiclass: one-vs-rest macro
    y_true = label_binarize(labels, classes=range(n_classes))
    return roc_auc_score(y_true, adv_probs, average="macro", multi_class="ovr")

def adversarial_pr_auc(results):
    labels = results["labels"].cpu().numpy()
    adv_probs = torch.softmax(results["adversarial_outputs"], dim=1).cpu().numpy()
    n_classes = adv_probs.shape[1]

    unique = np.unique(labels)
    if unique.size < 2:
        return float("nan")

    if n_classes == 2:
        return average_precision_score(labels, adv_probs[:, 1])
    y_true = label_binarize(labels, classes=range(n_classes))
    return average_precision_score(y_true, adv_probs, average="macro")

def evaluate_clean(results):
    acc = accuracy(results)
    roc = roc_auc(results)
    pr = pr_auc(results)

    return acc, roc, pr

def get_metrics(results):
    acc             = accuracy(results)
    roc             = roc_auc(results)
    pr              = pr_auc(results)
    robust_acc      = adversarial_accuracy(results)
    fool_ratio      = fooling_ratio(results)
    input_sens      = input_sensitivity(results)
    ad_roc          = adversarial_roc_auc(results)
    ad_pr           = adversarial_pr_auc(results)

    # print(
    #     f"Accuracy: {acc:.3f}, Robust Acc: {robust_acc:.3f}, "
    #     f"Fooling Ratio: {fool_ratio:.3f}, Input Sens: {input_sens:.3f}, "
    #     f"ROC AUC: {roc:.3f}, PR AUC: {pr:.3f}"
    # )
    return acc, roc, pr, robust_acc, fool_ratio, input_sens, ad_roc, ad_pr
