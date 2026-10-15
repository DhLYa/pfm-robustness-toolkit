# Derived from the original reetoolbox project (Apache License 2.0).
# Modified by Dhyey Yajnik, 2025.
#
# Every metric takes the `results` dict returned by Evaluator.predict(). Model outputs
# are log-probabilities (heads end in LogSoftmax).

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.preprocessing import label_binarize


def accuracy(results):
    _, predictions = torch.max(results["outputs"], 1)
    return accuracy_score(results["labels"], predictions)


def adversarial_accuracy(results):
    _, predictions = torch.max(results["adversarial_outputs"], 1)
    return accuracy_score(results["labels"], predictions)


def input_sensitivity(results):
    outputs = torch.exp(results["outputs"])
    adv_outputs = torch.exp(results["adversarial_outputs"])
    mean_out_diff = torch.mean(torch.abs(outputs - adv_outputs))
    return mean_out_diff.item()


def fooling_ratio(results):
    acc = accuracy(results)
    adv_acc = adversarial_accuracy(results)
    return (acc - adv_acc) / acc


def _auc(results, output_key, scorer, multiclass_kwargs):
    labels = results["labels"].cpu().numpy()
    probs = torch.softmax(results[output_key], dim=1).cpu().numpy()
    n_classes = probs.shape[1]

    # undefined if only one class is present
    if np.unique(labels).size < 2:
        return float("nan")

    if n_classes == 2:
        # binary: use p(class=1)
        return scorer(labels, probs[:, 1])
    y_true = label_binarize(labels, classes=range(n_classes))
    return scorer(y_true, probs, **multiclass_kwargs)


_ROC_MULTICLASS = dict(average="macro", multi_class="ovr")
_PR_MULTICLASS = dict(average="macro")


# Function names matter: sweeps use `metric.__name__` as the result column name.
def roc_auc(results):
    return _auc(results, "outputs", roc_auc_score, _ROC_MULTICLASS)


def pr_auc(results):
    return _auc(results, "outputs", average_precision_score, _PR_MULTICLASS)


def adversarial_roc_auc(results):
    return _auc(results, "adversarial_outputs", roc_auc_score, _ROC_MULTICLASS)


def adversarial_pr_auc(results):
    return _auc(results, "adversarial_outputs", average_precision_score, _PR_MULTICLASS)


def get_metrics(results):
    """Return (acc, roc, pr, robust_acc, fool_ratio, input_sens, adv_roc, adv_pr), in that order."""
    return (
        accuracy(results),
        roc_auc(results),
        pr_auc(results),
        adversarial_accuracy(results),
        fooling_ratio(results),
        input_sensitivity(results),
        adversarial_roc_auc(results),
        adversarial_pr_auc(results),
    )
