import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


# ----------------------------------------------------------------------
def auc_mu(y_true, y_score, A=None, W=None):
    """
    Compute the multi-class measure AUC Mu from prediction scores and labels.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        The true class labels in the range [0, n_samples-1]

    y_score : array, shape = [n_samples, n_classes]
        Target scores, where each row is a categorical distribution over the
        n_classes.

    A : array, shape = [n_classes, n_classes], optional
        The partition (or misclassification cost) matrix. If ``None`` A is the
        argmax partition matrix. Entry A_{i,j} is the cost of classifying an
        instance as class i when the true class is j. It is expected that
        diagonal entries in A are zero and off-diagonal entries are positive.

    W : array, shape = [n_classes, n_classes], optional
        The weight matrix for incorporating class skew into AUC Mu. If ``None``,
        the standard AUC Mu is calculated. If W is specified, it is expected to
        be a lower triangular matrix where entrix W_{i,j} is a positive float
        from 0 to 1 for the partial score between classes i and j. Entries not
        in the lower triangular portion of W must be 0 and the sum of all
        entries in W must be 1.

    Returns
    -------
    auc_mu : float

    References
    ----------
    .. [1] Kleiman, R., Page, D. ``AUC Mu: A Performance Metric for Multi-Class
           Machine Learning Models``, Proceedings of the 2019 International
           Conference on Machine Learning (ICML).

    """

    # Validate input arguments
    if not isinstance(y_score, np.ndarray):
        raise TypeError("Expected y_score to be np.ndarray, got: %s" % type(y_score))
    if not y_score.ndim == 2:
        raise ValueError("Expected y_score to be 2 dimensional, got: %s" % y_score.ndim)
    n_samples, n_classes = y_score.shape

    if not isinstance(y_true, np.ndarray):
        raise TypeError("Expected y_true to be np.ndarray, got: %s" % type(y_true))
    if not y_true.ndim == 1:
        raise ValueError("Expected y_true to be 1 dimensional, got: %s" % y_true.ndim)
    if not y_true.shape[0] == n_samples:
        raise ValueError(
            "Expected y_true to be shape %s, got: %s"
            % (str(y_score.shape), str(y_true.shape))
        )
    unique_labels = np.unique(y_true)
    if not np.all(unique_labels == np.arange(n_classes)):
        raise ValueError(
            "Expected y_true values in range 0..%i, got: %s"
            % (n_classes - 1, str(unique_labels))
        )

    if A is None:
        A = np.ones((n_classes, n_classes)) - np.eye(n_classes)
    if not isinstance(A, np.ndarray):
        raise TypeError("Expected A to be np.ndarray, got: %s" % type(A))
    if not A.ndim == 2:
        raise ValueError("Expected A to be 2 dimensional, got: %s" % A.ndim)
    if not A.shape == (n_classes, n_classes):
        raise ValueError(
            "Expected A to be shape (%i, %i), got: %s"
            % (n_classes, n_classes, str(A.shape))
        )
    if not np.all(A.diagonal() == np.zeros(n_classes)):
        raise ValueError("Expected A to be zero on the diagonals")
    if not np.all(A >= 0):
        raise ValueError("Expected A to be non-negative")

    if W is None:
        W = np.tri(n_classes, k=-1)
        W /= W.sum()
    if not isinstance(W, np.ndarray):
        raise TypeError("Expected W to be np.ndarray, got: %s" % type(W))
    if not W.ndim == 2:
        raise ValueError("Expected W to be 2 dimensional, got: %s" % W.ndim)
    if not W.shape == (n_classes, n_classes):
        raise ValueError(
            "Expected W to be shape (%i, %i), got: %s"
            % (n_classes, n_classes, str(W.shape))
        )

    auc_total = 0.0

    for class_i in range(n_classes):
        preds_i = y_score[y_true == class_i]
        n_i = preds_i.shape[0]
        for class_j in range(class_i):
            preds_j = y_score[y_true == class_j]
            temp_preds = np.vstack((preds_i, preds_j))
            n_j = preds_j.shape[0]
            n = n_i + n_j

            temp_labels = np.zeros((n), dtype=int)
            temp_labels[n_i:n] = 1

            v = A[class_i, :] - A[class_j, :]
            scores = np.dot(temp_preds, v)

            score_i_j = roc_auc_score(temp_labels, scores)
            auc_total += W[class_i, class_j] * score_i_j

    return auc_total


def cross_entropy_vectorized(y_true, outputs):
    """
    Parameters
    ----------
    y_true : Pytorch Tensor
        The tensor with actual labaels.
    outputs : Pytorch Tensor
        The predicted values from auxiliary head.

    Returns
    -------
    loss : Pytorch tensor
        The cross entropy loss for auxiliary head.

    """
    n_batch, n_class = outputs.shape
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)
    if isinstance(outputs, np.ndarray):
        outputs = torch.from_numpy(outputs)
    if len(y_true.shape) == 1:
        y_true = torch.unsqueeze(y_true, dim=-1)
    y_true = y_true.to(torch.int64)
    # log_ypred = torch.log(torch.gather(F.softmax(outputs, dim=1), 1, y_true))
    log_ypred = torch.gather(torch.log_softmax(outputs, dim=1), 1, y_true)
    # log_ypred = F.log_softmax()
    loss = -(torch.sum(log_ypred)) / n_batch
    return loss


def cross_entropy_selection_vectorized(y_true, hg, theta=0.5, lamda=32, c=0.9):
    """
    Parameters
    ----------
    y_true : Pytorch Tensor
        The tensor with actual labels.
    hg : Pytorch Tensor
        The outputs of predictive head and selecticve head.
    theta : float, optional
        The threshold to make g(x)=1. The default is .5.
    lamda : float, optional
        Parameter to weigh the importance of constraint for coverage. The default is 32.
    c : float, optional
        The desired coverage. The default is 0.9.

    Returns
    -------
    loss : Pytorch Tensor
        The selective loss from Geifman et al. (2019).

    """
    n_batch, n_class = hg[:, :-1].shape
    if len(y_true.shape) == 1:
        y_true = torch.unsqueeze(y_true, dim=-1)
    if c == 1:
        selected = n_batch
    else:
        selected = torch.sum(hg[:, -1]) + 0.00000001
    selection = torch.unsqueeze(hg[:, -1], dim=-1)
    y_true = y_true.to(torch.int64)
    # log_ypred = (
    #     torch.log(torch.gather(F.softmax(hg[:, :-1], dim=1), 1, y_true)+ 0.00000001) * selection
    # )
    log_ypred = (
        torch.gather(torch.log_softmax(hg[:, :-1], dim=1), 1, y_true) * selection
    )
    loss = (
        -((torch.sum(log_ypred)) / (selected))
        + lamda * (max(0, c - (selected / n_batch))) ** 2
    )

    return loss


def deep_gambler_loss(y_true, outputs, reward):
    outputs = F.softmax(outputs, dim=1)
    outputs, reservation = outputs[:, :-1], outputs[:, -1]
    # gain = torch.gather(outputs, dim=1, index=y_true.unsqueeze(1)).squeeze()
    gain = outputs[torch.arange(y_true.shape[0]), y_true]
    doubling_rate = (gain.add(reservation.div(reward))).log()
    return -doubling_rate.mean()


class SelfAdaptiveTraining:
    def __init__(self, num_examples=50000, num_classes=10, mom=0.9):
        self.prob_history = torch.zeros(num_examples, num_classes)
        self.updated = torch.zeros(num_examples, dtype=torch.int)
        self.mom = mom
        self.num_classes = num_classes

    def _update_prob(self, y_true, prob, index):
        onehot = torch.zeros_like(prob)
        onehot[torch.arange(y_true.shape[0]), y_true] = 1
        prob_history = self.prob_history[index].clone().to(prob.device)

        # if not inited, use onehot label to initialize runnning vector
        cond = (self.updated[index] == 1).to(prob.device).unsqueeze(-1).expand_as(prob)
        prob_mom = torch.where(cond, prob_history, onehot)
        # momentum update
        prob_mom = self.mom * prob_mom + (1 - self.mom) * prob
        self.updated[index] = 1
        self.prob_history[index] = prob_mom.to(self.prob_history.device)
        return prob_mom

    def __call__(self, y_true, logits, index):
        prob = F.softmax(logits.detach()[:, : self.num_classes], dim=1)
        prob = self._update_prob(y_true, prob, index)

        soft_label = torch.zeros_like(logits)
        soft_label[torch.arange(y_true.shape[0]), y_true] = prob[
            torch.arange(y_true.shape[0]), y_true
        ]
        soft_label[:, -1] = 1 - prob[torch.arange(y_true.shape[0]), y_true]
        soft_label = F.normalize(soft_label, dim=1, p=1)
        loss = torch.sum(-F.log_softmax(logits, dim=1) * soft_label, dim=1)
        return torch.mean(loss)


def entropy_loss(output):
    el = F.softmax(output, dim=1) * F.log_softmax(output, dim=1)
    loss = -1.0 * (el.sum())
    return loss


def entropy_term(output):
    softmax = torch.nn.Softmax(-1)
    el = (-softmax(output) * output).sum(-1).mean()
    return el


#
# def cross_entropy_selection_vectorized_BOH(y_true, hg):
#     """
#     Parameters
#     ----------
#     y_true : Pytorch Tensor
#         The tensor with actual labels.
#     hg : Pytorch Tensor
#         The outputs of predictive head and selecticve head.
#     theta : float, optional
#         The threshold to make g(x)=1. The default is .5.
#     lamda : float, optional
#         Parameter to weigh the importance of constraint for coverage. The default is 32.
#     c : float, optional
#         The desired coverage. The default is 0.9.
#
#     Returns
#     -------
#     loss : Pytorch Tensor
#         The selective loss from Geifman et al. (2019).
#
#     """
#     n_batch, n_class = hg[:, :-1].shape
#     if len(y_true.shape) == 1:
#         y_true = torch.unsqueeze(y_true, dim=-1)
#     else:
#         selected = torch.sum(hg[:, -1]) + 0.00000001
#     selection = torch.unsqueeze(hg[:, -1], dim=-1)
#     y_true = y_true.to(torch.int64)
#     # log_ypred = (
#     #     torch.log(torch.gather(F.softmax(hg[:, :-1], dim=1), 1, y_true)+ 0.00000001) * selection
#     # )
#     log_ypred_sel = (
#         torch.gather(torch.log_softmax(hg[:, :-1], dim=1), 1, y_true) * selection
#     )
#     log_ypred = torch.gather(torch.log_softmax(hg[:, :-1], dim=1), 1, y_true)
#     loss = -2 * ((torch.sum(log_ypred_sel)) / (selected)) + (torch.sum(log_ypred)) / (
#         selected
#     )
#
#     return loss
def MSE_confid_loss(y_true, output, num_classes, device, weighting=1):
    probs = F.softmax(output[:, :-1], dim=1)
    confidence = torch.sigmoid(output[:, -1]).squeeze()
    # Apply optional weighting
    weights = torch.ones_like(y_true).type(torch.FloatTensor).to(device)
    weights[(probs.argmax(dim=1) != y_true)] *= weighting
    tmp = torch.eye(num_classes).long().to(device)
    labels_hot = tmp[y_true]
    loss = weights * (confidence - (probs * labels_hot).sum(dim=1)) ** 2
    return torch.mean(loss)


def sele_loss(y_true, output):
    n = len(y_true)

    if len(y_true.shape) == 1:
        y_true = y_true.unsqueeze(dim=-1)
    y_true = y_true.to(torch.int64)

    log_probs = -torch.gather(torch.log_softmax(output[:, :-1], dim=1), 1, y_true)
    s = torch.sigmoid(output[:, -1])
    inner_terms = torch.log(1 + torch.exp(s.view(1, -1) - s.view(-1, 1)))
    loss = torch.sum((log_probs * inner_terms))

    return loss / (n * n)


def reg_loss(y_true, output):
    n = len(y_true)
    if len(y_true.shape) == 1:
        y_true = torch.unsqueeze(y_true, dim=-1)
    y_true = y_true.to(torch.int64)
    log_ypred = torch.gather(torch.log_softmax(output[:, :-1], dim=1), 1, y_true)
    return torch.mean((-log_ypred - (output[:, -1])) ** 2)
