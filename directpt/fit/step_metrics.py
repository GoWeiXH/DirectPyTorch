from torch.tensor import Tensor

from ..metrics import binary_correct, multi_class_correct
from ..metrics import confusion_matrix, recall_precision_fscore


class StepMetrics:

    def __init__(self, multi, threshold, device):
        self.multi = multi
        self.threshold = threshold
        self.device = device
        self.correct_func = multi_class_correct if multi else binary_correct

        self.metric_func_lib = {
            'acc': self.train_acc_step,
            'loss': None,
            'recall': self.train_recall_step,
            'precision': self.train_precision_step,
            'val_acc': self.test_acc_step,
            'val_loss': None,
            'val_recall': self.test_recall_step,
            'val_precision': self.test_precision_step
        }

    def train_acc_step(self, y_pre: Tensor, batch_y: Tensor) -> Tensor:
        step_correct = self.correct_func(y_pre, batch_y, threshold=self.threshold, device=self.device)
        step_acc = step_correct / len(batch_y)
        return step_acc

    def train_recall_step(self, y_pre: Tensor, batch_y: Tensor) -> Tensor:
        recall, *_ = recall_precision_fscore(y_pre, batch_y, self.multi, self.threshold, device=self.device)
        return recall

    def train_precision_step(self, y_pre: Tensor, batch_y: Tensor) -> Tensor:
        _, precision, _ = recall_precision_fscore(y_pre, batch_y, self.multi, self.threshold, device=self.device)
        return precision

    def test_acc_step(self, y_pre: Tensor, batch_y: Tensor) -> (Tensor, int):
        step_acc = self.correct_func(y_pre, batch_y, threshold=self.threshold, device=self.device)
        return step_acc, len(batch_y)

    def test_recall_step(self, y_pre: Tensor, batch_y: Tensor) -> (Tensor, Tensor):
        tp, fn, fp, tn = confusion_matrix(y_pre, batch_y, self.threshold, self.multi, device=self.device)
        return tp, (tp + fn)

    def test_precision_step(self, y_pre: Tensor, batch_y: Tensor) -> (Tensor, Tensor):
        tp, fn, fp, tn = confusion_matrix(y_pre, batch_y, self.threshold, self.multi, device=self.device)
        return tp, (tp + fp)
