import torch

from ..metrics import binary_correct, multi_class_correct
from ..metrics import recall_precision_fscore


class StepMetrics:

    def __init__(self, multi, threshold):
        self.multi = multi
        self.threshold = threshold
        self.correct_func = multi_class_correct if multi else binary_correct

        self.metric_func_lib = {
            'loss': None,
            'val_loss': None,
            'acc': self.train_acc_step,
            'val_acc': self.test_acc_step,
            'val_recall': self.train_recall_step,
        }

    def train_acc_step(self, model, batch_x: torch.Tensor, batch_y: torch.Tensor) -> float:
        step_correct = self.correct_func(model(batch_x), batch_y, threshold=self.threshold)
        step_acc = step_correct / len(batch_y)
        return step_acc

    def test_acc_step(self, model, batch_x: torch.Tensor, batch_y: torch.Tensor) -> float:
        step_acc = self.correct_func(model(batch_x), batch_y, threshold=self.threshold)
        return step_acc

    def train_recall_step(self, model, batch_x: torch.Tensor, batch_y: torch.Tensor,
                          zero_division=0) -> float:
        y_pre = model(batch_x)
        recall, *_ = recall_precision_fscore(y_pre, batch_y, self.multi, self.threshold, zero_division=zero_division)
        return recall
