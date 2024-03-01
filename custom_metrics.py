import torch
from torchmetrics import F1Score, Metric
from torchmetrics.utilities import dim_zero_cat


class PerClassF1score(Metric):
    
    def __init__(self, num_classes, class_names, **kwargs):
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.labels=range(num_classes)
        self.f1_score=F1Score(task="multiclass", num_classes=num_classes,average=None)
        self.labels_names=class_names

    def update(self, logits:torch.Tensor, target: torch.Tensor) -> None:
        preds=logits.softmax(dim=1).argmax(dim=1)
        self.preds.append(preds)
        self.target.append(target)

    def compute(self):
        preds = dim_zero_cat(self.preds).int()
        target = dim_zero_cat(self.target).int()
        f1_scores=self.f1_score(preds,target)
        f1_scores = {self.labels_names[label]:score for label,score in zip(self.labels,f1_scores)}
        return f1_scores
