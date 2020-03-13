import torch
from sklearn.metrics import recall_score


def lb_score(yp: torch.Tensor, y: torch.Tensor):
    return recall_score(y.cpu().numpy(),
                        yp.argmax(dim=1).cpu().numpy(),
                        labels=y.unique().cpu().numpy(),
                        average="macro")
