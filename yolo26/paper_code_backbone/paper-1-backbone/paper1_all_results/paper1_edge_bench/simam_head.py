"""Paper 13 head for YOLO26n: parameter-free SimAM on the P3/P4/P5 features in front of Detect."""
import torch
import torch.nn as nn

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK


class SimAM(nn.Module):
    """SimAM: energy-based, parameter-free 3D attention (lambda = 1e-4 as in the original paper)."""

    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        n = x.shape[2] * x.shape[3] - 1
        d = (x - x.mean(dim=(2, 3), keepdim=True)).pow(2)
        e = d / (4 * (d.sum(dim=(2, 3), keepdim=True) / n + self.e_lambda)) + 0.5
        return x * torch.sigmoid(e)


class SimAMDetect(Detect):
    """YOLO26 Detect head with SimAM applied to every input feature map first."""

    def forward(self, x):
        return super().forward([self.simam(xi) for xi in x])


def add_simam(model, e_lambda=1e-4):
    """Turn the Detect layer of a built DetectionModel into SimAMDetect (no new parameters)."""
    m = model.model[-1]
    assert isinstance(m, Detect), "last layer must be Detect"
    m.simam = SimAM(e_lambda)
    m.__class__ = SimAMDetect
    return model


class SimAMTrainer(DetectionTrainer):
    """DetectionTrainer that inserts SimAM in front of Detect and loads the pretrained weights."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=False)
        add_simam(model)
        if weights:
            model.load(weights)  # SimAM has no weights, so every layer gets its pretrained weights
        if RANK in {-1, 0}:
            print(f"Paper 13 head: SimAM in front of Detect on layers {model.model[-1].f}")
        if verbose and RANK in {-1, 0}:
            model.info()
        return model
