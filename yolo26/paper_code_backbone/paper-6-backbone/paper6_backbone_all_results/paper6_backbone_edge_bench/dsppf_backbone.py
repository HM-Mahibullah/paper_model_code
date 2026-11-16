"""Paper 6 backbone for YOLO26n: SPPF at layer 9 -> DS-SPPF (Dynamic Scale SPPF)."""
import torch
import torch.nn as nn

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules.block import SPPF
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK

DSSPPF_LAYER = 9
N_BRANCH = 4


class DSSPPF(nn.Module):
    """Dynamic Scale SPPF: identity, MaxPool and two dilated depthwise branches, fused with per-pixel, per-channel softmax weights."""

    def __init__(self, c1, c2, k=5, shortcut=False):
        super().__init__()
        c_ = c1 // 2  # hidden channels (same as SPPF)
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.dw2 = Conv(c_, c_, 3, 1, g=c_, d=2)
        self.dw3 = Conv(c_, c_, 3, 1, g=c_, d=3)
        self.gen = nn.Sequential(nn.Conv2d(c_, c_, 3, 1, 1), nn.SiLU(), nn.Conv2d(c_, N_BRANCH * c_, 1))
        nn.init.zeros_(self.gen[-1].weight)  # equal branch weights (0.25) at the start
        nn.init.zeros_(self.gen[-1].bias)
        self.cv2 = Conv(N_BRANCH * c_, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        f = self.cv1(x)
        branches = (f, self.m(f), self.dw2(f), self.dw3(f))
        # softmax across the 4 branches, written with 4-D ops only (NCNN friendly)
        logits = self.gen(f).chunk(N_BRANCH, 1)
        mx = torch.maximum(torch.maximum(logits[0], logits[1]), torch.maximum(logits[2], logits[3]))
        e = [torch.exp(l - mx) for l in logits]
        s = e[0] + e[1] + e[2] + e[3]
        y = self.cv2(torch.cat([b * w / s for b, w in zip(branches, e)], 1))
        return y + x if self.add else y


def add_dsppf(model, layer=DSSPPF_LAYER):
    """Replace the SPPF at the given layer with DS-SPPF, copying cv1/cv2 from the SPPF. Returns the model."""
    old = model.model[layer]
    assert isinstance(old, SPPF), f"layer {layer} is {type(old).__name__}, not SPPF"
    c1, c_, c2 = old.cv1.conv.in_channels, old.cv1.conv.out_channels, old.cv2.conv.out_channels
    assert old.cv2.conv.in_channels == N_BRANCH * c_, "SPPF concat width does not match the 4 DS-SPPF branches"
    new = DSSPPF(c1, c2, k=old.m.kernel_size)
    new.add = old.add  # same shortcut as the base SPPF
    new.cv1.load_state_dict(old.cv1.state_dict())
    new.cv2.load_state_dict(old.cv2.state_dict())
    with torch.no_grad():
        new.cv2.conv.weight.mul_(N_BRANCH)  # cancels the initial 0.25 branch weights
    new = new.to(next(old.parameters()).device)
    new.i, new.f, new.type = old.i, old.f, "DSSPPF"  # keep the routing attributes used by the model's forward
    new.np = sum(p.numel() for p in new.parameters())
    model.model[layer] = new
    return model


class DSSPPFTrainer(DetectionTrainer):
    """DetectionTrainer that loads the pretrained weights first, then swaps SPPF for DS-SPPF (cv1/cv2 keep the pretrained weights)."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=False)
        if weights:
            model.load(weights)
        add_dsppf(model)
        if RANK in {-1, 0}:
            print(f"Paper 6 backbone: DS-SPPF at layer {DSSPPF_LAYER}")
        if verbose and RANK in {-1, 0}:
            model.info()
        return model
