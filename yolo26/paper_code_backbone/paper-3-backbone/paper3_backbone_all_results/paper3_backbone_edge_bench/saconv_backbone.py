"""Paper 3 backbone for YOLO26n: stride-2 Conv at layers 3, 5, 7 -> SAConv (Switchable Atrous Convolution)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK

SACONV_LAYERS = (3, 5, 7)


class SAConv(nn.Module):
    """Switchable Atrous Convolution: shared 3x3 weight at dilation 1 and 3, mixed by a learned switch, with pre/post global context."""

    def __init__(self, c1, c2, k=3, s=2, rates=(1, 3)):
        super().__init__()
        self.c1, self.c2, self.k, self.s, self.rates = c1, c2, k, s, rates
        self.pre_context = nn.Conv2d(c1, c1, 1, bias=True)
        self.post_context = nn.Conv2d(c2, c2, 1, bias=True)
        for m in (self.pre_context, self.post_context):  # start as identity
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        self.switch = nn.Conv2d(c1, 1, 1, stride=s, bias=True)
        nn.init.zeros_(self.switch.weight)
        nn.init.zeros_(self.switch.bias)  # sigmoid(0) = 0.5: both branches equal at the start
        self.weight = nn.Parameter(torch.empty(c2, c1, k, k))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.weight_diff = nn.Parameter(torch.zeros(c2, c1, k, k))  # extra weight of the large-rate branch
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        self.static = False

    def to_static(self):
        """Replace the shared-weight branches by two plain Conv2d layers (same output), for NCNN export."""
        if self.static:
            return
        r1, r2 = self.rates
        p = self.k // 2
        self.conv_small = nn.Conv2d(self.c1, self.c2, self.k, self.s, p * r1, dilation=r1, bias=False)
        self.conv_large = nn.Conv2d(self.c1, self.c2, self.k, self.s, p * r2, dilation=r2, bias=False)
        with torch.no_grad():
            self.conv_small.weight.copy_(self.weight)
            self.conv_large.weight.copy_(self.weight + self.weight_diff)
        self.conv_small.to(self.weight.device, self.weight.dtype)
        self.conv_large.to(self.weight.device, self.weight.dtype)
        del self.weight, self.weight_diff
        self.static = True

    def forward(self, x):
        x = x + self.pre_context(F.adaptive_avg_pool2d(x, 1))
        s = torch.sigmoid(self.switch(F.avg_pool2d(x, 5, 1, 2)))
        if self.static:
            small, large = self.conv_small(x), self.conv_large(x)
        else:
            r1, r2 = self.rates
            p = self.k // 2
            small = F.conv2d(x, self.weight, None, self.s, p * r1, r1)
            large = F.conv2d(x, self.weight + self.weight_diff, None, self.s, p * r2, r2)
        y = s * small + (1 - s) * large
        y = y + self.post_context(F.adaptive_avg_pool2d(y, 1))
        return self.act(self.bn(y))


def add_saconv(model, layers=SACONV_LAYERS):
    """Replace the stride-2 Conv at the given layers with SAConv, starting from that Conv's (pretrained) weight and BN."""
    for i in layers:
        old = model.model[i]
        assert isinstance(old, Conv) and old.conv.kernel_size == (3, 3) and old.conv.stride == (2, 2), f"layer {i} is not a 3x3 s2 Conv"
        new = SAConv(old.conv.in_channels, old.conv.out_channels, 3, 2)
        with torch.no_grad():
            new.weight.copy_(old.conv.weight)
        new.bn.load_state_dict(old.bn.state_dict())
        new.i, new.f, new.type = old.i, old.f, "SAConv"  # keep the routing attributes used by the model's forward
        new.np = sum(p.numel() for p in new.parameters())
        model.model[i] = new
    return model


def to_static(model):
    """Convert every SAConv in the model to its deploy form. Returns the number of converted layers."""
    n = 0
    for m in model.modules():
        if isinstance(m, SAConv) and not m.static:
            m.to_static()
            n += 1
    return n


class SAConvTrainer(DetectionTrainer):
    """Load the pretrained weights into the base model first, then insert SAConv initialised from those weights."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=False)
        if weights:
            model.load(weights)
        add_saconv(model)
        if RANK in {-1, 0}:
            print(f"Paper 3 backbone: SAConv at layers {SACONV_LAYERS}")
        if verbose and RANK in {-1, 0}:
            model.info()
        return model
