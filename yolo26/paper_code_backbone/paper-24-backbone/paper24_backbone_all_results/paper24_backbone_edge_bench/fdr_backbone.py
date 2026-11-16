"""Paper 24 backbone for YOLO26n: LAE-RPD at layers 3, 5, 7 (stride-2 Conv) and/or UCDC-Lite at layer 9 (SPPF)."""
import torch
import torch.nn as nn

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules.block import SPPF
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK

LAERPD_LAYERS = (3, 5, 7)
UCDC_LAYER = 9
SUBS = ((0, 0), (1, 0), (0, 1), (1, 1))  # the 4 positions of each 2x2 neighbourhood


class LAERPD(nn.Module):
    """LAE-based response-preserving down-sampling: softmax weights over the 4 positions of each 2x2 block x 4 grouped-conv candidates."""

    def __init__(self, c1, c2, G=16):
        super().__init__()
        g = max(1, c1 // G)  # number of conv groups (G channels per group)
        assert c1 % g == 0 and (4 * c2) % g == 0, f"group count {g} does not divide {c1} / {4 * c2}"
        self.pool = nn.AvgPool2d(3, 1, 1)
        self.att = nn.Conv2d(c1, c2, 1)
        self.cand = nn.Conv2d(c1, 4 * c2, 3, 2, 1, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def weights(self, x):
        """Softmax over the 4 candidate positions, written with 4-D ops only (NCNN friendly). Returns 4 maps that sum to 1."""
        a = self.att(self.pool(x))
        logits = [a[..., i::2, j::2] for i, j in SUBS]
        mx = torch.maximum(torch.maximum(logits[0], logits[1]), torch.maximum(logits[2], logits[3]))
        e = [torch.exp(l - mx) for l in logits]
        s = e[0] + e[1] + e[2] + e[3]
        return [w / s for w in e]

    def forward(self, x):
        w = self.weights(x)
        v = self.cand(x).chunk(4, 1)
        return self.act(self.bn(w[0] * v[0] + w[1] * v[1] + w[2] * v[2] + w[3] * v[3]))


class UCDCLite(nn.Module):
    """Lightweight U-shaped dilated context aggregation: 1x1 -> 3x3 d2 -> 3x3 d4 -> 3x3 d2 on concat -> 1x1 on concat with X1."""

    def __init__(self, c1, c2):
        super().__init__()
        cm = c2 // 2
        self.cv1 = Conv(c1, cm, 1, 1)
        self.d1 = Conv(cm, cm // 2, 3, 1, d=2)
        self.d2 = Conv(cm // 2, cm // 2, 3, 1, d=4)
        self.d3 = Conv(cm, cm, 3, 1, d=2)
        self.cv2 = Conv(2 * cm, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv1(x)
        d1 = self.d1(x1)
        d2 = self.d2(d1)
        d3 = self.d3(torch.cat((d1, d2), 1))
        return self.cv2(torch.cat((x1, d3), 1))


def _swap(model, i, new, name):
    old = model.model[i]
    new = new.to(next(old.parameters()).device)
    new.i, new.f, new.type = old.i, old.f, name  # keep the routing attributes used by the model's forward
    new.np = sum(p.numel() for p in new.parameters())
    model.model[i] = new


def add_laerpd(model, layers=LAERPD_LAYERS):
    """Replace the 3x3 stride-2 Conv at the given backbone layers with LAE-RPD. Returns the model."""
    for i in layers:
        old = model.model[i]
        assert isinstance(old, Conv) and old.conv.kernel_size == (3, 3) and old.conv.stride == (2, 2), f"layer {i} is not a 3x3 s2 Conv"
        _swap(model, i, LAERPD(old.conv.in_channels, old.conv.out_channels), "LAERPD")
    return model


def add_ucdc(model, layer=UCDC_LAYER):
    """Replace the SPPF at the given layer with UCDC-Lite; its first 1x1 conv takes the SPPF cv1 weights. Returns the model."""
    old = model.model[layer]
    assert isinstance(old, SPPF), f"layer {layer} is {type(old).__name__}, not SPPF"
    new = UCDCLite(old.cv1.conv.in_channels, old.cv2.conv.out_channels)
    if new.cv1.conv.weight.shape == old.cv1.conv.weight.shape:
        new.cv1.load_state_dict(old.cv1.state_dict())
    _swap(model, layer, new, "UCDCLite")
    return model


class FDRTrainer(DetectionTrainer):
    """Loads the pretrained weights first, then applies LAE-RPD and/or UCDC-Lite."""

    LAERPD = False
    UCDC = False

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=False)
        if weights:
            model.load(weights)
        parts = []
        if self.LAERPD:
            add_laerpd(model)
            parts.append(f"LAE-RPD at layers {LAERPD_LAYERS}")
        if self.UCDC:
            add_ucdc(model)
            parts.append(f"UCDC-Lite at layer {UCDC_LAYER}")
        if RANK in {-1, 0}:
            print("Paper 24 backbone: " + ", ".join(parts))
        if verbose and RANK in {-1, 0}:
            model.info()
        return model


class LAERPDTrainer(FDRTrainer):
    LAERPD = True


class UCDCTrainer(FDRTrainer):
    UCDC = True


class LAERPDUCDCTrainer(FDRTrainer):
    LAERPD = True
    UCDC = True
