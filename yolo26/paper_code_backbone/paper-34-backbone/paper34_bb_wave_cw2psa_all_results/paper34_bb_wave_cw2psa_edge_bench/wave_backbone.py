"""Paper 34 backbone for YOLO26n: Wave (fixed Haar DWT, bilinear up, 1x1 + BN, residual) after the C3k2 at layers 2, 4, 6, 8
and/or Cw2PSA (C2PSA with the identity half replaced by a Wave branch) at layer 10."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules.block import C2PSA, C3k2
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK

WAVE_LAYERS = (2, 4, 6, 8)
CW2PSA_LAYER = 10


def haar_kernels():
    """The 4 Haar 2x2 kernels LL, LH, HL, HH (outer products of hL = [1, 1]/sqrt2 and hH = [1, -1]/sqrt2)."""
    hl = torch.tensor([1.0, 1.0]) / math.sqrt(2)
    hh = torch.tensor([1.0, -1.0]) / math.sqrt(2)
    return torch.stack([torch.outer(hl, hl), torch.outer(hl, hh), torch.outer(hh, hl), torch.outer(hh, hh)])  # (4, 2, 2)


class Wave(nn.Module):
    """Haar DWT (fixed grouped conv, stride 2) -> bilinear upsample x2 -> concat (4C) -> 1x1 conv + BN -> residual add."""

    def __init__(self, c):
        super().__init__()
        self.c = c
        self.register_buffer("haar", haar_kernels().unsqueeze(1).repeat(c, 1, 1, 1))  # (4C, 1, 2, 2), fixed
        self.fuse = nn.Conv2d(4 * c, c, 1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        nn.init.zeros_(self.bn.weight)  # the branch adds nothing at the start (exact warm start)

    def forward(self, x):
        s = F.conv2d(x, self.haar, stride=2, groups=self.c)  # (B, 4C, H/2, W/2): LL, LH, HL, HH per channel
        # back to the input size (not scale_factor=2): an odd H or W (rect validation, e.g. 21) would be off by one
        s = F.interpolate(s, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return x + self.bn(self.fuse(s))


class C3k2Wave(nn.Module):
    """A pretrained C3k2 block followed by a Wave bypass."""

    def __init__(self, block):
        super().__init__()
        self.block = block
        self.wave = Wave(block.cv2.conv.out_channels)

    def forward(self, x):
        return self.wave(self.block(x))


class Cw2PSA(nn.Module):
    """C2PSA layout: cv1 -> split -> (Wave branch, PSA branch) -> concat -> cv2. cv1, PSA block and cv2 keep their weights."""

    def __init__(self, c2psa):
        super().__init__()
        self.c = c2psa.c
        self.cv1, self.m, self.cv2 = c2psa.cv1, c2psa.m, c2psa.cv2
        self.wave = Wave(self.c)

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        return self.cv2(torch.cat((self.wave(a), self.m(b)), 1))


def _swap(model, i, new, name):
    old = model.model[i]
    new = new.to(next(old.parameters()).device)
    new.i, new.f, new.type = old.i, old.f, name  # keep the routing attributes used by the model's forward
    new.np = sum(p.numel() for p in new.parameters())
    model.model[i] = new


def add_wave(model, layers=WAVE_LAYERS):
    """Wrap the C3k2 at the given backbone layers with a Wave bypass. Returns the model."""
    for i in layers:
        old = model.model[i]
        assert isinstance(old, C3k2), f"layer {i} is {type(old).__name__}, not C3k2"
        _swap(model, i, C3k2Wave(old), "C3k2Wave")
    return model


def add_cw2psa(model, layer=CW2PSA_LAYER):
    """Turn the C2PSA at the given layer into Cw2PSA (all C2PSA weights kept). Returns the model."""
    old = model.model[layer]
    assert isinstance(old, C2PSA), f"layer {layer} is {type(old).__name__}, not C2PSA"
    _swap(model, layer, Cw2PSA(old), "Cw2PSA")
    return model


class WaveTrainerBase(DetectionTrainer):
    """Loads the pretrained weights first, then applies Wave and/or Cw2PSA."""

    WAVE = False
    CW2PSA = False

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=False)
        if weights:
            model.load(weights)
        parts = []
        if self.WAVE:
            add_wave(model)
            parts.append(f"Wave after layers {WAVE_LAYERS}")
        if self.CW2PSA:
            add_cw2psa(model)
            parts.append(f"Cw2PSA at layer {CW2PSA_LAYER}")
        if RANK in {-1, 0}:
            print("Paper 34 backbone: " + ", ".join(parts))
        if verbose and RANK in {-1, 0}:
            model.info()
        return model


class WaveTrainer(WaveTrainerBase):
    WAVE = True


class Cw2PSATrainer(WaveTrainerBase):
    CW2PSA = True


class WaveCw2PSATrainer(WaveTrainerBase):
    WAVE = True
    CW2PSA = True
