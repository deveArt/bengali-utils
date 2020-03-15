import functools
import operator
from torchvision.models import resnet50
from torch import nn
from copy import deepcopy
import torch
import numpy as np
from .metrics import lb_score
from efficientnet_pytorch import EfficientNet


def resnet50_arch(pretrained):
    root = resnet50(pretrained=pretrained)

    in_channels = 1  # Substitute 1st layer with 1 channel layer
    conv1 = nn.Conv2d(in_channels,
                      root.conv1.out_channels,
                      kernel_size=root.conv1.kernel_size,
                      stride=root.conv1.stride,
                      bias=deepcopy(root.conv1.bias))
    conv1.weight = nn.Parameter(root.conv1.weight.mean(1, keepdim=True), requires_grad=True)
    root.conv1 = conv1
    fc_out = root.fc.in_features
    root.fc = nn.Identity()

    return root, fc_out


def effnet_arch(pretrained, arch_name):
        root = EfficientNet.from_pretrained(arch_name) if pretrained else EfficientNet.from_name(arch_name)

        in_channels = 1  # Substitute 1st layer with 1 channel layer
        conv0 = nn.Conv2d(in_channels,
                          root._conv_stem.out_channels,
                          kernel_size=root._conv_stem.kernel_size,
                          stride=root._conv_stem.stride,
                          bias=deepcopy(root._conv_stem.bias))
        conv0.weight = nn.Parameter(root._conv_stem.weight.mean(1, keepdim=True), requires_grad=True)
        root._conv_stem = conv0

        fc_out = root._fc.in_features
        root._fc = nn.Identity()

        return root, fc_out


class BengaliBase(torch.nn.Module):
    def __init__(self, root_arch_fn, root_classes=168, vowel_classes=11, consonant_classes=7,
                 criterion=None, pretrained=False, **kwargs):
        super().__init__()

        root, fc_out = root_arch_fn(pretrained, **kwargs)
        self.root_model = root

        self.root_fc = nn.Linear(fc_out, root_classes)
        self.vowel_fc = nn.Linear(fc_out, vowel_classes)
        self.consonant_fc = nn.Linear(fc_out, consonant_classes)

        self.criterion = torch.nn.CrossEntropyLoss() if criterion is None else criterion

    def forward(self, x):
        out = self.root_model(x)
        root_pred = self.root_fc(out)
        vowel_pred = self.vowel_fc(out)
        consonant_pred = self.consonant_fc(out)

        return root_pred, vowel_pred, consonant_pred

    def run(self, x, y):
        outputs = self(x)
        losses, scores = [], []

        for i, pred in enumerate(outputs):
            losses.append(self.criterion(pred, y[:, i]))
            scores.append(lb_score(pred, y[:, i]))

        loss = sum(losses)
        score = np.average(scores, weights=[2, 1, 1])
        return loss, score

    def get_n_params(self):
        return sum((functools.reduce(operator.mul, p.size()) for p in self.parameters()))


class BengaliResNet50(BengaliBase):
    def __init__(self, root_classes=168, vowel_classes=11, consonant_classes=7,
                 criterion=None, pretrained=False):
        super().__init__(resnet50_arch, root_classes, vowel_classes, consonant_classes, criterion, pretrained)


class BengaliEfficient(BengaliBase):
    def __init__(self, arch_name, root_classes=168, vowel_classes=11, consonant_classes=7,
                 criterion=None, pretrained=False):

        super().__init__(effnet_arch, root_classes, vowel_classes, consonant_classes, criterion,
                         pretrained, arch_name=arch_name)
