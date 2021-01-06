from collections import OrderedDict
from torch import nn


class InterLayerGetter(nn.ModuleDict):
    def __init__(self, backbone):
        super(InterLayerGetter, self).__init__()
        self.backbone = backbone

    def forward(self, x):
        out = self.backbone(x)
        out = OrderedDict((i + 1, v) for (i, v) in enumerate(out[-3:]))
        return out
