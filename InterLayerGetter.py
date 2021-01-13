from collections import OrderedDict
from torch import nn

"""
Clase InterLayerGetter, construida de manera similar a torchvision.models._utils.IntermediateLayerGetter
Recibe una red backbone y en el forward selecciona las ultimas 3 capas intermedias, de esta forma
se maniene la estructura del forward en retinaface  
"""
class InterLayerGetter(nn.ModuleDict):
    def __init__(self, backbone):
        super(InterLayerGetter, self).__init__()
        self.backbone = backbone

    def forward(self, x):
        out = self.backbone(x)
        out = OrderedDict((i + 1, v) for (i, v) in enumerate(out[-3:]))
        return out
