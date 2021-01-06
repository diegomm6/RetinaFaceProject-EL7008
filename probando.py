import torch
import timm

m = timm.create_model('mobilenetv3_large_100', features_only=True, pretrained=True)
print(f'Feature channels: {m.feature_info.channels()}')

o = m(torch.randn(2, 3, 224, 224))
for x in o:
    print(x.shape)