import torch.nn as nn

from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .vgg import vgg11, vgg19
from .vision_transformer import vit_b_16, vit_b_32, vit_l_16, vit_l_32

model_dict = {
    "vgg11": vgg11,
    "vgg19": vgg19,
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
    "vit_b_16": vit_b_16,
    "vit_b_32": vit_b_32,
    "vit_l_16": vit_l_16,
    "vit_l_32": vit_l_32,
}


CHANNELS = {
    "vgg11": 4096,
    "vgg19": 4096,
    "resnet18": 512,
    "resnet34": 512,
    "resnet101": 2048,
    "resnet152": 2048,
    "resnet50": 2048,
    "vit_b_16": 768,
}


class ImagenetModel(nn.Module):
    def __init__(self, name, num_classes=1):
        super(ImagenetModel, self).__init__()

        self.model = model_dict[name](pretrained=True)
        self.fc = nn.Linear(CHANNELS[name], num_classes)  # manually define a fc layer here

    def forward(self, x):
        feature = self.model(x)["penultimate"]
        return self.fc(feature)
