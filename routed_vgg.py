import torch, pdb
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from . taskrouting import TaskRouter
import torch.nn.functional as F

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
}


class VGG(nn.Module):

    def __init__(self, features, task_count=10, init_weights=True, active_task=0, bottleneck_spatial=[7,7]):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.task_count = task_count
        self.active_task = active_task
        for ix in range(self.task_count):
            self.add_module("classifier_" + str(ix), nn.Sequential(
                nn.Linear(1024 * bottleneck_spatial[0] * bottleneck_spatial[1], 2)
            ))

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        output = self.get_layer("classifier_" + str(self.active_task)).forward(x)

        return output

    def set_active_task(self, active_task):
        self.active_task = active_task
        return active_task

    def get_layer(self, name):
        return getattr(self, name)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, task_count, sigma, batch_norm=False):
    layers = []
    in_channels = 3
    for ix, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            router = TaskRouter(v, task_count, int(v * sigma), "taskrouter_"+str(ix))
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), router, nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, router, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
}


def vgg11(pretrained=False, task_count=10, sigma=0.5, **kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], task_count, sigma), task_count, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model


def vgg11_bn(pretrained=False, task_count=10, sigma=0.5, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], task_count, sigma), task_count, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model
