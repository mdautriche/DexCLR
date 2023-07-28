from torchvision.models.segmentation import deeplabv3_resnet101, deeplabv3_resnet50
from torch import nn
import copy

class Shaper(nn.Module):
    def __init__(self, shape=(-1,2048)):
        self.shape = shape
        super(Shaper, self).__init__()
        
    def forward(self, x):
        return x.reshape(self.shape)
    
class DeepLab(nn.Module):
    def __init__(self, nb_class = 2, nb_channels = 4, n_features=2048, projection_dim=20, phase='pretrain', depth=50):
        super(DeepLab, self).__init__()
        self.phase = phase
        self.nb_class = nb_class
        self.nb_channels = nb_channels
        self.n_features = n_features
        if depth == 50:
            model_ = deeplabv3_resnet50(weights=None, num_classes=nb_class)
            print('Deplav3 with ResNet depth 50 is loaded')
        elif depth == 101:
            model_ = deeplabv3_resnet101(weights=None, num_classes=nb_class)
            print('Deplav3 with ResNet depth 101 is loaded')
        else:
            raise ValueError('Provided resnet depth is not found. It should be either 50 or 101')
            
        model_.backbone.conv1 =  nn.Conv2d(self.nb_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        if phase == 'pretrain':
            self.net = copy.deepcopy(model_.backbone)
            self.adaptive_pool = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=1),Shaper())
            self.projector = nn.Sequential(nn.Linear(self.n_features, self.n_features, bias=False),
                                                         nn.ReLU(),
                                                         nn.Linear(self.n_features, projection_dim, bias=False))
            del model_
        else:
            self.net = copy.deepcopy(model_)
            del model_

    def forward(self, x1, x2=None):
        if self.phase == 'pretrain':
            h_i = self.adaptive_pool(self.net(x1)['out'])
            h_j = self.adaptive_pool(self.net(x2)['out'])
            z_i = self.projector(h_i)
            z_j = self.projector(h_j)
            return h_i, h_j, z_i, z_j
        else:
            x = self.net(x1)['out']
            return x
