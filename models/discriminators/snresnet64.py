import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn import utils

from models.discriminators.resblocks import Block
from models.discriminators.resblocks import OptimizedBlock


class Omniglot_Discriminator(nn.Module):

    def __init__(self, num_features=32, num_classes=0, activation=F.relu):
        super(Omniglot_Discriminator, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation

        self.block1 = OptimizedBlock(1, num_features)
        self.block2 = Block(num_features, num_features * 2,
                            activation=activation, downsample=True)
        self.block3 = Self_Attn(num_features * 2)
        self.block4 = Block(num_features * 2, num_features * 4,
                            activation=activation, downsample=True)
        self.l4 = utils.spectral_norm(nn.Linear(num_features * 4, 1))
        if num_classes > 0:
            self.l_y = utils.spectral_norm(
                nn.Embedding(num_classes, num_features * 4))

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.l4.weight.data)
        optional_l_y = getattr(self, 'l_y', None)
        if optional_l_y is not None:
            init.xavier_uniform_(optional_l_y.weight.data)

    def forward(self, x, y=None):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h, _ = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.l4(h)
        if y is not None:
            output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
        return output


class VGG_Discriminator(nn.Module):

    def __init__(self, num_features=64, num_classes=0, activation=F.relu):
        super(VGG_Discriminator, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation

        self.block1 = OptimizedBlock(3, num_features // 4)
        self.block2 = Block(num_features // 4, num_features // 2,
                            activation=activation, downsample=True)
        self.block3 = Block(num_features // 2, num_features,
                            activation=activation, downsample=True)
        self.block4 = Block(num_features, num_features * 2,
                            activation=activation, downsample=True)

        self.block5 = Self_Attn(num_features * 2)

        self.block6 = Block(num_features * 2, num_features * 4,
                            activation=activation, downsample=True)

        self.block7 = Self_Attn(num_features * 4)


        self.l4 = utils.spectral_norm(nn.Linear(num_features * 4, 1))
        if num_classes > 0:
            self.l_y = utils.spectral_norm(
                nn.Embedding(num_classes, num_features * 4))

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.l4.weight.data)
        optional_l_y = getattr(self, 'l_y', None)
        if optional_l_y is not None:
            init.xavier_uniform_(optional_l_y.weight.data)

    def forward(self, x, y=None):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h, _ = self.block5(h)
        h = self.block6(h)
        h, _ = self.block7(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.l4(h)
        if y is not None:
            output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
        return output


class Animal_Discriminator(nn.Module):

    def __init__(self, num_features=64, num_classes=0, activation=F.relu):
        super(Animal_Discriminator, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation

        self.block1 = OptimizedBlock(3, num_features // 8)                        # 128
        self.block2 = Block(num_features // 8, num_features // 4,
                            activation=activation, downsample=True)               # 64
        self.block3 = Block(num_features // 4, num_features // 2,
                            activation=activation, downsample=True)               # 32
        self.block4 = Block(num_features // 2, num_features,
                            activation=activation, downsample=True)               # 16

        self.block5 = Self_Attn(num_features)                             # 8

        self.block6 = Block(num_features, num_features * 2,                        # 8
                            activation=activation, downsample=True)

        self.block7 = Block(num_features * 2, num_features * 4,                   # 4
                            activation=activation, downsample=True)



        self.l4 = utils.spectral_norm(nn.Linear(num_features * 4, 1))
        if num_classes > 0:
            self.l_y = utils.spectral_norm(
                nn.Embedding(num_classes, num_features * 4))

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.l4.weight.data)
        optional_l_y = getattr(self, 'l_y', None)
        if optional_l_y is not None:
            init.xavier_uniform_(optional_l_y.weight.data)

    def forward(self, x, y=None):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h, _ = self.block5(h)
        h = self.block6(h)
        h= self.block7(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.l4(h)
        if y is not None:
            output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
        return output
    

class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height) \
            .permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention
