import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from models.generators.resblocks import Block


class Omniglot_Generator(nn.Module):
    """Generator generates 64x64."""
    def __init__(self, num_features=32, dim_z=128, bottom_width=7,
                 activation=F.relu, num_classes=0, distribution='normal'):
        super(Omniglot_Generator, self).__init__()
        self.num_features = num_features
        self.dim_z = dim_z
        self.bottom_width = bottom_width
        self.activation = activation
        self.num_classes = num_classes
        self.distribution = distribution

        self.l1 = nn.Linear(dim_z, 4 * num_features * bottom_width ** 2)

        self.block2 = Block(num_features * 4, num_features * 2,
                            activation=activation, upsample=True,
                            num_classes=num_classes)

        self.block3 = Self_Attn(num_features * 2)

        self.block4 = Block(num_features * 2, num_features,
                            activation=activation, upsample=True,
                            num_classes=num_classes)
        self.b4 = nn.BatchNorm2d(num_features)
        self.conv4 = nn.Conv2d(num_features, 1, 1, 1)

    def _initialize(self):
        init.xavier_uniform_(self.l1.weight.tensor)
        init.xavier_uniform_(self.conv4.weight.tensor)

    def forward(self, z, y=None, **kwargs):
        h = self.l1(z).view(z.size(0), -1, self.bottom_width, self.bottom_width)
        for i in range(2, 5):
            if i == 3:
                h, _ = getattr(self, 'block{}'.format(i))(h)
            else:
                h = getattr(self, 'block{}'.format(i))(h, y, **kwargs)
        h = self.activation(self.b4(h))
        return torch.tanh(self.conv4(h))


class VGG_Generator(nn.Module):
    """Generator generates 64x64."""
    def __init__(self, num_features=64, dim_z=128, bottom_width=4,
                 activation=F.relu, num_classes=0, distribution='normal'):
        super(VGG_Generator, self).__init__()
        self.num_features = num_features
        self.dim_z = dim_z
        self.bottom_width = bottom_width
        self.activation = activation
        self.num_classes = num_classes
        self.distribution = distribution

        self.l1 = nn.Linear(dim_z, 4 * num_features * bottom_width ** 2)

        self.block2 = Block(num_features * 4, num_features * 2,
                            activation=activation, upsample=True,
                            num_classes=num_classes)
        self.block3 = Block(num_features * 2, num_features,
                            activation=activation, upsample=True,
                            num_classes=num_classes)
        self.block4 = Block(num_features, num_features // 2,
                            activation=activation, upsample=True,
                            num_classes=num_classes)

        self.block5 = Self_Attn(num_features // 2)

        self.block6 = Block(num_features // 2, num_features // 4,
                            activation=activation, upsample=True,
                            num_classes=num_classes)

        self.block7 = Self_Attn(num_features // 4)

        self.b4 = nn.BatchNorm2d(num_features // 4)
        self.conv4 = nn.Conv2d(num_features // 4, 3, 1, 1)

    def _initialize(self):
        init.xavier_uniform_(self.l1.weight.tensor)
        init.xavier_uniform_(self.conv4.weight.tensor)

    def forward(self, z, y=None, **kwargs):
        h = self.l1(z).view(z.size(0), -1, self.bottom_width, self.bottom_width)
        for i in range(2, 8):
            if i == 5 or i == 7:
                h, _ = getattr(self, 'block{}'.format(i))(h)
            else:
                h = getattr(self, 'block{}'.format(i))(h, y, **kwargs)
        h = self.activation(self.b4(h))
        return torch.tanh(self.conv4(h))


class VGG_Generator2(nn.Module):
    """Generator generates 64x64."""
    def __init__(self, num_features=64, dim_z=128, bottom_width=4,
                 activation=F.relu, num_classes=0, distribution='normal'):
        super(VGG_Generator2, self).__init__()
        self.num_features = num_features
        self.dim_z = dim_z
        self.bottom_width = bottom_width
        self.activation = activation
        self.num_classes = num_classes
        self.distribution = distribution

        self.l1 = nn.Linear(dim_z, 4 * num_features * bottom_width ** 2)

        self.block2 = Block(num_features * 4, num_features * 2,
                            activation=activation, upsample=True,
                            num_classes=num_classes)
        self.block3 = Block(num_features * 2, num_features,
                            activation=activation, upsample=True,
                            num_classes=num_classes)
        self.block4 = Block(num_features, num_features // 2,
                            activation=activation, upsample=True,
                            num_classes=num_classes)

        self.block5 = Self_Attn(num_features // 2)

        self.block6 = Block(num_features // 2, num_features // 4,
                            activation=activation, upsample=True,
                            num_classes=num_classes)

        self.b4 = nn.BatchNorm2d(num_features // 4)
        self.conv4 = nn.Conv2d(num_features // 4, 3, 1, 1)

    def _initialize(self):
        init.xavier_uniform_(self.l1.weight.tensor)
        init.xavier_uniform_(self.conv4.weight.tensor)

    def forward(self, z, y=None, **kwargs):
        h = self.l1(z).view(z.size(0), -1, self.bottom_width, self.bottom_width)
        for i in range(2, 7):
            if i == 5:
                h, _ = getattr(self, 'block{}'.format(i))(h)
            else:
                h = getattr(self, 'block{}'.format(i))(h, y, **kwargs)
        h = self.activation(self.b4(h))
        return torch.tanh(self.conv4(h))

    
class Animal_Generator(nn.Module):
    """Generator generates 64x64."""
    def __init__(self, num_features=64, dim_z=128, bottom_width=4,
                 activation=F.relu, num_classes=0, distribution='normal'):
        super(Animal_Generator, self).__init__()
        self.num_features = num_features
        self.dim_z = dim_z
        self.bottom_width = bottom_width
        self.activation = activation
        self.num_classes = num_classes
        self.distribution = distribution

        self.l1 = nn.Linear(dim_z, 4 * num_features * bottom_width ** 2)    # 4

        self.block2 = Block(num_features * 4, num_features * 2,
                            activation=activation, upsample=True,
                            num_classes=num_classes)                        # 8
        self.block3 = Block(num_features * 2, num_features,
                            activation=activation, upsample=True,
                            num_classes=num_classes)                        # 16
        self.block4 = Block(num_features, num_features // 2,
                            activation=activation, upsample=True,
                            num_classes=num_classes)                        # 32

        self.block5 = Self_Attn(num_features // 2)                          # 32

        self.block6 = Block(num_features // 2, num_features // 4,
                            activation=activation, upsample=True,
                            num_classes=num_classes)                        # 64
        self.block7 = Block(num_features // 4, num_features // 8,
                            activation=activation, upsample=True,
                            num_classes=num_classes)                        # 128

        self.b4 = nn.BatchNorm2d(num_features // 8)
        self.conv4 = nn.Conv2d(num_features // 8, 3, 1, 1)

    def _initialize(self):
        init.xavier_uniform_(self.l1.weight.tensor)
        init.xavier_uniform_(self.conv4.weight.tensor)

    def forward(self, z, y=None, **kwargs):
        h = self.l1(z).view(z.size(0), -1, self.bottom_width, self.bottom_width)
        for i in range(2, 8):
            if i == 5:
                h, _ = getattr(self, 'block{}'.format(i))(h)
            else:
                h = getattr(self, 'block{}'.format(i))(h, y, **kwargs)
        h = self.activation(self.b4(h))
        return torch.tanh(self.conv4(h))


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

