import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# torch.manual_seed(0)
import math
from torch.autograd import Variable

np.set_printoptions(suppress=True, precision=2)


class Conv3D_Block(nn.Module):
    """ 3D Conv Block used for the dynamic branch """

    def __init__(self, in_channels: int = 18, out_channels: int = 32, kernel_size: int = (3, 3, 3),
                 stride: int = (1, 1, 1), padding: int = (1, 1, 1), bias: bool = True, norm: bool = True):

        super(Conv3D_Block, self).__init__()

        """
        Parameters
        ----------
        in_channels : int (default 18)
            number of input channels
        out_channels : int (default 32)
            number of output channels
        kernel_size : int (default 3)
            kernel size of output channels 
        stride : int (default 1)
            stride of convolution filters
        padding : int (default 1)
            padding for input image
        bias : bool (default True)
            option to use bias
        norm : bool (default True)
            option to do normalization after convolution     
        """

        if norm:
            self.Block = nn.Sequential(
                nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            )

        else:
            self.Block = nn.Sequential(
                nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                # nn.ReLU(inplace=True),
            )

    def forward(self, x: torch.Tensor):

        """ input tensor x [N, K, D, W, H] """

        return self.Block(x)


class Conv2D_Block(nn.Module):
    """ 2D Conv Block used for the dynamic branch """

    def __init__(self, in_channels: int = 18, out_channels: int = 32, kernel_size: int = (3, 3),
                 stride: int = (1, 1), padding: int = (1, 1), bias: bool = True, norm: bool = True):

        super(Conv2D_Block, self).__init__()

        """
        Parameters
        ----------
        in_channels : int (default 18)
            number of input channels
        out_channels : int (default 32)
            number of output channels
        kernel_size : int (default 3)
            kernel size of output channels 
        stride : int (default 1)
            stride of convolution filters
        padding : int (default 1)
            padding for input image
        bias : bool (default True)
            option to use bias
        norm : bool (default True)
            option to do normalization after convolution     
        """

        if norm:
            self.Block = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.Block = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                # nn.ReLU(inplace=True),
            )

    def forward(self, x: torch.Tensor):

        """ input tensor x [N, K, W, H] """

        return self.Block(x)


class LOAN(nn.Module):

    """ Location-aware Adaptive Normalization layer """

    def __init__(self, in_channels: int, cond_channels: int, free_norm: str = 'batch',
                 kernel_size: int = 3, norm: bool = True):
        super(LOAN, self).__init__()

        """
        Parameters
        ----------
        in_channels : int
            number of input channels
        cond_channels : int
            number of channels for conditional map
        free_norm : int (default batch)
            type of normalization to be used for the modulated map
        kernel_size : int (default 3)
            kernel size of output channels 
        norm : bool (default True)
            option to do normalization of the modulated map
        """

        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.kernel_size = kernel_size
        self.norm = norm
        self.k_channels = cond_channels

        if norm:
            if free_norm == 'batch':
                self.free_norm = nn.BatchNorm3d(self.in_channels, affine=False)
            else:
                raise ValueError('%s is not a recognized free_norm type in SPADE' % free_norm)

        # self.mlp = nn.Sequential(
        #    nn.Conv2d(in_channels=self.cond_channels, out_channels=self.k_channels, kernel_size=self.kernel_size,
        #             padding=self.kernel_size//2, padding_mode='replicate'),
        #    nn.ReLU(inplace=True)
        #    )

        # projection layers
        self.mlp_gamma = nn.Conv2d(self.k_channels, self.in_channels, kernel_size=self.kernel_size,
                                   padding=self.kernel_size // 2)
        self.mlp_beta = nn.Conv2d(self.k_channels, self.in_channels, kernel_size=self.kernel_size,
                                  padding=self.kernel_size // 2)

        # initialize projection layers
        # self.mlp.apply(self.init_weights)
        self.mlp_beta.apply(self.init_weights)
        self.mlp_gamma.apply(self.init_weights)

        # normalization for the conditional map
        self.free_norm_cond = torch.nn.BatchNorm2d(cond_channels, affine=False)

    def init_weights(self, m):
        # classname = m.__class__.__name__
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)

    def generate_one_hot(self, labels: torch.Tensor):

        """
        Convert the semantic map into one-hot encoded
        This method can be used for the CORINE land cover data_m
        """

        con_map = torch.nn.functional.one_hot(labels, num_classes=10)
        con_map = torch.permute(con_map, (0, 3, 2, 1))
        return con_map.float()

    def forward(self, x: torch.Tensor, con_map: torch.Tensor):

        """
        input tensor x [N, K, D, W, H]
        conditional map tensor con_map [N, K, W, H]
        """

        # parameter-free normalized map
        if self.norm:
            normalized = self.free_norm(x)
        else:
            normalized = x

        # used for data_m
        # con_map = self.generate_one_hot(con_map)
        # con_map = con_map.float()

        # produce scaling and bias conditioned on semantic map
        # con_map = F.interpolate(con_map, size=x.size()[-2:], mode='nearest')

        # normalize the conditional map
        actv = self.free_norm_cond(con_map)
        actv = nn.functional.relu(actv)

        # actv = self.mlp(con_map)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias after duplication along the D time dimension
        out = normalized * (1 + gamma[:, :, None, :, :]) + beta[:, :, None, :, :]

        return out

class CNN(nn.Module):

    """
    A PyTorch implementation of:
    Location-aware Adaptive Normalization: A Deep Learning Approach For Wildfire Danger Forecasting
    https://arxiv.org/abs/2212.08208 - https://doi.org/10.1109/TGRS.2023.3285401

    CNN, the model is hard codded
    """

    def __init__(self, input_channels_d: int = 10, input_channels_s: int = 5, input_channels_c: int = 10,
                 n_classes: int = 2, drop_out: float = 0.5, pe: bool = True, device: str = 'cuda'):

        super(CNN, self).__init__()

        """
        Parameters
        ----------
        input_channels_d : int (default 10)
            number of input dynamic features Vd
        input_channels_s : int (default 5)
            number of input static features
        input_channels_c : int (default 10)
            number of input land cover features
        n_classes : int (default 2)
            number of classes
        drop_out : float (default 0.5)
            dropout ratio
        pe : bool (default True)
            option to use positional encoding
        device : str (default cuda)
            device GPU or CPU
        """

        # TODO convert the model into a generic model and remove magic numbers

        self.input_channels_d = input_channels_d
        self.input_channels_s = input_channels_s
        self.input_channels_c = input_channels_c

        self.drop_out = drop_out

        self.n_classes = n_classes
        self.pe = pe
        self.device = device

        # hard coded layers

        cond_channels = 32

        self.Block1 = Conv3D_Block(10, 32, (3, 3, 3), (1, 1, 1), (0, 0, 0), True, False)
        self.loan1 = LOAN(in_channels=32, cond_channels=cond_channels, free_norm='batch', kernel_size=3)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        cond_channels = 64

        self.Block2 = Conv3D_Block(32, 64, (3, 3, 3), (1, 1, 1), (0, 0, 0), True, False)
        self.loan2 = LOAN(in_channels=64, cond_channels=cond_channels, free_norm='batch', kernel_size=3, norm=False)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.Block3 = Conv3D_Block(64, 128, (2, 2, 2), (1, 2, 2), (0, 0, 0), True, False)
        self.GAP = nn.AdaptiveAvgPool3d((2, 1, 1))

        self.Block4 = Conv2D_Block(15, 32, (3, 3), (1, 1), (0, 0), True, False)
        self.bn4 = nn.BatchNorm2d(32)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.Block5 = Conv2D_Block(32, 64, (3, 3), (1, 1), (0, 0), True, False)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.Block6 = Conv2D_Block(64, 128, (2, 2), (2, 2), (0, 0), True, False)
        self.GAP6 = nn.AdaptiveAvgPool2d(1)

        self.lastlayer1 = nn.Conv1d(128 * 2 + 128, 256, 1)
        self.lastlayer2 = nn.Conv1d(256, 128, 1)
        self.lastlayer3 = nn.Conv1d(128, 32, 1)
        self.lastlayer4 = nn.Conv1d(32, n_classes, 1)

        self.drop = nn.Dropout(drop_out)

        self.activation = nn.LogSoftmax(dim=1)

        if pe:
            self.PositionalEncoding = PositionalEncoder(256, 366, device).to(device)
            # PE_Weights should be used carefully i.e., with nonlinear activation/dropout otherwise they have no effect
            self.PE_Weights = torch.nn.Parameter(torch.zeros(256))

    def forward(self, x_s: torch.Tensor, x_c: torch.Tensor, x_d: torch.Tensor, x_t: torch.Tensor):

        """
        input static tensor x_s [N, C, W, H]
        input land cover tensor x_c [N, C, W, H]
        input dynamic tensor x_s [N, Vd, D, W, H]
        input day of the year x_t [N]
        """

        # concatenate static and land cover features to have Vs
        x_s = torch.cat((x_s, x_c), dim=1)

        x_d = self.Block1(x_d)
        x_s = self.Block4(x_s)

        x_d = self.loan1(x_d, x_s)

        x_s = self.bn4(x_s)
        x_s = F.relu(x_s, inplace=True)
        x_d = F.relu(x_d, inplace=True)
        x_d = self.pool1(x_d)
        x_s = self.pool4(x_s)

        x_d = self.Block2(x_d)
        x_s = self.Block5(x_s)

        x_d = self.loan2(x_d, x_s)

        x_d = F.relu(x_d, inplace=True)
        x_s = F.relu(x_s, inplace=False)
        x_d = self.pool2(x_d)
        x_s = self.pool5(x_s)

        x_d = self.Block3(x_d)
        x_s = self.Block6(x_s)
        x_d = F.relu(x_d, inplace=True)
        x_s = F.relu(x_s, inplace=True)

        x_d = self.GAP(x_d)
        x_d = x_d.view(-1, 128 * 2, 1)
        x_s = self.GAP6(x_s)
        x_s = x_s.view(-1, 128, 1)

        if self.pe:
            x_t = self.PositionalEncoding(x_t)
            x_t = x_t * (1 + self.PE_Weights)
            x_d = x_d + x_t.unsqueeze(-1)

        #x_d = self.drop(x_d)

        x = torch.cat((x_d, x_s), dim=1)

        x = self.lastlayer1(x)
        x = F.relu(x)
        x = self.drop(x)

        x = self.lastlayer2(x)
        x = F.relu(x)
        x = self.drop(x)

        x = self.lastlayer3(x)
        x = F.relu(x)

        x = self.lastlayer4(x)

        x = torch.squeeze(x, -1)
        x = self.activation(x)

        return x


class PositionalEncoder(nn.Module):

    """ Positional Encoding """

    def __init__(self, d_model: int = 256, n_days: int = 366, device: str = 'cuda'):
        super(PositionalEncoder, self).__init__()

        """
        Parameters
        ----------
        d_model : int (default 256)
            number of dimensions for encoding
        n_days : int (default 366)
            number of days
        device : str (default cuda)
            device GPU or CPU
        """

        self.d_model = d_model
        self.n_days = n_days
        self.device = device

        pe = torch.zeros(n_days, d_model).to(device)

        # precompute the encoding
        for pos in range(n_days):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10 ** ((2 * (i+1)) / d_model)))

        # store in buffer for fast access
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        input day of the year x_t [N]
        """
        x = Variable(self.pe[x - 1, :], requires_grad=False).to(self.device)
        # x = Variable(self.pe, requires_grad=False).cuda()

        return x


if __name__ == '__main__':

    device = 'cuda'

    test_d = torch.randn((2, 10, 10, 25, 25)).to(device)
    test_s = torch.randn((2, 5, 25, 25)).to(device)
    test_c = torch.randn((2, 10, 25, 25)).to(device)
    test_t = torch.randint(366, (2,)).long().to(device)

    model = CNN(device=device).to(device)

    test = model(test_s, test_c, test_d, test_t)

    print(test.shape)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(count_parameters(model))
