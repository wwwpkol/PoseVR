import torch
import torch.nn as nn
from einops import rearrange
# from .block.vanilla_transformer_encoder import Transformer
from .block.strided_transformer_encoder import Transformer as Transformer_reduce


# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.nn as nn


class TemporalModelBase(nn.Module):
    """
    Do not instantiate this class.
    """

    def __init__(self, num_joints_in, num_joints_out,
                 filter_widths, causal, dropout, channels):
        super().__init__()

        # Validate input
        for fw in filter_widths:
            assert fw % 2 != 0, 'Only odd filter widths are supported'

        self.num_3d = 9
        self.num_joints_in = num_joints_in
        self.num_joints_out = num_joints_out
        self.filter_widths = filter_widths

        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        self.pad = [filter_widths[0] // 2]
        self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)
        self.shrink = nn.Conv1d(channels, num_joints_out * 3, 1)

    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.layers_bn:
            bn.momentum = momentum

    def receptive_field(self):
        """
        Return the total receptive field of this model as # of frames.
        """
        frames = 0
        for f in self.pad:
            frames += f
        return 1 + 2 * frames

    def total_causal_shift(self):
        """
        Return the asymmetric offset for sequence padding.
        The returned value is typically 0 if causal convolutions are disabled,
        otherwise it is half the receptive field.
        """
        frames = self.causal_shift[0]
        next_dilation = self.filter_widths[0]
        for i in range(1, len(self.filter_widths)):
            frames += self.causal_shift[i] * next_dilation
            next_dilation *= self.filter_widths[i]
        return frames

    def forward(self, x):
        sz = x.shape[:2]
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = self._forward_blocks(x)

        x = x.permute(0, 2, 1)


        return x




class TemporalModelOptimized1f(TemporalModelBase):
    """
    3D pose estimation model optimized for single-frame batching, i.e.
    where batches have input length = receptive field, and output length = 1.
    This scenario is only used for training when stride == 1.

    This implementation replaces dilated convolutions with strided convolutions
    to avoid generating unused intermediate results. The weights are interchangeable
    with the reference implementation.
    """

    def __init__(self, num_joints_in, num_joints_out,
                 filter_widths, causal=False, dropout=0.25, channels=1024):
        """
        Initialize this model.

        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        """
        super().__init__(num_joints_in, num_joints_out, filter_widths, causal, dropout, channels)

        self.expand_conv = nn.Conv1d(num_joints_in, channels, filter_widths[0], stride=filter_widths[0],
                                     bias=False)

        layers_conv = []
        layers_bn = []

        self.causal_shift = [(filter_widths[0] // 2) if causal else 0]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1) * next_dilation // 2)
            self.causal_shift.append((filter_widths[i] // 2) if causal else 0)

            layers_conv.append(nn.Conv1d(channels, channels, filter_widths[i], stride=filter_widths[i], bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            next_dilation *= filter_widths[i]

        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)

    def _forward_blocks(self, x):


        for i in range(len(self.pad) - 1):
            res = x[:, :, self.causal_shift[i + 1] + self.filter_widths[i + 1] // 2:: self.filter_widths[i + 1]]

            x = self.drop(self.relu(self.layers_bn[2 * i](self.layers_conv[2 * i](x))))
            x = res + self.drop(self.relu(self.layers_bn[2 * i + 1](self.layers_conv[2 * i + 1](x))))

        return x

class bata_v0(nn.Module):
    def __init__(self, args):
        super(bata_v0, self).__init__()
        self.fc_1 = nn.Sequential(
            nn.Conv1d(args.n_joints * 2 + 9, args.channels, args.stride_num[0], stride=args.stride_num[0], bias=False),
            nn.BatchNorm1d(args.channels, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.fc_2 = nn.Sequential(
            nn.Conv1d(2*args.n_joints + 9, args.channel, kernel_size=1),
            nn.BatchNorm1d(args.channel, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25)
        )



        self.Conv_layer = TemporalModelOptimized1f(args.n_joints * 2 + 9, args.n_joints * 3, filter_widths=args.stride_num,
                                                   causal=args.causal, dropout=args.dropout, channels=args.channels)
        self.Transformer_reduce = Transformer_reduce(len(args.stride_num), args.channel, args.d_hid, \
            length=args.frames, stride_num=args.stride_num)

        self.fcn_leaf = nn.Sequential(nn.BatchNorm1d(args.channel + args.channels, momentum=0.1),
            nn.Conv1d(args.channel + args.channels, 3 * 5, kernel_size=1)
        )

        self.fcn = nn.Sequential(nn.BatchNorm1d(args.channel + args.channels + 3 * 5, momentum=0.1),
            nn.Conv1d(args.channel + args.channels + 3 * 5, 3 * 17, kernel_size=1)
        )

    def forward(self, x, y):
        r'''
        :param x: [batchsize, frame, joint, feature] #[B, 27, 17, 2]
        :param y: [batchsize, frame, joint, feature] #[B, 27, 3, 3]
        :return:
        '''
        B,F,J,C = x.shape
        x_s = x.reshape(B, F, -1)
        y_s = y.reshape(B, F, -1)
        input = torch.concat((x_s, y_s), 2)
        x = input.permute(0, 2, 1).contiguous()
        x_ = self.fc_1(x)
        x_ = x_.permute(0, 2, 1).contiguous()
        x_conv = self.Conv_layer(x_)

        x__ = self.fc_2(x)
        x__ = x__.permute(0, 2, 1).contiguous()
        x_tran = self.Transformer_reduce(x__)
        x_tran = x_tran.permute(0, 2, 1).contiguous()

        a = x_conv.reshape(B, -1, 1)
        b = x_tran.reshape(B, -1, 1)
        x_t = torch.cat((a, b), 1)

        x_leaf = self.fcn_leaf(x_t)
        x_ = torch.concat((x_t, x_leaf), 1)
        x = self.fcn(x_)
        x = x.reshape(B, -1, 17, 3)
        x_leaf = x_leaf.reshape(B, -1, 5, 3)
        return x, x_leaf

