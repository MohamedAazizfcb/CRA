"""Parallel WaveGAN Modules."""

import logging
import math

import torch

from parallel_wavegan.layers import Conv1d
from parallel_wavegan.layers import Conv1d1x1
from parallel_wavegan.layers import ResidualBlock
from parallel_wavegan.layers import upsample
from parallel_wavegan import models


class ParallelWaveGANGenerator(torch.nn.Module):
    """Parallel WaveGAN Generator module."""

    # TODO YOUSSRY









class ParallelWaveGANDiscriminator(torch.nn.Module):
    """Parallel WaveGAN Discriminator module."""
    # TODO DALIA


    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 kernel_size=3,
                 layers=10,
                 conv_channels=64,
                 dilation_factor=1,
                 nonlinear_activation="LeakyReLU",
                 nonlinear_activation_params={"negative_slope": 0.2},
                 bias=True,
                 use_weight_norm=True,
                 ):

        """Initialize Parallel WaveGAN Discriminator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): width x height of the filter mask.
            layers (int): Number of conv layers.
            conv_channels (int): Number of chnn layers.
            dilation_factor (int): Dilation factor. For example, if dilation_factor = 2,
                the dilation will be 2, 4, 8, ..., and so on.
            nonlinear_activation (str): Nonlinear function after each conv.
            nonlinear_activation_params (dict): Nonlinear function parameters
            bias (bool): Whether to use bias parameter in conv.
            use_weight_norm (bool) Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
        """

        super(ParallelWaveGANDiscriminator, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."
        assert dilation_factor > 0, "Dilation factor must be > 0."
        self.conv_layers = torch.nn.ModuleList()
        conv_in_channels = in_channels
        for i in range(layers - 1):
            if i == 0:
                dilation = 1
            else:
                dilation = i if dilation_factor == 1 else dilation_factor ** i
                conv_in_channels = conv_channels
            padding = (kernel_size - 1) // 2 * dilation    #//divide and take floor
            conv_layer = [
                Conv1d(conv_in_channels, conv_channels,
                       kernel_size=kernel_size, padding=padding,
                       dilation=dilation, bias=bias),
                getattr(torch.nn, nonlinear_activation)(inplace=True, **nonlinear_activation_params)
            ]
            self.conv_layers += conv_layer

        #output convlayer
        padding = (kernel_size - 1) // 2
        last_conv_layer = Conv1d(
            conv_in_channels, out_channels,
            kernel_size=kernel_size, padding=padding, bias=bias)
        self.conv_layers += [last_conv_layer]

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()




    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)