#!/usr/bin/env python

import torch
import torch.nn as nn

in_channels = 3
out_channels = 64
kernel_size = (3, 3, 3)

# Create a 3D convolutional layer
conv3d = nn.Conv3d(in_channels, out_channels, kernel_size)

# Generate a random input volume
batch_size = 1
depth = 10
height = 32
width = 32
input_volume = torch.randn(batch_size, in_channels, depth, height, width)

# Perform the 3D convolution
output_volume = conv3d(input_volume)

print("Input volume shape:", input_volume.shape)
print("Output volume shape:", output_volume.shape)
