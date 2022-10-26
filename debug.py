from svconv import SVConv2d
import torch
import torch.nn.functional as F

input = torch.rand(1, 2, 3, 3)
print(input)

convlayer = SVConv2d(in_channels=2, out_channels=2, kernel_size=3, spatial_scalar_hint=input.size(), stride=1, padding=(1,1), padding_mode='circular', bias=False)
print(convlayer(input))