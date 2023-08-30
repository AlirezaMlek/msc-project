from utils.BlockNode import InputType
from torch import nn
import numpy as np
import torch.nn.functional as F

class Link(nn.Module):
    def __init__(self, src_node, dst_node):
        super(Link, self).__init__()

        if src_node.inputType == InputType.D2 and dst_node.inputType == InputType.D1:
            in_dept = src_node.outputShape[1]
            in_size = src_node.outputShape[-1]
            out_size = int(np.sqrt(dst_node.inputShape))
            out_dept = 5
            self.flatten = nn.Flatten(start_dim=2)
        elif src_node.inputType == InputType.D1 and dst_node.inputType == InputType.D2:
            in_dept = 5
            in_size = int(np.sqrt(src_node.outputShape))
            out_size = dst_node.inputShape[-1]
            out_dept = dst_node.inputShape[1]
            self.unflatten = nn.Unflatten(2, (in_size, in_size))

        elif src_node.inputType == InputType.D2 and dst_node.inputType == InputType.D2:
            in_dept = src_node.outputShape[1]
            in_size = src_node.outputShape[-1]
            out_size = dst_node.outputShape[1]
            out_dept = dst_node.outputShape[-1]

        else:
            in_size = src_node.outputShape
            out_size = dst_node.inputShape

        if src_node.inputType == InputType.D2 or dst_node.inputType == InputType.D2:
            if in_size >= out_size:
                kernel_size, padding, stride = param(out_size, in_size)
                self.layer = nn.Conv2d(in_dept, out_dept, kernel_size=kernel_size, stride=stride, padding=padding)
            else:
                kernel_size, padding, stride = param(in_size, out_size)
                self.layer = nn.ConvTranspose2d(in_dept, out_dept, kernel_size=kernel_size, stride=stride,
                                                padding=padding)

        else:
            self.layer = nn.Linear(in_size, out_size)
            nn.init.eye_(self.layer.weight)
            nn.init.zeros_(self.layer.bias)

    def forward(self, x):
        if hasattr(self, 'unflatten'):
            x = self.unflatten(x)

        x = F.relu(self.layer(x))

        if hasattr(self, 'flatten'):
            x = self.flatten(x)

        return x



def param(m, n):
    stride = round(n / m)
    rem = n - (m - 1) * stride
    if rem > 4:
        stride += 1
        rem = n - (m - 1) * stride

    if rem % 2 == 0:
        kernel_size = 4
    else:
        kernel_size = 3

    padding = int((kernel_size - rem) / 2)

    return kernel_size, padding, stride