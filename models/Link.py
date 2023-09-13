from utils.BlockNode import InputType
from torch import nn
import numpy as np
import torch

class Link(nn.Module):
    def __init__(self, src_node, dst_node):
        super(Link, self).__init__()

        self.layers = nn.ModuleList()

        add_flatten = False

        if src_node.inputType == InputType.D2 and dst_node.inputType == InputType.D1:
            in_dept = src_node.outputShape[1]
            in_size = src_node.outputShape[-1]
            out_size = int(np.sqrt(dst_node.inputShape))
            out_dept = 5
            add_flatten = True

        elif src_node.inputType == InputType.D1 and dst_node.inputType == InputType.D2:
            in_dept = 5
            in_size = int(np.sqrt(src_node.outputShape))
            out_size = dst_node.inputShape[-1]
            out_dept = 5
            unflatten = nn.Unflatten(2, (in_size, in_size))
            self.layers.append(unflatten)

        elif src_node.inputType == InputType.D2 and dst_node.inputType == InputType.D2:
            in_dept = src_node.outputShape[1]
            in_size = src_node.outputShape[-1]
            out_dept = dst_node.inputShape[1]
            out_size = dst_node.inputShape[-1]

        else:
            in_size = src_node.outputShape
            out_size = dst_node.inputShape

        if src_node.inputType == InputType.D2 or dst_node.inputType == InputType.D2:
            if in_size >= out_size:
                kernel_size, padding, stride = param(out_size, in_size)
                layer = nn.Conv2d(in_dept, out_dept, kernel_size=kernel_size, stride=stride, padding=padding)
            else:
                kernel_size, padding, stride = param(in_size, out_size)
                layer = nn.ConvTranspose2d(in_dept, out_dept, kernel_size=kernel_size, stride=stride, padding=padding)

            self.layers.append(layer)

            # batchNorm = nn.BatchNorm2d(num_features=out_dept)
            # self.layers.append(batchNorm)
            #
            # maxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # self.layers.append(maxPool)

            # if dst_node.inputType != InputType.D2:
            coeff = CoeffLayer(out_dept, 1)
            self.layers.append(coeff)

            if add_flatten:
                flatten = nn.Flatten(start_dim=2)
                self.layers.append(flatten)

        else:

            layer1 = nn.Linear(in_size, 100)
            nn.init.eye_(layer1.weight)
            nn.init.zeros_(layer1.bias)

            layer2 = nn.Linear(100, out_size)
            nn.init.eye_(layer2.weight)
            nn.init.zeros_(layer2.bias)

            norm = nn.LayerNorm(normalized_shape=out_size)
            drop = nn.Dropout(p=.1)

            self.layers.append(layer1)
            self.layers.append(layer2)
            self.layers.append(norm)
            # self.layers.append(drop)

    def forward(self, x):

        for layer in self.layers:
            if isinstance(x, dict):
                x['input_ids'] = layer(x['input_ids'][0] if isinstance(x['input_ids'], tuple) else x['input_ids'])
            else:
                x = layer(x)

        return x



def param(m, n):

    if n == m:
        return 1, 0, 1

    stride = int(np.sqrt(n / 2))
    kernel_size = stride * 2
    rem = n - (m - 1) * stride
    while rem > kernel_size:
        stride += 1
        rem = n - (m - 1) * stride

    while rem < - kernel_size:
        stride -= 1
        kernel_size -= 1
        rem = n - (m - 1) * stride


    padding = int((kernel_size - rem) / 2)

    kernel_size += (kernel_size-rem) % 2

    if (n-kernel_size+2*padding)/stride + 1 < m:
        add = (n-kernel_size+2*padding) % stride
        kernel_size -= stride - add
    elif (n-kernel_size+2*padding)/stride + 1 > m:
        kernel_size += ((n-kernel_size+2*padding) % stride)

    return kernel_size, padding, stride



class CoeffLayer(nn.Module):
    def __init__(self, in_channels, coeff=0.5):
        super(CoeffLayer, self).__init__()
        self.coeff = nn.Parameter(torch.ones(1,in_channels,1,1)*coeff)

    def forward(self, x):
        return self.coeff * x


class ConcatLayer(nn.Module):
    def __init__(self, node):
        super(ConcatLayer, self).__init__()

        self.layers = nn.ModuleList()

        if node.inputType == InputType.D2:
            self.dim = 1
            in_dept = node.inputShape[1] + 5
            out_dept = node.inputShape[1]
            layer = nn.Conv2d(in_dept, out_dept, kernel_size=1)
            self.layers.append(layer)
        else:
            self.dim = -1
            in_dept = node.inputShape * 2
            out_dept = node.inputShape
            layer = nn.Linear(in_dept, out_dept)
            self.layers.append(layer)


    def forward(self, data):

        x = torch.cat((data['main'], data['branch']), dim=self.dim)
        for layer in self.layers:
            x = layer(x)
        return x
