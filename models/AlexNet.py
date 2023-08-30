from torchvision import models as cmodels
from utils.BlockNetwork import DnnApp
from utils.BlockNode import InputType
import torch.nn as nn
import torch
from torchvision import ops
from models import Link

def create_model():

    def predictor(scores):
        pass


    model = cmodels.alexnet(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    embLayer = nn.Sequential(ops.Permute([0,3,1,2]))


    networkLayers = [model.features[0:3], model.features[3:6], model.features[6:8],
                     model.features[8:10], model.features[10:13]]
    outputBlock = [nn.Flatten(start_dim=0), model.classifier, nn.Softmax(dim=0)]


    # calculating size #########
    a = embLayer(torch.randn((1, 224, 224, 3)))
    sizes = [[3, 244, 244]]

    for blk in networkLayers:
        for i in range(len(blk)):
            layer = blk[i]
            a = layer(a)
        sizes.append(list(a.shape))
    ###########################

    App = DnnApp('alexnet', 'alx', predictor=predictor)
    return App.instantiate(None, embLayer, networkLayers, outputBlock, sizes, inputType=InputType.D2, forward=forward)


def forward(inputs, layers):

    x = torch.tensor(inputs, dtype=torch.float32)
    for layer in layers:
        if isinstance(layer, nn.Sequential) or isinstance(layer, Link.Link) or \
                isinstance(layer, nn.Flatten) or isinstance(layer, nn.Softmax):
            x = layer(x[0] if isinstance(x, tuple) else x)
        else:
            s = (x[0] if isinstance(x, tuple) else x).shape[1]
            attention = torch.ones(1, 1, s, s, dtype=torch.long)
            x = layer(x[0] if isinstance(x, tuple) else x, attention, attention)

    return x




