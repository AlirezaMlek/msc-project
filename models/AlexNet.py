import os.path

import numpy as np
from torchvision import models as cmodels
from utils.BlockNetwork import DnnApp
from utils.BlockNode import InputType
import torch.nn as nn
from tqdm import tqdm
from models import Link
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models import trainer
def create_model():

    classes = ('Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')

    def predictor(scores):
        index = int(np.argmax(scores))
        return classes[index]


    model = cmodels.alexnet(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier[1] = nn.Linear(9216, 4096)
    model.classifier[4] = nn.Linear(4096, 1024)
    model.classifier[6] = nn.Linear(1024, 10)

    for param in model.classifier.parameters():
        param.requires_grad = True

    embLayer = nn.Sequential()


    networkLayers = [model.features[0:3], model.features[3:6], model.features[6:8],
                     model.features[8:10], model.features[10:13]]
    outputBlock = [model.avgpool, nn.Flatten(start_dim=1), model.classifier, nn.Softmax(dim=0)]


    # calculating size #########
    a = embLayer(torch.randn((1, 3, 224, 224)))
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
                isinstance(layer, nn.Flatten) or isinstance(layer, nn.Softmax) or isinstance(layer,nn.AdaptiveAvgPool2d):
            x = layer(x[0] if isinstance(x, tuple) else x)
        else:
            s = (x[0] if isinstance(x, tuple) else x).shape[1]
            attention = torch.ones(1, 1, s, s, dtype=torch.long)
            x = layer(x[0] if isinstance(x, tuple) else x, attention, attention)

    return x




def text_data_loader(data_file, device=None, tokenizer=None, batch_size=32, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if data_file == 'train':
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    else:
        dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)



def validate_model(model, data_loader, batch='all', train=False):

    model.eval()

    all_preds = []
    all_targets = []

    num_iter = len(data_loader)
    flag = False
    with torch.no_grad():
        pbar = tqdm(enumerate(data_loader), total=num_iter, disable=train)
        for step, data in pbar:
            inputs = data[0]
            targets = data[1]

            outputs = model.forward(inputs)

            # Get predictions in numpy array
            preds = outputs.detach().cpu().numpy()
            all_preds.append(np.argmax(preds, 1))

            # Get targets
            targets = targets.numpy()
            all_targets.append(targets)

            if batch == 'all':
                continue
            else:
                if step % batch == 0 and step != 0 and flag:
                    break
                flag = True


    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    accuracy = np.sum(all_preds == all_targets) / len(all_targets)
    print(f'Accuracy: {accuracy}')
    return accuracy


