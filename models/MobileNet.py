import numpy as np
from utils.BlockNetwork import DnnApp
from utils.BlockNode import InputType
import torch.nn as nn
from tqdm import tqdm
from models import Link
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def create_model():
    classes = ('Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')

    def predictor(scores):
        index = int(np.argmax(scores))
        return classes[index]

    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)

    model.classifier[1] = nn.Linear(1280, 10)

    for param in model.parameters():
        param.requires_grad = False

    embLayer = nn.Sequential()

    networkLayers = [model.features[0:3], model.features[3:6], model.features[6:8],
                     model.features[8:10], model.features[10:13]]
    outputBlock = [nn.Flatten(start_dim=1), model.classifier]

    # calculating size #########
    a = embLayer(torch.randn((1, 3, 224, 224)))
    sizes = [[1, 3, 224, 224]]

    for blk in networkLayers:
        for i in range(len(blk)):
            layer = blk[i]
            a = layer(a)
        sizes.append(list(a.shape))
    ###########################

    App = DnnApp('mobilenet', 'mbn', predictor=predictor)
    return App.instantiate(None, embLayer, networkLayers, outputBlock, sizes, inputType=InputType.D2, forward=forward,
                           link=Link.Link)


def forward(x, layers):
    for layer in layers:
        x = layer(x[0] if isinstance(x, tuple) else x)

    return x[0] if isinstance(x, tuple) else x


def text_data_loader(data_file, device=None, tokenizer=None, batch_size=4, shuffle=True):
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def validate_model(model, branch, data_loader, batch='all', train=False):
    model.eval()

    all_preds = []
    all_targets = []

    num_iter = len(data_loader)
    flag = False
    with torch.no_grad():
        pbar = tqdm(enumerate(data_loader), total=num_iter, disable=train)
        for step, data in pbar:
            inputs = data[0].to(device)
            targets = data[1]

            outputs = model(inputs, branch)

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

