# Load model directly
from transformers import AutoProcessor, AutoModel
from models.Link import Link
import torch
from torch import nn
from utils.BlockNetwork import DnnApp

def create_model():
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = AutoModel.from_pretrained("openai/clip-vit-large-patch14")

    for param in model.parameters():
        param.requires_grad = False

    embLayer = model.vision_model.embeddings
    networkLayers = model.vision_model.encoder.layers
    outputBlock = [nn.Sequential()]

    def predictor(scores):
        pass

    App = DnnApp('clip-vit-large-patch14', 'cvl', predictor=predictor)
    return App.instantiate(None, embLayer, networkLayers, outputBlock, 1024, 1024,
                           forward=forward, link=Link)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def forward(x, layers):
    for layer in layers:
        s = (x[0].shape if isinstance(x, tuple) else x.shape)
        attention = torch.ones(s[0], 1, s[1], s[1], dtype=torch.long).to(device)
        x = layer(x[0] if isinstance(x, tuple) else x, attention, attention)
    return x[0] if isinstance(x, tuple) else x

