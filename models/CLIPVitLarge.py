# Load model directly
from transformers import AutoProcessor, AutoModel
from utils.BlockNode import InputType
import torch
from torch import nn
from utils.BlockNetwork import DnnApp, BlockNetwork
import cv2

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
                           forward=forward)


device = torch.device('mps')

def forward(x, layers):
    for layer in layers:
        s = (x[0].shape if isinstance(x, tuple) else x.shape)
        attention = torch.ones(s[0], 1, s[1], s[1], dtype=torch.long).to(device)
        x = layer(x[0] if isinstance(x, tuple) else x, attention, attention)
    return x[0] if isinstance(x, tuple) else x



# processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
#
# model = AutoModel.from_pretrained("openai/clip-vit-large-patch14")
#
# img = cv2.imread('/Users/alireza/Desktop/test.jpg')
# # image = torch.tensor(img)
#
# # img = processor.image_processor(img)
#
# inputs = processor(images=img, return_tensors="pt")
#
# batch_size = inputs.pixel_values.shape[0]
# attention_mask = torch.ones(1, 1, 257, 257, dtype=torch.long)
# causal_attn_mask = torch.ones(1, 1, 257, 257, dtype=torch.long)
#
#
# # img = torch.tensor(img['pixel_values'][0]).unsqueeze(0)
# img = model.vision_model.embeddings(inputs.pixel_values)
# # attention = torch.ones(1, 1024)
# l = model.vision_model.encoder.layers[0]
# img = l(img, attention_mask, causal_attn_mask)
# print(1)
