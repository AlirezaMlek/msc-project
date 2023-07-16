from models.trainer import text_data_loader, train, TextDataset
from utils.BlockNetwork import *
from models import models
from utils.PathMaker import create_new_path
from datasets import load_dataset
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tarfile
import torch
import random


myNetwork = BlockNetwork('myNetwork')
DnnApp.network = myNetwork

path1, App1 = models.create_bert_base_uncased()
path2, App2 = models.create_german_sentiment_bert()


model = create_new_path('test', App1, 3, 10, App2, 4, 4)
tokenizer = App1.get_input_node().tokenizer



model.fetch_fc()



train_loader = text_data_loader('books_large_p1.txt', tokenizer, shuffle=False)
valid_loader = text_data_loader('books_large_p2.txt', tokenizer, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
loss_fn = torch.nn.functional.cross_entropy

# torch.autograd.set_detect_anomaly(True)

train(model, train_loader, valid_loader, loss_fn, optimizer)


