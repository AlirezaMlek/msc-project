import pickle

from models.trainer import train
from utils.BlockNetwork import *
from models import BertBase, GermanSentimentBert, FarsiBertSentiment, Link
from utils.PathMaker import create_new_path
import torch

device = torch.device('mps')

myNetwork = BlockNetwork('myNetwork')
DnnApp.network = myNetwork


path1, App1 = BertBase.create_model()


model = GermanSentimentBert
data_path = '/Users/alireza/Documents/Project/no-scare-balanced/'
# data_path = '/Users/alireza/Desktop/snappfood/'

path2, App2 = model.create_model()



path = create_new_path('test', App2, 2, 7, forward=model.forward, link=Link.Link)
tokenizer = App2.get_input_node().tokenizer


path.fetch_fc()
path.to(device)


torch.save(path, f'./cache/{path.name}/model.pth')



train_loader = model.text_data_loader(data_path+'train.txt', tokenizer, device, shuffle=False)
valid_loader = model.text_data_loader(data_path+'dev.txt', tokenizer, device, shuffle=False, batch_size=128)



optimizer = torch.optim.Adam(path.parameters(), lr=2e-6)
loss_fn = torch.nn.functional.cross_entropy


train(path, train_loader, loss_fn, optimizer, valid_loader, model.validate_model, val_stops=10)


# model.validate_model(path, valid_loader, 128)

