import pickle

from models.trainer import train
from utils.BlockNetwork import *
from models import BertBase, GermanSentimentBert
from utils.PathMaker import create_new_path
import torch

device = torch.device('mps')

myNetwork = BlockNetwork('myNetwork')
DnnApp.network = myNetwork


path1, App1 = BertBase.create_model()


model = GermanSentimentBert
data_path = '/Users/alireza/Documents/Project/no-scare-balanced/'

path2, App2 = model.create_model()




path = create_new_path('test', App2, 3, 7, App1, 4, 5, forward=model.forward,
                        forward_backUp=model.forward_backup)
tokenizer = App2.get_input_node().tokenizer

with open('tokenizer.pickle', 'wb') as f:
    pickle.dump(tokenizer, f)



path.fetch_fc()
path.to(device)


torch.save(path, f'./cache/{path.name}/model.pth')


# data_path = '/Users/alireza/Desktop/snappfood/'

train_loader = model.text_data_loader(data_path+'train.txt', tokenizer, shuffle=False)
valid_loader = model.text_data_loader(data_path+'dev.txt', tokenizer, shuffle=False)




optimizer = torch.optim.Adam(path.parameters(), lr=2e-5)
loss_fn = torch.nn.functional.cross_entropy

# torch.autograd.set_detect_anomaly(True)


train(path, train_loader, loss_fn, optimizer, model.validate_model, valid_loader)


