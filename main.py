from models.trainer import train
from utils.BlockNetwork import *
from models import BertBase, GermanSentimentBert, FarsiBertSentiment, AlexNet, CLIPVitLarge, Link
from utils.PathMaker import create_new_path
import torch

device = torch.device('cpu')

myNetwork = BlockNetwork('myNetwork')
DnnApp.network = myNetwork


path1, App1 = BertBase.create_model()

App1.set_device(device)



model = FarsiBertSentiment

# data_path = '/Users/alireza/Documents/Project/no-scare-balanced/'
data_path = '/Users/alireza/Desktop/snappfood/'
# data_path = 'train'

path2, App2 = model.create_model()

App2.set_device(device)



path = create_new_path('test', App2, 1, 3, link=Link.Link)
tokenizer = App2.get_input_node().tokenizer

path.to(device)
path.link_require_grad('main', True)

print('Parameters with require_grad True:')
for name, param in path.named_parameters():
    if param.requires_grad:
        print(name)



torch.save(path, f'./cache/{path.name}/model.pth')



train_loader = model.text_data_loader(data_path+'train.txt', tokenizer, device, shuffle=False)
valid_loader = model.text_data_loader(data_path+'dev.txt', tokenizer, device, shuffle=False, batch_size=128)

# train_loader = model.text_data_loader('train')
# valid_loader = model.text_data_loader('valid')




optimizer = torch.optim.Adam(path.parameters(), lr=2e-4)
loss_fn = nn.CrossEntropyLoss()


train(path, 'main', train_loader, loss_fn, optimizer, valid_loader, model.validate_model, val_stops=10)


# model.validate_model(path, valid_loader, 128)

