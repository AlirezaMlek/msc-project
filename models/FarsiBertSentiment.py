from utils.BlockNetwork import *
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertLayer
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm


"""
HooshvareLab/bert-fa-base-uncased-sentiment-snappfood refers to a specific variant of the BERT language model developed
 by HooshvareLab, an Iranian research and development company.
"""
def create_model():
    tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-base-uncased-sentiment-snappfood")
    model = AutoModelForSequenceClassification.from_pretrained("HooshvareLab/bert-fa-base-uncased-sentiment-snappfood")

    for param in model.parameters():
        param.requires_grad = False

    embLayer = model.bert.embeddings
    networkLayers = model.bert.encoder.layer
    outputBlock = [model.bert.pooler, model.classifier, torch.nn.Softmax(dim=1)]

    def predictor(scores):
        labels = ['HAPPY', 'SAD']
        return labels[torch.argmax(scores)]

    App = DnnApp('bert-fa-base-uncased-sentiment', 'fsb', predictor=predictor)
    return App.instantiate(tokenizer, embLayer, networkLayers, outputBlock, 768, 768, forward=forward,
                           forward_backup=forward_backup)


def zero_pad(data, max_len):
    l = data.shape[1]

    if l >= max_len:
        return data[0, :max_len]
    else:
        return F.pad(data, (0, max_len - l), value=0)[0]


class TextDataset(Dataset):
    def __init__(self, file_name, tokenizer, loading_batch=1000000):
        self.file_name = file_name
        self.tokenizer = tokenizer
        self.current_batch = -1
        self.num_lines = self.line_counter()
        self.data = []
        self.loading_batch = loading_batch
        self.label_tags = ['HAPPY', 'SAD']

    def __len__(self):
        return self.num_lines

    def __getitem__(self, index):
        max_len = 64
        index = self.check_index(index)
        data_sample = self.data[index].rstrip().split('\t')
        label, data_text = data_sample[3], data_sample[1]
        data_text = data_text + ' [SEP]'

        token_data = self.tokenizer(data_text, return_tensors="pt")
        token_data = {k: zero_pad(token_data[k], max_len) for k in token_data.keys()}
        label = int(label)
        one_hot = np.eye(2)[label]

        return (token_data, one_hot)

    def partial_data_loader(self):
        self.current_batch += 1
        start = self.current_batch * self.loading_batch
        end = start + self.loading_batch
        end = min(end, self.num_lines)

        with open(self.file_name, 'r') as f:
            self.data = f.readlines()[start:end]
        f.close()

    def check_index(self, index):
        bias = (self.current_batch + 1) * self.loading_batch
        if index >= len(self.data) + bias:
            self.partial_data_loader()
            index -= self.current_batch * self.loading_batch
        return index

    def line_counter(self):
        with open(self.file_name, 'r') as f:
            num_lines = sum(1 for _ in f)
        f.close()
        return num_lines



def text_data_loader(data_file, tokenizer, batch_size=32, shuffle=True):
    dataset = TextDataset(data_file, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)



def validate_model(model, data_loader):

    model.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for step, data in tqdm(enumerate(data_loader)):
            inputs = data[0]
            targets = data[1]

            outputs = model.forward(inputs)

            # Get predictions in numpy array
            preds = outputs.detach().cpu().numpy()
            all_preds.append(np.argmax(preds, 1))

            # Get targets
            targets = targets.numpy()
            all_targets.append(np.argmax(targets, 1))


    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    L = len(all_preds)
    L6 = int(L / 6)
    accuracy = np.sum(all_preds == all_targets) / len(all_targets)
    print(f'Accuracy: {accuracy}')




def forward_backup(inputs, name, App, app):
    currentNode = App[app].get_input_node()


    x = currentNode.block(inputs['input_ids'], token_type_ids=inputs['token_type_ids'])
    attention_mask = inputs['attention_mask']
    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    attention_mask = attention_mask.expand(-1, 12, -1, -1)
    while currentNode.outputGates.keys().__contains__(name):
        gate = currentNode.outputGates[name]
        currentNode = gate.nextNode
        if gate.fc is not None:
            for fc in gate.fc:
                if x[0].ndim == 3:
                    x = fc(x[0])
                else:
                    x = fc(torch.tensor(x))

        for b in currentNode.block:
            if isinstance(b, BertLayer):
                x = b(x[0] if isinstance(x, tuple) else x, attention_mask=attention_mask)
            else:
                x = b(x[0] if isinstance(x, tuple) else x)

    return x



def forward(inputs, layers):

    attention_mask = inputs['attention_mask']
    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    attention_mask = attention_mask.expand(-1, 12, -1, -1)

    x = layers[0](inputs['input_ids'], token_type_ids=inputs['token_type_ids'])
    x = torch.squeeze(x, dim=1)

    for layer in layers[1:]:
        if isinstance(layer, BertLayer):
            x = layer(x[0] if isinstance(x, tuple) else x, attention_mask=attention_mask)
        elif isinstance(layer, nn.BatchNorm1d):
            x = x.permute(0, 2, 1)
            x = layer(x)
            x = x.permute(0, 2, 1)
        else:
            x = layer(x[0] if isinstance(x, tuple) else x)

    return x