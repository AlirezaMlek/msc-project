import random
import copy
from utils.BlockNetwork import *
import torch
from transformers import BertTokenizer, BertForMaskedLM
from torch.utils.data import DataLoader, Dataset



"""
    dataset class
"""
class TextDataset(Dataset):
    def __init__(self, file_name, tokenizer, loading_batch=1000000):
        self.file_name = file_name
        self.tokenizer = tokenizer
        self.current_batch = -1
        self.num_lines = self.line_counter()
        self.data = []
        self.loading_batch = loading_batch

    def __len__(self):
        return self.num_lines

    def __getitem__(self, index):
        max_len = 64
        index = self.check_index(index)
        data = self.data[index].rstrip()
        masked_txt, masked_word = mask_text(data + ' [SEP]', max_len)
        token_data = self.tokenizer.encode_plus(masked_txt, return_token_type_ids=True, truncation=True,
                                                max_length=max_len, return_attention_mask=True, padding='max_length',
                                                return_tensors='pt', add_special_tokens=True).data

        indexed_tokens_raw = copy.copy(token_data['input_ids'])
        if masked_word is not None:
            mask_index = token_data['input_ids'][0].tolist().index(self.tokenizer.mask_token_id)
            word_token = self.tokenizer.convert_tokens_to_ids(masked_word)
            indexed_tokens_raw[0][mask_index] = word_token

        _data = {**token_data, 'labels': indexed_tokens_raw}
        return _data

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




"""
    random masking text
"""
def mask_text(data, max_len, mask_prob=0.8):
    split_data = data.split(' ')
    if random.random() < mask_prob:
        rand_idx = random.randint(0, min(len(split_data), max_len/2) - 2)
        rand_word = split_data[rand_idx]
        split_data[rand_idx] = '[MASK]'
        return ' '.join(split_data), rand_word
    else:
        return data, None



def text_data_loader(data, tokenizer, batch_size=32, shuffle=True):
    dataset = TextDataset(data, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)



def create_model():
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    for param in model.parameters():
        param.requires_grad = False

    embLayer = model.bert.embeddings
    networkLayers = model.bert.encoder.layer
    outputBlock = [model.cls]


    def predictor(scores, tokenizer, mask=None):
        scores = torch.nn.Softmax(dim=2)(scores)
        indices = torch.argmax(scores, dim=2)
        if mask is None:
            return tokenizer.convert_ids_to_tokens(indices[0][1:-1],)
        else:
            n = torch.sum(mask) - 1
            return tokenizer.convert_ids_to_tokens(indices[0][1:n],)

    App = DnnApp('bert-base-uncased', 'bbu', predictor=predictor)
    return App.instantiate(tokenizer, embLayer, networkLayers, outputBlock, 768, 768)
