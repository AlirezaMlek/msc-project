import os.path
import random
from torch.utils.data import DataLoader, Dataset
import torch
import copy
import pickle

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
        masked_txt, masked_word = mask_text(self.data[index].rstrip() + ' [SEP]', max_len)
        token_data = self.tokenizer.encode_plus(masked_txt, return_token_type_ids=True, truncation=True, max_length=64,
                                                return_attention_mask=True, padding='max_length', return_tensors='pt',
                                                add_special_tokens=True).data

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
        rand_idx = random.randint(0, min(len(split_data), max_len) - 1)
        rand_word = split_data[rand_idx]
        split_data[rand_idx] = '[MASK]'
        return ' '.join(split_data), rand_word
    else:
        return data, None



def text_data_loader(data, tokenizer, batch_size=32, shuffle=True):
    dataset = TextDataset(data, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)




"""
    train path
"""
def train(model, train_loader, val_loader, loss_fn, optimizer):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    last_step_file = './cache/last_step-' + model.name + '.pickle'
    if os.path.exists(last_step_file):
        with open(last_step_file, 'rb') as file:
            last_step = pickle.load(file)
    else:
        last_step = 0

    for epoch in range(3):
        for step, batch in enumerate(train_loader):
            if step < last_step: continue

            outputs = model(batch)
            attention_mask = torch.squeeze(batch['attention_mask'], dim=1)

            outputs = select_masked(outputs, attention_mask)

            label = {'input_ids': batch['labels'], 'attention_mask': batch['attention_mask'],
                     'token_type_ids': batch['token_type_ids']}
            label = model.forward_label(label)
            label = select_masked(label, attention_mask)

            loss = torch.tensor(0.0)
            for l, o in zip(label, outputs):
                l = torch.nn.Softmax(dim=0)(l)
                o = torch.nn.Softmax(dim=0)(o)
                loss += loss_fn(o, l)

            loss /= len(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % 10 == 0:
                print(f'Epoch {epoch}, Step {step}, Loss {loss.item()}')
                model.save_fcs()
                with open(last_step_file, 'wb') as f:
                    pickle.dump(step, f)

        # Evaluate the model on the validation set
        model.eval()
        val_loss, val_acc = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(batch)
                val_loss += loss_fn(outputs[1], batch['labels']).item()
                val_acc += (outputs[1].argmax(-1) == batch['labels']).sum().item()
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        print(f'Validation Loss {val_loss}, Validation Accuracy {val_acc}')
        model.train()



"""
    convert redundant outputs to zero by using mask
"""
def select_masked(x, mask):
    a, b = mask.shape

    zero_tensor = torch.zeros(x.shape[2])

    tmp = [torch.tensor([]) for _ in range(a)]
    for i in range(a):
        for j in range(b):
            if mask[i][j] != 0:
                tmp[i] = torch.cat((tmp[i],x[i,j]), dim=0)
    return tmp
