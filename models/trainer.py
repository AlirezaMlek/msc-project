import os.path
import torch
import pickle
from tqdm import tqdm

"""
    train path
"""
def train(model, branch, train_loader, loss_fn, optimizer, valid_loader, validator, val_stops=3):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()
    cpu_device = torch.device('cpu')
    last_step_file = f'./cache/{model.name}/last_step.pickle'

    total_steps = len(train_loader)

    last_valid_acc = -1
    prev_valid_acc = -2
    max_valid_acc = -1

    stops = get_stops(val_stops, total_steps)

    if os.path.exists(last_step_file):
        with open(last_step_file, 'rb') as file:
            last_step = pickle.load(file)
    else:
        last_step = 0

    epoch = -1
    flag = True
    while flag:
        epoch += 1
        print('epoch', epoch)
        pbar = tqdm(enumerate(train_loader), total=total_steps)
        optimizer.param_groups[0]['lr'] /= 2
        for step, batch in pbar:
            if step < last_step: continue
            inputs = batch[0]
            label = batch[1]
            outputs = model(inputs, branch)
            # attention_mask = torch.squeeze(batch['attention_mask'], dim=1)

            # outputs = select_masked(outputs, attention_mask)

            # label = {'input_ids': batch['labels'], 'attention_mask': batch['attention_mask'],
            #          'token_type_ids': batch['token_type_ids']}
            # label = model.forward_label(label)
            # label = select_masked(label, attention_mask)


            loss = torch.tensor(0.0)
            for l, o in zip(label, outputs):
                o = o.to(cpu_device)
                loss += loss_fn(o, l)

            loss /= len(batch[1])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % 20 == 0:
                message = {'Loss': loss.item(), 'Valid acc': last_valid_acc}
                pbar.set_postfix(message)

            if step in stops:
                prev_valid_acc = last_valid_acc
                last_valid_acc = validator(model, valid_loader, train=True)

                with open(last_step_file, 'wb') as f:
                    pickle.dump(step, f)

                if max_valid_acc < last_valid_acc:
                    model.save_fcs()
                    max_valid_acc = last_valid_acc


                if last_valid_acc < prev_valid_acc and epoch >= 3:
                    print('stop training... max valid acc:  ', max_valid_acc)
                    flag = False
                    break




"""
    convert redundant outputs to zero by using mask
"""
def select_masked(x, mask):
    a, b = mask.shape

    tmp = [torch.tensor([]) for _ in range(a)]
    for i in range(a):
        for j in range(b):
            if mask[i][j] != 0:
                tmp[i] = torch.cat((tmp[i],x[i,j]), dim=0)
    return tmp


def get_stops(val_stops, total_steps):
    step_per_valid = int(total_steps / val_stops)
    stops = []
    for i in range(val_stops):
        if i < val_stops - 1:
            stop = step_per_valid * (i + 1) - 1
        else:
            stop = total_steps - 1
        stops.append(stop)

    return stops



# def optimizer(path, branch_name='main', currentNode=None, collected_params=None):
#     if currentNode is None:
#         currentNode = path.get_input_node()
#         collected_params = []
#
#     for gate in currentNode.inputGates.keys():
#         link = currentNode.inputGates[gate]
#         collected_params.append(link)
#
#     outputGates = currentNode.outputGates
#     for gate in outputGates.keys():
#         if gate in [self.name, branch_name]:
#             link = outputGates[gate].link
#             if link is not None:
#                 for param in link.parameters():
#                     param.requires_grad = require
#
#             self.link_require_grad(branch_name, require, outputGates[gate].nextNode)

